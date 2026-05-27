use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::io::Write as _;
use std::sync::mpsc;

use crate::config::Config;
use crate::tools::{anthropic_tool_definitions, tool_definitions, ToolCall};

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub message: String,
    #[allow(dead_code)]
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub raw_messages: Vec<Value>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    ReasoningDelta(String),
    ContentDelta(String),
    Done(LLMResponse),
    Error(String),
}

fn debug_log(msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/ax_debug.log")
    {
        let _ = writeln!(f, "[{}] {}", chrono::Local::now().format("%H:%M:%S%.3f"), msg);
    }
}

/// Try to extract Anthropic-style tool_use blocks from a Value (string or array).
/// Returns (text_content, tool_calls) if tool_use blocks found.
fn extract_tool_use_from_value(content_val: &Value) -> Option<(String, Vec<ToolCall>)> {
    let arr = match content_val {
        Value::Array(a) => a.clone(),
        Value::String(s) => {
            let parsed: Value = serde_json::from_str(s).ok()?;
            parsed.as_array()?.clone()
        }
        _ => return None,
    };
    let mut tool_calls = Vec::new();
    let mut text_parts = Vec::new();
    for block in &arr {
        match block["type"].as_str() {
            Some("tool_use") => {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                let input = block["input"].clone();
                debug_log(&format!("Detected Anthropic tool_use: id={id}, name={name}"));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            Some("text") => {
                if let Some(text) = block["text"].as_str() {
                    text_parts.push(text.to_string());
                }
            }
            _ => {}
        }
    }
    if tool_calls.is_empty() {
        None
    } else {
        Some((text_parts.join(""), tool_calls))
    }
}

/// Build an OpenAI-format assistant message with tool_calls.
/// Always includes `reasoning_content` field (empty string if no reasoning),
/// as MiMo models require it in multi-turn agent conversations.
fn build_openai_assistant_msg(
    content: &str,
    reasoning: &str,
    tool_calls: &[ToolCall],
) -> Value {
    let mut msg = json!({
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
    });
    if !tool_calls.is_empty() {
        msg["tool_calls"] = json!(tool_calls.iter().map(|tc| {
            json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default()
                }
            })
        }).collect::<Vec<_>>());
    }
    msg
}

pub fn call_llm_streaming(
    config: &Config,
    messages: &[Value],
    system_prompt: &str,
    mut on_event: impl FnMut(&StreamEvent),
) -> Result<Vec<StreamEvent>, String> {
    let (tx, rx) = mpsc::channel();

    let config = config.clone();
    let messages = messages.to_vec();
    let system_prompt = system_prompt.to_string();

    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        rt.block_on(async {
            let result = match config.protocol.as_str() {
                "open_chat" => stream_open_chat(&config, &messages, &system_prompt, &tx).await,
                "openai_response" => stream_openai_response(&config, &messages, &system_prompt, &tx).await,
                "anthropic_message" => stream_anthropic(&config, &messages, &system_prompt, &tx).await,
                _ => Err(format!("Unknown protocol: {}", config.protocol)),
            };
            // Send the final result (Done event or Error)
            match result {
                Ok(final_event) => { let _ = tx.send(final_event); }
                Err(e) => { let _ = tx.send(StreamEvent::Error(e)); }
            }
        });
    });

    // Process events in real-time as they arrive from the channel
    let mut events = Vec::new();
    while let Ok(event) = rx.recv() {
        on_event(&event);
        let is_done = matches!(event, StreamEvent::Done(_) | StreamEvent::Error(_));
        events.push(event);
        if is_done {
            break;
        }
    }

    let _ = handle.join();
    Ok(events)
}

async fn process_sse_stream(
    mut stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
    events: &mut Vec<StreamEvent>,
    mut on_line: impl FnMut(&str, &mut Vec<StreamEvent>) -> Result<(), String>,
) -> Result<(), String> {
    let mut buffer = String::new();

    let mut chunk_count = 0u32;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream read error: {e}"))?;
        chunk_count += 1;
        let text = String::from_utf8_lossy(&chunk);
        debug_log(&format!("SSE chunk #{chunk_count}: {} bytes, preview={}", chunk.len(), &text.chars().take(200).collect::<String>()));
        buffer.push_str(&text);

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim().to_string();
            buffer = buffer[newline_pos + 1..].to_string();
            if line.is_empty() {
                continue;
            }
            let prev_len = events.len();
            on_line(&line, events)?;
            // Log newly added events
            for event in &events[prev_len..] {
                match event {
                    StreamEvent::ReasoningDelta(d) => debug_log(&format!("  -> ReasoningDelta: {} chars", d.len())),
                    StreamEvent::ContentDelta(d) => debug_log(&format!("  -> ContentDelta: {} chars", d.len())),
                    StreamEvent::Done(_) => debug_log("  -> Done"),
                    StreamEvent::Error(e) => debug_log(&format!("  -> Error: {e}")),
                }
            }
            if events.last().map_or(false, |e| matches!(e, StreamEvent::Done(_))) {
                return Ok(());
            }
        }
    }

    let remaining = buffer.trim().to_string();
    if !remaining.is_empty() {
        on_line(&remaining, events)?;
    }

    Ok(())
}

// ─── OpenAI Chat Completions Streaming ──────────────────────────────────────

async fn stream_open_chat(
    config: &Config,
    messages: &[Value],
    system_prompt: &str,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<StreamEvent, String> {
    let client = Client::new();
    let url = format!(
        "{}/chat/completions",
        config.effective_base_url().trim_end_matches('/')
    );

    let mut api_messages: Vec<Value> = vec![json!({"role": "system", "content": system_prompt})];
    api_messages.extend_from_slice(messages);

    let body = json!({
        "model": config.effective_model(),
        "messages": api_messages,
        "tools": tool_definitions().into_iter().map(|t| {
            json!({"type": "function", "function": t})
        }).collect::<Vec<_>>(),
        "tool_choice": "auto",
        "stream": true,
    });

    debug_log(&format!("Request URL: {url}"));
    debug_log(&format!("Request body: {}", serde_json::to_string(&body).unwrap_or_default()));

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    let status = resp.status();
    debug_log(&format!("Response status: {status}"));
    debug_log(&format!("Response headers: {:?}", resp.headers()));
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error ({status}): {text}"));
    }

    let stream = resp.bytes_stream();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls_map: std::collections::HashMap<usize, (String, String, String)> =
        std::collections::HashMap::new();
    let mut events: Vec<StreamEvent> = Vec::new();

    process_sse_stream(stream, &mut events, |line, events| {
        if !line.starts_with("data: ") {
            return Ok(());
        }
        let data = &line[6..];
        if data == "[DONE]" {
            return Ok(());
        }
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        if let Some(choices) = chunk["choices"].as_array() {
            for choice in choices {
                let delta = &choice["delta"];
                if let Some(rc) = delta["reasoning_content"].as_str() {
                    if !rc.is_empty() {
                        reasoning.push_str(rc);
                        let event = StreamEvent::ReasoningDelta(rc.to_string());
                        let _ = tx.send(event.clone());
                        events.push(event);
                    }
                }
                if let Some(c) = delta["content"].as_str() {
                    if !c.is_empty() {
                        content.push_str(c);
                        let event = StreamEvent::ContentDelta(c.to_string());
                        let _ = tx.send(event.clone());
                        events.push(event);
                    }
                }
                if let Some(calls) = delta["tool_calls"].as_array() {
                    for call in calls {
                        let index = call["index"].as_u64().unwrap_or(0) as usize;
                        let entry = tool_calls_map.entry(index).or_default();
                        if let Some(id) = call["id"].as_str() {
                            if !id.is_empty() {
                                entry.0 = id.to_string();
                            }
                        }
                        if let Some(name) = call["function"]["name"].as_str() {
                            if !name.is_empty() {
                                entry.1 = name.to_string();
                            }
                        }
                        if let Some(args) = call["function"]["arguments"].as_str() {
                            entry.2.push_str(args);
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .await?;

    // Build tool_calls from streaming deltas
    let mut indices: Vec<usize> = tool_calls_map.keys().copied().collect();
    indices.sort();
    let mut tool_calls: Vec<ToolCall> = indices
        .iter()
        .filter_map(|i| {
            tool_calls_map.get(i).map(|(id, name, args_str)| {
                debug_log(&format!("tool_call_map[{i}]: id={id}, name={name}, args_str={args_str}"));
                let arguments: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments,
                }
            })
        })
        .collect();

    // Fallback: detect Anthropic-style tool_use blocks in content string
    if tool_calls.is_empty() {
        if let Some((text, extracted)) = extract_tool_use_from_value(&Value::String(content.clone())) {
            tool_calls = extracted;
            content = text;
        }
    }

    let assistant_msg = build_openai_assistant_msg(&content, &reasoning, &tool_calls);
    debug_log(&format!("Final assistant_msg: {}", serde_json::to_string(&assistant_msg).unwrap_or_default()));

    Ok(StreamEvent::Done(LLMResponse {
        message: content,
        reasoning,
        tool_calls,
        raw_messages: vec![assistant_msg],
    }))
}

// ─── OpenAI Responses API Streaming ─────────────────────────────────────────

async fn stream_openai_response(
    config: &Config,
    messages: &[Value],
    system_prompt: &str,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<StreamEvent, String> {
    let client = Client::new();
    let url = format!(
        "{}/responses",
        config.effective_base_url().trim_end_matches('/')
    );

    let mut input: Vec<Value> = vec![json!({"role": "developer", "content": system_prompt})];
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        if role == "tool" {
            let call_id = msg["tool_call_id"].as_str().unwrap_or("");
            let content = msg["content"].as_str().unwrap_or("");
            input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            }));
        } else if role == "assistant" {
            // Reconstruct function_call items from tool_calls field
            if let Some(calls) = msg["tool_calls"].as_array() {
                for call in calls {
                    let id = call["id"].as_str().unwrap_or("");
                    let name = call["function"]["name"].as_str().unwrap_or("");
                    let args_str = call["function"]["arguments"].as_str().unwrap_or("{}");
                    let arguments: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    input.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": serde_json::to_string(&arguments).unwrap_or_default(),
                    }));
                }
            }
            // Handle reasoning_content from OpenAI format
            if let Some(rc) = msg["reasoning_content"].as_str() {
                if !rc.is_empty() {
                    input.push(json!({
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": rc}],
                    }));
                }
            }
            // Handle Anthropic thinking blocks in content array
            if let Some(arr) = msg["content"].as_array() {
                for block in arr {
                    if block["type"].as_str() == Some("thinking") {
                        if let Some(thinking) = block["thinking"].as_str() {
                            input.push(json!({
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": thinking}],
                            }));
                        }
                    } else if block["type"].as_str() == Some("text") {
                        if let Some(text) = block["text"].as_str() {
                            if !text.is_empty() {
                                input.push(json!({"role": "assistant", "content": text}));
                            }
                        }
                    }
                }
            }
            if let Some(content) = msg["content"].as_str() {
                if !content.is_empty() {
                    input.push(json!({"role": "assistant", "content": content}));
                }
            }
        } else {
            input.push(msg.clone());
        }
    }

    let body = json!({
        "model": config.effective_model(),
        "input": input,
        "tools": tool_definitions().into_iter().map(|t| {
            json!({
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            })
        }).collect::<Vec<_>>(),
        "stream": true,
    });

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error ({status}): {text}"));
    }

    let stream = resp.bytes_stream();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut args_map: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut events: Vec<StreamEvent> = Vec::new();

    process_sse_stream(stream, &mut events, |line, events| {
        if !line.starts_with("data: ") {
            return Ok(());
        }
        let data = &line[6..];
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let event_type = chunk["type"].as_str().unwrap_or("");

        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = chunk["delta"].as_str() {
                    content.push_str(delta);
                    let event = StreamEvent::ContentDelta(delta.to_string());
                    let _ = tx.send(event.clone());
                    events.push(event);
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = chunk["delta"].as_str() {
                    reasoning.push_str(delta);
                    let event = StreamEvent::ReasoningDelta(delta.to_string());
                    let _ = tx.send(event.clone());
                    events.push(event);
                }
            }
            "response.function_call_arguments.delta" => {
                let index = chunk["output_index"].as_u64().unwrap_or(0) as usize;
                if let Some(delta) = chunk["delta"].as_str() {
                    args_map.entry(index).or_default().push_str(delta);
                }
            }
            "response.output_item.done" => {
                let item = &chunk["item"];
                if item["type"].as_str() == Some("function_call") {
                    let id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let index = chunk["output_index"].as_u64().unwrap_or(0) as usize;
                    let args_str = args_map
                        .remove(&index)
                        .filter(|s| !s.is_empty())
                        .or_else(|| item["arguments"].as_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| "{}".to_string());
                    let arguments: Value = serde_json::from_str(&args_str).unwrap_or(json!({}));
                    tool_calls.push(ToolCall { id, name, arguments });
                }
            }
            "response.completed" => {
                let output = &chunk["response"]["output"];
                if let Some(arr) = output.as_array() {
                    for item in arr {
                        match item["type"].as_str() {
                            Some("reasoning") => {
                                if reasoning.is_empty() {
                                    if let Some(summary) = item["summary"].as_array() {
                                        for s in summary {
                                            if let Some(text) = s["text"].as_str() {
                                                reasoning.push_str(text);
                                            }
                                        }
                                    }
                                }
                            }
                            Some("message") => {
                                if content.is_empty() {
                                    if let Some(content_arr) = item["content"].as_array() {
                                        for c in content_arr {
                                            if c["type"].as_str() == Some("output_text") {
                                                if let Some(text) = c["text"].as_str() {
                                                    content.push_str(text);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Some("function_call") => {
                                if tool_calls.is_empty() {
                                    let id = item["call_id"].as_str().unwrap_or("").to_string();
                                    let name = item["name"].as_str().unwrap_or("").to_string();
                                    let args_str = item["arguments"].as_str().unwrap_or("{}");
                                    let arguments: Value =
                                        serde_json::from_str(args_str).unwrap_or(json!({}));
                                    tool_calls.push(ToolCall { id, name, arguments });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    })
    .await?;

    // Fallback: detect Anthropic-style tool_use blocks in content string
    if tool_calls.is_empty() {
        if let Some((text, extracted)) = extract_tool_use_from_value(&Value::String(content.clone())) {
            tool_calls = extracted;
            content = text;
        }
    }

    // Build assistant message in Responses API format (function_call items)
    let mut assistant_items: Vec<Value> = Vec::new();
    if !reasoning.is_empty() {
        assistant_items.push(json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}],
        }));
    }
    if !content.is_empty() {
        assistant_items.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }));
    }
    for tc in &tool_calls {
        assistant_items.push(json!({
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.name,
            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
        }));
    }

    // For the raw_messages, we use the Responses API input format
    // These items go directly into the next request's input array
    Ok(StreamEvent::Done(LLMResponse {
        message: content.trim().to_string(),
        reasoning: reasoning.trim().to_string(),
        tool_calls,
        raw_messages: assistant_items,
    }))
}

// ─── Anthropic Messages Streaming ───────────────────────────────────────────

async fn stream_anthropic(
    config: &Config,
    messages: &[Value],
    system_prompt: &str,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<StreamEvent, String> {
    let client = Client::new();
    let url = format!(
        "{}/v1/messages",
        config.effective_base_url().trim_end_matches('/')
    );

    let mut api_messages: Vec<Value> = Vec::new();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        match role {
            "assistant" => {
                let mut content_blocks: Vec<Value> = Vec::new();
                // Check tool_calls field (OpenAI format)
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for call in calls {
                        let id = call["id"].as_str().unwrap_or("");
                        let name = call["function"]["name"].as_str().unwrap_or("");
                        let args_str = call["function"]["arguments"].as_str().unwrap_or("{}");
                        let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                        content_blocks.push(json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input,
                        }));
                    }
                }
                // Also check content for Anthropic-style tool_use blocks
                if let Some(text) = msg["content"].as_str() {
                    if !text.is_empty() {
                        if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                            if let Some(arr) = parsed.as_array() {
                                for block in arr {
                                    if block["type"].as_str() == Some("tool_use")
                                        || block["type"].as_str() == Some("text")
                                    {
                                        content_blocks.push(block.clone());
                                    }
                                }
                            }
                        } else {
                            content_blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                }
                // Also check if content is already an array (Anthropic native format)
                if let Some(arr) = msg["content"].as_array() {
                    for block in arr {
                        if block["type"].as_str() == Some("tool_use")
                            || block["type"].as_str() == Some("text")
                            || block["type"].as_str() == Some("thinking")
                        {
                            content_blocks.push(block.clone());
                        }
                    }
                }
                // Handle reasoning_content from OpenAI format
                if let Some(rc) = msg["reasoning_content"].as_str() {
                    if !rc.is_empty() {
                        content_blocks.push(json!({"type": "thinking", "thinking": rc}));
                    }
                }
                if !content_blocks.is_empty() {
                    api_messages.push(json!({"role": "assistant", "content": content_blocks}));
                }
            }
            "tool" => {
                let tool_call_id = msg["tool_call_id"].as_str().unwrap_or("");
                let content = msg["content"].as_str().unwrap_or("");
                let is_error = msg.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                api_messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                        "is_error": is_error,
                    }],
                }));
            }
            _ => {
                api_messages.push(msg.clone());
            }
        }
    }

    let body = json!({
        "model": config.effective_model(),
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": api_messages,
        "tools": anthropic_tool_definitions(),
        "stream": true,
    });

    let resp = client
        .post(&url)
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error ({status}): {text}"));
    }

    let stream = resp.bytes_stream();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut raw_content_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut events: Vec<StreamEvent> = Vec::new();

    let mut current_block_type: Option<String> = None;
    let mut current_tool_id = String::new();
    let mut current_tool_name = String::new();
    let mut current_tool_input = String::new();
    let mut current_text_block_index: Option<usize> = None;

    process_sse_stream(stream, &mut events, |line, events| {
        if line.starts_with("event: ") {
            return Ok(());
        }
        if !line.starts_with("data: ") {
            return Ok(());
        }
        let data = &line[6..];
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let msg_type = chunk["type"].as_str().unwrap_or("");

        match msg_type {
            "content_block_start" => {
                let block = &chunk["content_block"];
                let block_type = block["type"].as_str().unwrap_or("");
                current_block_type = Some(block_type.to_string());
                match block_type {
                    "thinking" => {}
                    "text" => {
                        current_text_block_index = Some(raw_content_blocks.len());
                        raw_content_blocks.push(json!({"type": "text", "text": ""}));
                    }
                    "tool_use" => {
                        current_tool_id = block["id"].as_str().unwrap_or("").to_string();
                        current_tool_name = block["name"].as_str().unwrap_or("").to_string();
                        current_tool_input.clear();
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let delta = &chunk["delta"];
                let delta_type = delta["type"].as_str().unwrap_or("");
                match delta_type {
                    "thinking_delta" => {
                        if let Some(thinking) = delta["thinking"].as_str() {
                            reasoning.push_str(thinking);
                            let event = StreamEvent::ReasoningDelta(thinking.to_string());
                            let _ = tx.send(event.clone());
                            events.push(event);
                        }
                    }
                    "text_delta" => {
                        if let Some(text) = delta["text"].as_str() {
                            content.push_str(text);
                            if let Some(idx) = current_text_block_index {
                                if let Some(block) = raw_content_blocks.get_mut(idx) {
                                    if let Some(existing) = block["text"].as_str() {
                                        let mut s = existing.to_string();
                                        s.push_str(text);
                                        block["text"] = json!(s);
                                    }
                                }
                            }
                            let event = StreamEvent::ContentDelta(text.to_string());
                            let _ = tx.send(event.clone());
                            events.push(event);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(json_str) = delta["partial_json"].as_str() {
                            current_tool_input.push_str(json_str);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                match current_block_type.as_deref() {
                    Some("tool_use") => {
                        let arguments: Value = serde_json::from_str(&current_tool_input)
                            .unwrap_or(json!({}));
                        tool_calls.push(ToolCall {
                            id: current_tool_id.clone(),
                            name: current_tool_name.clone(),
                            arguments: arguments.clone(),
                        });
                        raw_content_blocks.push(json!({
                            "type": "tool_use",
                            "id": current_tool_id,
                            "name": current_tool_name,
                            "input": arguments,
                        }));
                        current_tool_id.clear();
                        current_tool_name.clear();
                        current_tool_input.clear();
                    }
                    _ => {}
                }
                current_block_type = None;
                current_text_block_index = None;
            }
            "message_stop" => {}
            "error" => {
                let err_msg = chunk["error"]["message"].as_str().unwrap_or("Unknown error");
                let event = StreamEvent::Error(err_msg.to_string());
                let _ = tx.send(event.clone());
                events.push(event);
            }
            _ => {}
        }
        Ok(())
    })
    .await?;

    // Build assistant message in Anthropic native format (content blocks)
    let assistant_msg = json!({
        "role": "assistant",
        "content": raw_content_blocks,
    });

    Ok(StreamEvent::Done(LLMResponse {
        message: content.trim().to_string(),
        reasoning: reasoning.trim().to_string(),
        tool_calls,
        raw_messages: vec![assistant_msg],
    }))
}
