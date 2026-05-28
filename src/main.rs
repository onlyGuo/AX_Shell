mod config;
mod context;
mod display;
mod protocol;
mod tools;

use serde_json::{json, Value};
use std::env;
use std::io::{self, IsTerminal, Read};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|a| a == "-s" || a == "--set") {
        handle_settings(&args);
        return;
    }

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }

    if args.iter().any(|a| a == "-v" || a == "--version") {
        println!("ax {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let user_prompt = if args.is_empty() {
        if !io::stdin().is_terminal() {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf).unwrap_or_default();
            buf.trim().to_string()
        } else {
            print_help();
            return;
        }
    } else {
        args.join(" ")
    };

    if user_prompt.is_empty() {
        print_help();
        return;
    }

    let cfg = config::Config::load();
    if cfg.api_key.is_empty() {
        display::print_error("No API key configured. Run: ax -s API_KEY=your_key");
        std::process::exit(1);
    }

    if let Err(e) = run_agent(&cfg, &user_prompt) {
        display::print_error(&e);
        std::process::exit(1);
    }
}

fn debug_log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/ax_debug.log")
    {
        let _ = writeln!(f, "[{}] {}", chrono::Local::now().format("%H:%M:%S%.3f"), msg);
    }
}

fn run_agent(cfg: &config::Config, user_prompt: &str) -> Result<(), String> {
    let system_prompt = context::build_system_prompt();

    // Build message list: history → context → current user message
    let history = context::load_history();
    let mut messages: Vec<Value> = context::history_to_messages(&history);
    messages.push(context::build_context_message());
    messages.push(json!({"role": "user", "content": user_prompt}));

    let mut all_reasoning = String::new();
    let mut all_message = String::new();
    let mut executed_commands: Vec<String> = Vec::new();
    let max_rounds = 20;

    for _round in 0..max_rounds {
        debug_log(&format!("=== Round {} ===", _round + 1));
        debug_log(&format!("Messages count: {}", messages.len()));
        for (i, msg) in messages.iter().enumerate() {
            let role = msg["role"].as_str().unwrap_or("?");
            let preview = if role == "tool" {
                format!("tool_call_id={}, content_len={}", msg["tool_call_id"].as_str().unwrap_or(""), msg["content"].as_str().map(|s| s.len()).unwrap_or(0))
            } else if role == "assistant" {
                let tc_count = msg["tool_calls"].as_array().map(|a| a.len()).unwrap_or(0);
                format!("content_len={}, tool_calls={}", msg["content"].as_str().map(|s| s.len()).unwrap_or(0), tc_count)
            } else {
                format!("content_len={}", msg["content"].as_str().map(|s| s.len()).unwrap_or(0))
            };
            debug_log(&format!("  msg[{}]: role={} {}", i, role, preview));
        }

        let mut has_reasoning = false;
        let mut has_content = false;

        let events = protocol::call_llm_streaming(cfg, &messages, &system_prompt, |event| {
            match event {
                protocol::StreamEvent::ReasoningDelta(text) => {
                    if !has_reasoning {
                        display::start_reasoning_stream();
                        has_reasoning = true;
                    }
                    display::push_reasoning_delta(text);
                    all_reasoning.push_str(text);
                }
                protocol::StreamEvent::ContentDelta(text) => {
                    if !has_content {
                        if has_reasoning {
                            display::end_reasoning_stream();
                            has_reasoning = false;
                        }
                        display::start_content_stream();
                        has_content = true;
                    }
                    display::push_content_delta(text);
                    all_message.push_str(text);
                }
                protocol::StreamEvent::Done(_) => {
                    if has_reasoning && !has_content {
                        display::end_reasoning_stream();
                        has_reasoning = false;
                    }
                }
                protocol::StreamEvent::Error(_) => {
                    if has_reasoning {
                        display::end_reasoning_stream();
                        has_reasoning = false;
                    }
                    if has_content {
                        display::end_content_stream();
                        has_content = false;
                    }
                }
            }
        })?;

        if has_content {
            display::end_content_stream();
        }

        // Extract the final response from the Done event
        let resp = events
            .iter()
            .find_map(|e| match e {
                protocol::StreamEvent::Done(r) => Some(r.clone()),
                _ => None,
            })
            .ok_or("No response from LLM")?;

        // Check for errors
        if let Some(err) = events.iter().find_map(|e| match e {
            protocol::StreamEvent::Error(msg) => Some(msg.clone()),
            _ => None,
        }) {
            return Err(err);
        }

        debug_log(&format!("Response: message_len={}, tool_calls={}", resp.message.len(), resp.tool_calls.len()));
        for tc in &resp.tool_calls {
            debug_log(&format!("  tool_call: id={}, name={}, args={}", tc.id, tc.name, tc.arguments));
        }

        // If no tool calls, we're done
        if resp.tool_calls.is_empty() {
            // Show message only if it adds value (not just echoing the result)
            if !resp.message.is_empty() && !has_content {
                display::print_message(&resp.message);
            }
            context::append_history(user_prompt, &all_message, &all_reasoning, executed_commands.clone());
            return Ok(());
        }

        // Add assistant message to conversation
        for raw_msg in &resp.raw_messages {
            debug_log(&format!("  raw_msg: {}", serde_json::to_string(raw_msg).unwrap_or_default()));
            messages.push(raw_msg.clone());
        }

        // Process tool calls
        for tool_call in &resp.tool_calls {
            let display_cmd = match tool_call.name.as_str() {
                "execute_command" | "execute_command_with_timeout" => {
                    tool_call.arguments["command"]
                        .as_str()
                        .unwrap_or("")
                        .to_string()
                }
                "write_file" => {
                    let path = tool_call.arguments["path"].as_str().unwrap_or("");
                    format!("write file: {path}")
                }
                _ => format!("{}({})", tool_call.name, tool_call.arguments),
            };

            display::print_execute(&display_cmd);
            let result = tools::execute_tool_call(tool_call);
            display::print_result(&result.content, result.is_error);
            executed_commands.push(display_cmd.clone());

            let context_content = tools::truncate_for_context(&result.content, 4000);

            let tool_msg = match cfg.protocol.as_str() {
                "anthropic_message" => json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": context_content,
                    "is_error": result.is_error,
                }),
                _ => json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": context_content,
                }),
            };
            debug_log(&format!("  tool_result: {}", serde_json::to_string(&tool_msg).unwrap_or_default()));
            messages.push(tool_msg);
        }
    }

    display::print_error("Maximum tool call rounds exceeded (20). Stopping.");
    context::append_history(user_prompt, &all_message, &all_reasoning, executed_commands);
    Ok(())
}

fn handle_settings(args: &[String]) {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-s" || args[i] == "--set" {
            i += 1;
            if i >= args.len() {
                display::print_error("Missing value for -s flag");
                return;
            }
            let setting = &args[i];
            if let Some((key, value)) = setting.split_once('=') {
                match config::apply_setting(key, value) {
                    Ok(msg) => display::print_setting_saved(&msg),
                    Err(e) => display::print_error(&e),
                }
            } else {
                display::print_error(&format!("Invalid format: {setting}. Use KEY=VALUE"));
            }
        }
        i += 1;
    }
}

fn print_help() {
    eprintln!(
        r#"
  {title}

  {usage}
    ax <command>              Execute a natural language command
    ax -s KEY=VALUE           Configure settings
    ax -v                     Show version
    ax -h                     Show this help

  {settings}
    ax -s API_KEY=<key>       Set LLM API key
    ax -s BASE_URL=<url>      Set LLM API base URL
    ax -s PROTOCOL=<proto>    Set protocol: open_chat | openai_response | anthropic_message
    ax -s MODEL=<model>       Set model name

  {examples}
    ax 列出当前目录内容
    ax 查看系统内存使用情况
    ax -s API_KEY=sk-xxx -s BASE_URL=https://api.deepseek.com/v1
    ax -s PROTOCOL=open_chat
"#,
        title = ansi_term::Color::Cyan.bold().paint("AX - AI Command Agent"),
        usage = ansi_term::Color::Yellow.bold().paint("Usage:"),
        settings = ansi_term::Color::Yellow.bold().paint("Settings:"),
        examples = ansi_term::Color::Yellow.bold().paint("Examples:"),
    );
}
