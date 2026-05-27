use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub timestamp: String,
    pub user: String,
    pub assistant: String,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub tools_used: Vec<String>,
}

fn history_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ax")
        .join("history.json")
}

pub fn load_history() -> Vec<HistoryEntry> {
    let path = history_path();
    if path.exists() {
        let data = fs::read_to_string(&path).unwrap_or_default();
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    }
}

pub fn save_history(entries: &[HistoryEntry]) {
    let path = history_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(data) = serde_json::to_string_pretty(entries) {
        let _ = fs::write(&path, data);
    }
}

pub fn append_history(user: &str, assistant: &str, reasoning: &str, tools_used: Vec<String>) {
    let mut history = load_history();
    history.push(HistoryEntry {
        timestamp: Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
        user: user.to_string(),
        assistant: assistant.to_string(),
        reasoning: reasoning.to_string(),
        tools_used,
    });
    // Keep last 100 entries
    let len = history.len();
    if len > 100 {
        history = history[len - 100..].to_vec();
    }
    save_history(&history);
}

fn system_info() -> String {
    let info = os_info::get();
    let os_type = format!("{}", info.os_type());
    let version = format!("{}", info.version());
    let arch = std::env::consts::ARCH;
    format!("{os_type} {version} ({arch})")
}

fn current_time() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn current_dir() -> String {
    env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn hostname_str() -> String {
    hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Build the fully static system prompt (rules + environment).
/// This part never changes within a session, maximizing cache hit.
pub fn build_system_prompt() -> String {
    format!(
        r#"You are AX, a system command execution agent. Your job is to help users execute system commands by understanding their natural language instructions.

## Rules
1. You MUST use tools to execute commands. Do NOT just describe what to do.
2. For each command, you MUST specify the type:
   - "query": Read-only commands (ls, cat, grep, ps, df, etc.) - execute directly
   - "modify": Commands that change system state (rm, mv, chmod, apt install, firewall rules, etc.) - requires user confirmation
3. When a task needs multiple steps, chain tool calls. For example, to add a firewall port:
   - First check firewall status (query)
   - Check if port exists (query)
   - Add the port (modify, needs confirmation)
4. Use `execute_command_with_timeout` for commands that may not terminate (tail -f, top, watch, etc.)
5. For file creation/modification, use `write_file` tool (always modify type).
6. Keep responses concise. If the command output is long, summarize key points only.
7. Respond in the same language as the user.

## Output Guidelines
- Do NOT repeat command output in your message if the user can already see the Execute Result.
- Only provide summary messages when they add value (analysis, interpretation, next steps).
- If the result is self-explanatory (like listing files), a brief note or no message is fine.

## Current Environment
- System: {system}
- Hostname: {hostname}"#,
        system = system_info(),
        hostname = hostname_str(),
    )
}

/// Build a dynamic context message (time, cwd, execution note).
/// Injected right before the current user message.
/// Each tool command runs in a separate shell session, so `cd` only affects
/// that single command. To run in another directory, use `cd dir && cmd` or full paths.
pub fn build_context_message() -> Value {
    let content = format!(
        "Current Time: {time}\n\
         Current Directory: {cwd}\n\
         Note: Each command you execute runs in an independent shell session under the current directory. \
         Even if you run `cd /some/dir`, the next command will still start from the original current directory. \
         To execute a command in another directory, use `cd /some/dir && your_command` or use absolute paths.",
        time = current_time(),
        cwd = current_dir(),
    );
    json!({"role": "user", "content": content})
}

/// Convert last 10 history entries into proper conversation message pairs.
/// Each entry becomes a user message + assistant message.
/// Tool usage is briefly noted in the assistant message.
pub fn history_to_messages(history: &[HistoryEntry]) -> Vec<Value> {
    let recent: Vec<&HistoryEntry> = history.iter().rev().take(10).collect();
    let mut messages = Vec::new();

    for entry in recent.into_iter().rev() {
        messages.push(json!({"role": "user", "content": entry.user}));

        let mut assistant_content = String::new();
        if !entry.tools_used.is_empty() {
            assistant_content.push_str(&format!("[Executed: {}]\n", entry.tools_used.join("; ")));
        }
        assistant_content.push_str(if entry.assistant.is_empty() {
            "(no response)"
        } else {
            &entry.assistant
        });
        messages.push(json!({"role": "assistant", "content": assistant_content}));
    }

    messages
}
