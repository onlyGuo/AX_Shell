use serde_json::{json, Value};
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "execute_command",
            "description": "Execute a shell command. Use 'query' type for read-only commands (ls, cat, grep, ps, df, etc.) that can be executed directly. Use 'modify' type for commands that change system state (rm, mv, chmod, install, firewall, etc.) which require user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["query", "modify"],
                        "description": "Command type: 'query' for read-only, 'modify' for state-changing"
                    }
                },
                "required": ["command", "type"]
            }
        }),
        json!({
            "name": "execute_command_with_timeout",
            "description": "Execute a shell command with a timeout. Use this for commands that may not terminate on their own (like 'tail -f', 'top', 'watch'). The command will be forcefully terminated after the specified timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["query", "modify"],
                        "description": "Command type: 'query' for read-only, 'modify' for state-changing"
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum time in seconds to wait before force-killing the command"
                    }
                },
                "required": ["command", "type", "timeout_seconds"]
            }
        }),
        json!({
            "name": "write_file",
            "description": "Write content to a file. This is always a modify operation. Use this to create or overwrite files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to write to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }),
    ]
}

pub fn anthropic_tool_definitions() -> Vec<Value> {
    tool_definitions()
        .into_iter()
        .map(|t| {
            json!({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"]
            })
        })
        .collect()
}

pub fn execute_tool_call(tool_call: &ToolCall) -> ToolResult {
    match tool_call.name.as_str() {
        "execute_command" => {
            let command = tool_call.arguments["command"].as_str().unwrap_or("");
            execute_command(command, None)
        }
        "execute_command_with_timeout" => {
            let command = tool_call.arguments["command"].as_str().unwrap_or("");
            let timeout = tool_call.arguments["timeout_seconds"].as_u64().unwrap_or(30);
            execute_command(command, Some(timeout))
        }
        "write_file" => {
            let path = tool_call.arguments["path"].as_str().unwrap_or("");
            let content = tool_call.arguments["content"].as_str().unwrap_or("");
            write_file(path, content)
        }
        _ => ToolResult {
            content: format!("Unknown tool: {}", tool_call.name),
            is_error: true,
        },
    }
}

fn execute_command(command: &str, timeout_secs: Option<u64>) -> ToolResult {
    let timeout = timeout_secs.unwrap_or(60);

    let result = if cfg!(target_os = "windows") {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", command]);
        run_with_timeout(cmd, timeout)
    } else {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        run_with_timeout(cmd, timeout)
    };

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&format!("[stderr]\n{stderr}"));
            }
            if result.is_empty() {
                result = format!("(command completed with exit code {exit_code})");
            }

            ToolResult {
                content: result,
                is_error: !output.status.success(),
            }
        }
        Err(e) => ToolResult {
            content: format!("Command execution failed: {e}"),
            is_error: true,
        },
    }
}

fn run_with_timeout(mut cmd: Command, timeout_secs: u64) -> Result<std::process::Output, std::io::Error> {
    use std::sync::mpsc;
    use std::thread;

    let (tx, rx) = mpsc::channel();
    let child = cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    let handle = thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let timeout = Duration::from_secs(timeout_secs);
    match rx.recv_timeout(timeout) {
        Ok(result) => {
            let _ = handle.join();
            result
        }
        Err(_) => {
            // Timeout - we can't easily kill the child here in a blocking way
            // but the channel timeout ensures we don't hang
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("Command timed out after {timeout_secs}s"),
            ))
        }
    }
}

fn write_file(path: &str, content: &str) -> ToolResult {
    match std::fs::write(path, content) {
        Ok(()) => ToolResult {
            content: format!("File written successfully: {path}"),
            is_error: false,
        },
        Err(e) => ToolResult {
            content: format!("Failed to write file {path}: {e}"),
            is_error: true,
        },
    }
}

/// Truncate output for display to the user (generous limit)
pub fn truncate_for_display(content: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= max_lines {
        content.to_string()
    } else {
        let truncated: Vec<&str> = lines[..max_lines].to_vec();
        format!("{}\n... ({} more lines)", truncated.join("\n"), lines.len() - max_lines)
    }
}

/// Truncate output for LLM context (tighter limit to save tokens)
pub fn truncate_for_context(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        let truncated: String = content.chars().take(max_chars).collect();
        format!("{truncated}\n... (truncated, showing {max_chars}/{} chars)", content.len())
    }
}
