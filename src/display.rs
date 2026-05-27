use ansi_term::Color::{Cyan, Red, White, Yellow};
use ansi_term::Style;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU16, Ordering};
use terminal_size::{terminal_size, Width};

/// Track how many lines the thinking section occupies (for clearing)
static THINKING_LINES: AtomicU16 = AtomicU16::new(0);

fn stderr() -> io::Stderr {
    io::stderr()
}

fn write_str(s: &str) {
    let _ = stderr().write_all(s.as_bytes());
    let _ = stderr().flush();
}

fn term_width() -> usize {
    terminal_size().map(|(Width(w), _)| w as usize).unwrap_or(80)
}

fn separator() -> String {
    let w = term_width();
    let label_width = 10;
    let dash_count = w.saturating_sub(label_width + 3);
    "─".repeat(dash_count)
}

/// Start streaming reasoning output (inline, dimmed italic)
pub fn start_reasoning_stream() {
    THINKING_LINES.store(0, Ordering::Relaxed);
    let label = Style::new().dimmed().paint("  Thinking │");
    write_str(&format!("\n{label} "));
}

/// Append a reasoning delta to the stream
pub fn push_reasoning_delta(text: &str) {
    // Count newlines for clearing later
    let newlines = text.matches('\n').count() as u16;
    if newlines > 0 {
        THINKING_LINES.fetch_add(newlines, Ordering::Relaxed);
    }
    let styled = Style::new().dimmed().italic().paint(text);
    write_str(&styled.to_string());
}

/// End reasoning stream and CLEAR the thinking content from terminal
pub fn end_reasoning_stream() {
    let newlines = THINKING_LINES.load(Ordering::Relaxed);
    // Layout:
    //   [previous line]
    //   \n              <- empty line from start_reasoning_stream
    //   Thinking │ ...  <- label + first content (line A)
    //   more content    <- if newlines >= 1 (line A+1)
    //   ...
    // Total lines to clear = newlines (content) + 1 (label) + 1 (empty) = newlines + 2
    // Cursor is on line A + newlines.
    // Move up to the empty line: newlines + 1
    let total_lines = newlines + 2;
    let move_up = newlines + 1;
    let mut seq = String::new();
    // Move cursor up to the empty line before "Thinking │"
    seq.push_str(&format!("\x1b[{}A", move_up));
    // Clear each line from top to bottom
    for _ in 0..total_lines {
        seq.push_str("\x1b[2K\x1b[0G");
        seq.push_str("\x1b[1B");
    }
    // Move cursor back to the start of cleared area
    seq.push_str(&format!("\x1b[{}A\x1b[0G", total_lines));
    write_str(&seq);
    THINKING_LINES.store(0, Ordering::Relaxed);
}

/// Start streaming content output
pub fn start_content_stream() {
    // Content streams directly, no special setup needed
}

/// Append a content delta to the stream (typewriter effect)
pub fn push_content_delta(text: &str) {
    write_str(text);
}

/// End content stream
pub fn end_content_stream() {
    write_str("\n");
}

pub fn print_execute(command: &str) {
    let label = Cyan.bold().paint("  Execute  │");
    let cmd = Cyan.paint(command);
    let sep = Cyan.paint(separator());
    write_str(&format!("\n{label} {sep}\n"));
    write_str(&format!("           │ {cmd}\n"));
}

pub fn print_result(content: &str, is_error: bool) {
    let max_lines = 50;
    let truncated = crate::tools::truncate_for_display(content, max_lines);

    let label = if is_error {
        Red.bold().paint("  Result   │")
    } else {
        Style::new().dimmed().paint("  Result   │")
    };
    let sep = if is_error {
        Red.paint(separator())
    } else {
        Style::new().dimmed().paint(separator())
    };
    write_str(&format!("{label} {sep}\n"));
    for line in truncated.lines() {
        if is_error {
            let styled = Red.paint(line);
            write_str(&format!("           │ {styled}\n"));
        } else {
            let styled = Style::new().dimmed().paint(line);
            write_str(&format!("           │ {styled}\n"));
        }
    }
}

pub fn print_message(message: &str) {
    if message.is_empty() {
        return;
    }
    let label = White.bold().paint("  Message  │");
    let sep = Style::new().dimmed().paint(separator());
    write_str(&format!("\n{label} {sep}\n"));
    for line in message.lines() {
        let styled = White.bold().paint(line);
        write_str(&format!("           │ {styled}\n"));
    }
}

pub fn print_error(msg: &str) {
    let prefix = Red.bold().paint("  Error");
    write_str(&format!("\n{prefix} │ {msg}\n"));
}

pub fn print_info(msg: &str) {
    let prefix = Yellow.bold().paint("  Info");
    write_str(&format!("{prefix}  │ {msg}\n"));
}

pub fn print_setting_saved(msg: &str) {
    let prefix = ansi_term::Color::Green.bold().paint("  Saved");
    write_str(&format!("{prefix}  │ {msg}\n"));
}

pub fn prompt_confirmation(command: &str) -> bool {
    let label = Yellow.bold().paint("  Confirm");
    write_str(&format!("\n{label} │ The following modify command requires your approval:\n"));
    let cmd = Yellow.paint(command);
    write_str(&format!("           │ {cmd}\n"));
    write_str("           │ Execute? [y/N] ");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap_or_default();
    input.trim().eq_ignore_ascii_case("y")
}
