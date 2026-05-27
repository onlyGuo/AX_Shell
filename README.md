# AX - AI Command Agent

一个用 Rust 编写的 CLI 智能命令行代理。用自然语言与系统交互，AX 会理解你的意图并自动执行对应的系统命令。

```bash
ax 列出当前目录内容
ax 查看系统内存使用情况
ax 找到所有大于100MB的文件
ax 创建一个 Nginx 配置文件
```

## 特性

- **自然语言交互** — 用中文或英文描述你想做什么，AX 自动翻译为系统命令
- **多协议支持** — 兼容 OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 三种 API 协议
- **流式输出** — SSE 流式传输，实时显示思考过程和回复内容（打字机效果）
- **安全确认** — 修改类命令（删除、安装、权限变更等）执行前会请求用户确认
- **多轮工具调用** — 支持最多 20 轮连续工具调用，处理复杂任务
- **上下文记忆** — 自动记录历史对话，支持跨会话的上下文延续
- **智能截断** — 长输出自动截断，节省 token 开销
- **跨平台** — 支持 macOS、Linux、Windows

## 安装

### 从源码编译

```bash
git clone https://github.com/your-username/AX_Shell.git
cd AX_Shell
cargo build --release
```

编译产物在 `target/release/ax`，可以复制到 PATH 目录：

```bash
cp target/release/ax /usr/local/bin/
```

## 配置

首次使用需要配置 API 密钥和接口地址：

```bash
# 设置 API 密钥
ax -s API_KEY=sk-xxxxxxxxxxxxxxxx

# 设置 API 地址（默认为 OpenAI 官方地址）
ax -s BASE_URL=https://api.openai.com/v1

# 设置协议（默认 open_chat）
ax -s PROTOCOL=open_chat

# 设置模型（默认 gpt-4）
ax -s MODEL=gpt-4
```

可以一次性设置多个参数：

```bash
ax -s API_KEY=sk-xxx -s BASE_URL=https://api.deepseek.com/v1 -s MODEL=deepseek-chat -s PROTOCOL=open_chat
```

### 支持的协议

| 协议 | 说明 | 默认模型 |
|------|------|----------|
| `open_chat` | OpenAI Chat Completions API | gpt-4 |
| `openai_response` | OpenAI Responses API | gpt-4 |
| `anthropic_message` | Anthropic Messages API | claude-sonnet-4-20250514 |

### 配置示例

**DeepSeek:**
```bash
ax -s API_KEY=sk-xxx -s BASE_URL=https://api.deepseek.com/v1 -s MODEL=deepseek-chat
```

**MiMo (小米):**
```bash
ax -s API_KEY=sk-xxx -s BASE_URL=https://api.mimo.xiaomi.com/v1 -s MODEL=mimo-v2.5-pro
```

**Anthropic Claude:**
```bash
ax -s API_KEY=sk-ant-xxx -s PROTOCOL=anthropic_message -s MODEL=claude-sonnet-4-20250514
```

**本地模型 (Ollama 等):**
```bash
ax -s API_KEY=ollama -s BASE_URL=http://localhost:11434/v1 -s MODEL=qwen2.5
```

## 使用

### 基本用法

```bash
ax <自然语言命令>
```

### 示例

```bash
# 查询类（自动执行）
ax 列出当前目录内容
ax 查看系统内存使用情况
ax 查看当前进程
ax 查看磁盘使用率

# 修改类（需要确认）
ax 创建一个 hello.txt 文件，内容为 Hello World
ax 删除所有 .tmp 文件
ax 安装 nginx

# 复杂任务（多步执行）
ax 查看 80 端口是否被占用，如果被占用就杀掉那个进程
ax 找到所有大于 100MB 的日志文件并列出它们
ax 检查防火墙状态并开放 8080 端口
```

### 管道输入

```bash
echo "分析这个文件的内容" | ax
cat error.log | ax 帮我分析这个错误日志
```

### 命令行选项

```bash
ax <command>              执行自然语言命令
ax -s KEY=VALUE           配置设置
ax -v                     查看版本
ax -h                     查看帮助
```

## 工具说明

AX 向 LLM 注册了三个工具：

| 工具 | 说明 | 类型 |
|------|------|------|
| `execute_command` | 执行 shell 命令 | query / modify |
| `execute_command_with_timeout` | 带超时的命令执行 | query / modify |
| `write_file` | 写入文件 | modify（始终需要确认） |

- **query** — 只读命令（ls、cat、grep、ps 等），直接执行
- **modify** — 修改系统状态的命令（rm、mv、chmod、install 等），需要用户确认

## 工作流程

```
用户输入 → 构建上下文（系统提示 + 历史 + 动态信息 + 用户消息）
         → 调用 LLM API（流式）
         → 显示思考过程（Thinking）
         → 显示回复内容（Content）
         → 如果有工具调用：
            ├── query 类型 → 直接执行
            └── modify 类型 → 请求用户确认
         → 执行命令并显示结果
         → 将结果回传 LLM，继续下一轮
         → 无工具调用时结束，保存历史
```

## 上下文策略

AX 采用缓存友好的上下文构建策略：

1. **系统提示词**（静态）— 规则、系统信息、主机名，永远不变
2. **历史消息**（增长）— 最近 10 轮对话，逐步追加
3. **动态上下文**（变化）— 当前时间、工作目录、执行说明
4. **当前用户消息**（变化）— 本次请求
5. **工具调用消息**（追加）— Agent 循环中产生

## 配置文件

配置存储在 `~/.ax/settings.json`：

```json
{
  "api_key": "sk-xxx",
  "base_url": "https://api.openai.com/v1",
  "protocol": "open_chat",
  "model": "gpt-4"
}
```

历史记录存储在 `~/.ax/history.json`，保留最近 100 条。

## 构建

```bash
# 开发构建
cargo build

# 发布构建（优化）
cargo build --release

# 运行
cargo run -- 列出当前目录内容
```

## 依赖

- [reqwest](https://github.com/seanmonstar/reqwest) — HTTP 客户端
- [tokio](https://github.com/tokio-rs/tokio) — 异步运行时
- [serde](https://github.com/serde-rs/serde) / [serde_json](https://github.com/serde-rs/json) — JSON 序列化
- [chrono](https://github.com/chronotope/chrono) — 时间处理
- [ansi_term](https://github.com/ogham/rust-ansi-term) — 终端颜色
- [terminal_size](https://github.com/eminence/terminal-size) — 终端尺寸检测

## License

MIT
