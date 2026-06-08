# lastmem MCP Server

Exposes two MCP tools — `store_messages` and `retrieve_messages` — backed by S3 Vectors memory. Built with [FastMCP](https://github.com/jlowin/fastmcp).

## Prerequisites

Same as the main project: Ollama running locally, AWS credentials, and a `.env` file. See [README.md](README.md).

## Install dependencies

```powershell
uv sync
```

## Start the Server

```powershell
uv run fastmcp run server.py --transport streamable-http --port 8000
```

The MCP endpoint will be at `http://localhost:8000/mcp`.

For SSE transport (older clients):

```powershell
uv run fastmcp run server.py --transport sse --port 8000
```

## Connecting an MCP Client

### opencode / Kiro

```json
{
  "mcp": {
    "lastmem": {
      "type": "remote",
      "url": "http://localhost:8000/mcp",
      "enabled": true
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lastmem": {
      "type": "remote",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Claude Desktop config location:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

---

## Available Tools

### `store_messages`

Stores a conversation with metadata into vector memory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | array | yes | List of `{role, content}` objects |
| `metadata.user_id` | string | yes | Identifies the memory owner |
| `metadata.category` | string | no | Optional category label |
| `metadata.tags` | string[] | no | Optional list of tags |
| `metadata.source` | string | no | Optional source identifier |
| `metadata.*` | any | no | Extra keys stored as-is |

### `retrieve_messages`

Searches stored memories by semantic similarity.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural language search query |
| `user_id` | string | yes | — | Filter results to this user |
| `top_k` | integer | no | 10 | Max results to return |

---

## Troubleshooting

- **404 on /mcp** — make sure you started with `--transport streamable-http`, not the default stdio
- **Server fails to start** — check Ollama is running and `.env` has valid AWS credentials
- **Empty results** — `user_id` must match exactly what was used when storing
