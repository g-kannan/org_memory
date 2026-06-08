# lastmem MCP Server

Org-scoped memory server built with [FastMCP](https://github.com/jlowin/fastmcp) and deployed on [Modal](https://modal.com). Memories are stored in AWS S3 Vectors and retrieved by semantic similarity. Identity context is carried in a single HTTP header — no changes to tool-call signatures per request.

## Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store a memory string at org / team / project / user scope |
| `retrieve_memory` | Search memories by semantic similarity, with optional scope filter |

---

## Deployment (Modal)

### One-time setup

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Create the secret with all required env vars
modal secret create org-memory-secrets \
    AWS_ACCESS_KEY_ID=<your-key> \
    AWS_SECRET_ACCESS_KEY=<your-secret> \
    AWS_DEFAULT_REGION=ap-south-1 \
    VECTOR_BUCKET=orgmem-vector \
    COLLECTION=orgmem-vector-ix \
    LANGFUSE_SECRET_KEY=<optional> \
    LANGFUSE_PUBLIC_KEY=<optional> \
    LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com \
    LLM_PROVIDER=aws_bedrock \
    BEDROCK_MODEL=openai.gpt-oss-20b-1:0
```

### Deploy

```bash
modal deploy modal_server.py
```

### Serve locally (hot-reload)

```bash
modal serve modal_server.py
```

The MCP endpoint will be at:
```
https://<your-workspace>--org-memory-mcp-web.modal.run/mcp/
```

---

## Connecting an MCP Client

### Kiro / opencode

Add to your MCP config (replace the URL with your Modal endpoint):

```json
{
  "mcpServers": {
    "lastmem": {
      "type": "streamable-http",
      "url": "https://<your-workspace>--org-memory-mcp-web.modal.run/mcp/",
      "headers": {
        "X-Memory-Context": "org:<org_id>"
      }
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
      "type": "streamable-http",
      "url": "https://<your-workspace>--org-memory-mcp-web.modal.run/mcp/",
      "headers": {
        "X-Memory-Context": "org:<org_id>:team:<team_id>"
      }
    }
  }
}
```

Config file location:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Sample configs by scope level

**Org scope** — all memories shared across the organisation:
```json
{
  "mcpServers": {
    "lastmem": {
      "type": "streamable-http",
      "url": "https://<workspace>--org-memory-mcp-web.modal.run/mcp/",
      "headers": { "X-Memory-Context": "org:acme" }
    }
  }
}
```

**Team scope** — memories visible to a specific team:
```json
{
  "mcpServers": {
    "lastmem": {
      "type": "streamable-http",
      "url": "https://<workspace>--org-memory-mcp-web.modal.run/mcp/",
      "headers": { "X-Memory-Context": "org:acme:team:engineering" }
    }
  }
}
```

**Project scope** — memories scoped to one project:
```json
{
  "mcpServers": {
    "lastmem": {
      "type": "streamable-http",
      "url": "https://<workspace>--org-memory-mcp-web.modal.run/mcp/",
      "headers": { "X-Memory-Context": "org:acme:team:engineering:project:api-v2" }
    }
  }
}
```

**User scope** — no header needed; pass `user_id` in the tool call directly.

---

## X-Memory-Context Header

Every request targeting org-, team-, or project-scoped memories **must** include this header. Valid formats:

```
org:<org_id>
org:<org_id>:team:<team_id>
org:<org_id>:team:<team_id>:project:<project_id>
```

Each component (`org_id`, `team_id`, `project_id`) must match `[A-Za-z0-9_-]`.

User-scoped operations pass `user_id` in the tool body instead and do not require this header.

---

## Tool Reference

### `store_memory`

Store a single memory string under the resolved scope.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory` | string | yes | Text to store. Must be non-empty. |
| `scope` | string | yes | One of `org`, `team`, `project`, `user` |
| `team_id` | string | no | Overrides `team_id` from header (team/project scope) |
| `project_id` | string | no | Overrides `project_id` from header (project scope) |
| `user_id` | string | no* | Required only when `scope="user"` |
| `metadata` | object | no | Extra key-value pairs attached to the memory entry |

**Returns**: `{ "memory_id": "<uuid>", "scope_key": "<resolved-scope-key>" }`

**Example tool call**:
```json
{
  "tool": "store_memory",
  "arguments": {
    "memory": "Our API rate limit is 1000 req/min per team.",
    "scope": "team",
    "metadata": { "source": "runbook", "tags": ["api", "limits"] }
  }
}
```

---

### `retrieve_memory`

Search memories by semantic similarity with optional scope filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural language search query |
| `scope` | string | no | — | Filter to a specific scope level |
| `team_id` | string | no | — | Override `team_id` from header |
| `project_id` | string | no | — | Override `project_id` from header |
| `user_id` | string | no | — | Required when `scope="user"` |
| `top_k` | integer | no | `10` | Max results to return (must be ≥ 1) |

**Returns**: Array of `{ "id", "memory", "score", "metadata" }` objects. Empty array when no results.

When `scope` is omitted the search spans all entries accessible from the org in the header.

**Example tool call** (scoped to team):
```json
{
  "tool": "retrieve_memory",
  "arguments": {
    "query": "What is our API rate limit?",
    "scope": "team",
    "top_k": 5
  }
}
```

**Example tool call** (org-wide, no scope filter):
```json
{
  "tool": "retrieve_memory",
  "arguments": {
    "query": "deployment process",
    "top_k": 10
  }
}
```

---

## Scope Key Reference

| Scope | Header required | Resolved scope_key |
|-------|-----------------|--------------------|
| `org` | `org:<org_id>` | `org:<org_id>` |
| `team` | `org:<org_id>:team:<team_id>` | `team:<org_id>:<team_id>` |
| `project` | `org:<org_id>:team:<team_id>:project:<project_id>` | `project:<org_id>:<team_id>:<project_id>` |
| `user` | not required | `user:<user_id>` |

---

## Local Development

For local testing, use `server.py` with Ollama instead of Bedrock:

```bash
uv sync
uv run fastmcp run server.py --transport streamable-http --port 8000
```

Point your MCP client at `http://localhost:8000/mcp/`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `X-Memory-Context header is required` | Add the header to your MCP client config |
| `team_id is required for team scope` | Use `org:<org_id>:team:<team_id>` header format |
| Empty results | Scope key must match exactly what was used when storing |
| 404 on `/mcp` | Confirm Modal deployment completed and URL is correct |
