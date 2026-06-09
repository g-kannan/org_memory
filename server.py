from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from mem0_vectors import Memory, build_memory_config, langfuse_observation, flush_langfuse

mcp = FastMCP("lastmem MCP Message Store")

# Initialise memory backend once at module load (shared across all tool calls)
_memory: Memory = Memory.from_config(build_memory_config())


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str = Field(..., description="Required. Identifies the memory owner")
    category: str | None = Field(None, description="Optional category label")
    tags: list[str] | None = Field(None, description="Optional list of tags")
    source: str | None = Field(None, description="Optional source identifier")


class MemoryResult(BaseModel):
    id: str
    score: float | None = None
    memory: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def store_messages(messages: list[Message], metadata: Metadata) -> Any:
    """Store a list of messages with metadata into the memory backend.

    Args:
        messages: One or more messages to store (each with role and content).
        metadata: Metadata to attach; user_id is required.
    """
    if not messages:
        raise ValueError("messages must contain at least one item")
    with langfuse_observation(
        name="mcp-store_messages",
        input={"messages": [m.model_dump() for m in messages], "metadata": metadata.model_dump()},
        metadata={"user_id": metadata.user_id},
    ) as span:
        result = _memory.add(
            [m.model_dump() for m in messages],
            user_id=metadata.user_id,
            metadata=metadata.model_dump(),
        )
        if span:
            span.update(output=result)
        flush_langfuse()
        return result


@mcp.tool()
def retrieve_messages(query: str, user_id: str, top_k: int = 10) -> list[MemoryResult]:
    """Search stored memories by semantic similarity.

    Args:
        query: Natural language search query.
        user_id: Filter results to this user.
        top_k: Maximum number of results to return (default 10).
    """
    with langfuse_observation(
        name="mcp-retrieve_messages",
        input={"query": query, "user_id": user_id, "top_k": top_k},
    ) as span:
        response = _memory.search(query, filters={"user_id": user_id}, limit=top_k)
        items = response.get("results", response) if isinstance(response, dict) else response
        results = [
            MemoryResult(
                id=r.get("id", ""),
                score=r.get("score"),
                memory=r.get("memory", ""),
                metadata=r.get("metadata") or {},
            )
            for r in items
            if isinstance(r, dict)
        ]
        if span:
            span.update(output=[r.model_dump() for r in results])
        flush_langfuse()
        return results


@mcp.tool()
def update_message(memory_id: str, data: str, metadata: dict[str, Any] | None = None) -> Any:
    """Update a stored memory by ID.

    Args:
        memory_id: ID of the memory to update.
        data: New memory text.
        metadata: Optional metadata to attach to the updated memory.
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id must be a non-empty string")
    if not data or not data.strip():
        raise ValueError("data must be a non-empty string")
    with langfuse_observation(
        name="mcp-update_message",
        input={"memory_id": memory_id, "data": data, "metadata": metadata or {}},
    ) as span:
        result = _memory.update(memory_id=memory_id, data=data, metadata=metadata)
        if span:
            span.update(output=result)
        flush_langfuse()
        return result


@mcp.tool()
def delete_message(memory_id: str) -> Any:
    """Delete a stored memory by ID.

    Args:
        memory_id: ID of the memory to delete.
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id must be a non-empty string")
    with langfuse_observation(
        name="mcp-delete_message",
        input={"memory_id": memory_id},
    ) as span:
        result = _memory.delete(memory_id=memory_id)
        if span:
            span.update(output=result)
        flush_langfuse()
        return result


@mcp.tool()
def delete_all_messages(user_id: str) -> Any:
    """Delete all memories for a user.

    Args:
        user_id: User ID whose memories should be deleted.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    with langfuse_observation(
        name="mcp-delete_all_messages",
        input={"user_id": user_id},
    ) as span:
        result = _memory.delete_all(user_id=user_id)
        if span:
            span.update(output=result)
        flush_langfuse()
        return result
