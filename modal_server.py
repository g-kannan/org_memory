"""
Modal deployment for the org_memory MCP server.
Self-contained — no local module dependencies.

Deploy:    modal deploy modal_server.py
Serve:     modal serve modal_server.py

MCP endpoint: https://<workspace>--org-memory-mcp-web.modal.run/mcp/
Transport: Streamable HTTP

One-time secret setup:
    modal secret create org-memory-secrets \
        AWS_ACCESS_KEY_ID=... \
        AWS_SECRET_ACCESS_KEY=... \
        AWS_DEFAULT_REGION=ap-south-1 \
        VECTOR_BUCKET=orgmem-vector \
        COLLECTION=orgmem-vector-ix \
        LANGFUSE_SECRET_KEY=... \
        LANGFUSE_PUBLIC_KEY=... \
        LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com \
        LLM_PROVIDER=aws_bedrock \
        BEDROCK_MODEL=openai.gpt-oss-20b-1:0
"""

import json
import re
import logging

import modal

secrets = [modal.Secret.from_name("org-memory-secrets")]

TITAN_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
TITAN_EMBED_DIMS = 1024

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "mem0ai[nlp]",
        "boto3",
        "langfuse>=4.7.1",
        "fastmcp>=2.0.0",
        "fastapi==0.115.14",
        "uvicorn[standard]",
    )
)

app = modal.App("org-memory-mcp", image=image, secrets=secrets)


# ---------------------------------------------------------------------------
# Custom classes defined at module level so VectorStoreFactory / LlmFactory
# can locate them via their dotted module path (they call cls.rsplit(".", 1)).
# These MUST be at module level — defining them inside a function means the
# factory cannot import them by their dotted path at instantiation time.
# ---------------------------------------------------------------------------

from mem0.llms.aws_bedrock import AWSBedrockLLM
from mem0.configs.llms.aws_bedrock import AWSBedrockConfig
from mem0.utils.factory import LlmFactory, VectorStoreFactory
from mem0.vector_stores.s3_vectors import S3Vectors, OutputData

logger = logging.getLogger(__name__)


class S3VectorsSimilarity(S3Vectors):
    """S3 Vectors store that converts distance to a similarity score."""

    def _distance_to_similarity(self, distance):
        if distance is None:
            return None
        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        if self.distance_metric == "euclidean":
            return 1.0 / (1.0 + distance)
        return distance

    def _parse_output(self, vectors):
        results = []
        for v in vectors:
            payload = v.get("metadata", {})
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            results.append(
                OutputData(
                    id=v.get("key"),
                    score=self._distance_to_similarity(v.get("distance")),
                    payload=payload,
                )
            )
        return results


class BedrockOpenAILLM(AWSBedrockLLM):
    """Bedrock LLM adapter for openai.* cross-region inference model IDs."""

    def _build_openai_messages(self, messages):
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "developer":
                role = "system"
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            result.append({"role": role, "content": content})
        return result

    def _parse_openai_response(self, response):
        body = response.get("body").read().decode("utf-8")
        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            return str(data)
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return content

    def _extract_json_response(self, text: str) -> str:
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
        decoder = json.JSONDecoder()
        decoded = []
        for i, ch in enumerate(text):
            if ch not in "[{":
                continue
            try:
                val, _ = decoder.raw_decode(text[i:])
                decoded.append(val)
            except json.JSONDecodeError:
                continue
        if not decoded:
            return text
        for val in reversed(decoded):
            if isinstance(val, dict) and isinstance(val.get("memory"), list):
                return json.dumps(val)
        val = decoded[-1]
        if isinstance(val, list):
            return json.dumps({"memory": val})
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, list):
                    return json.dumps({"memory": v})
        return json.dumps(val)

    def generate_response(self, messages, response_format=None, tools=None,
                          tool_choice="auto", stream=False, **kwargs):
        request_body = {
            "model": self.config.model,
            "messages": self._build_openai_messages(messages),
            "max_completion_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if self.config.top_p is not None:
            request_body["top_p"] = self.config.top_p
        if response_format:
            request_body["response_format"] = response_format
        if getattr(self.config, "model_kwargs", None):
            request_body.update(self.config.model_kwargs)
        response = self.client.invoke_model(
            body=json.dumps(request_body),
            modelId=self.config.model,
            accept="application/json",
            contentType="application/json",
        )
        parsed = self._parse_openai_response(response)
        if response_format:
            return self._extract_json_response(parsed)
        return parsed


# Register providers once at import time using the module's actual dotted path.
# The factory resolves "modal_server.S3VectorsSimilarity" via importlib, so the
# classes must exist at module scope — not inside a nested function.
# Always use the literal module filename — __name__ becomes "__main__" when Modal
# executes the file directly, which breaks importlib resolution inside the container.
_module_name = "modal_server"
VectorStoreFactory.provider_to_class["s3_vectors"] = f"{_module_name}.S3VectorsSimilarity"
LlmFactory.provider_to_class["aws_bedrock"] = (f"{_module_name}.BedrockOpenAILLM", AWSBedrockConfig)


def _make_app():
    """Build and return the FastAPI/FastMCP ASGI app. Called inside the container."""
    import os
    from contextlib import contextmanager
    from typing import Any

    from fastapi import FastAPI
    from fastmcp import FastMCP
    from pydantic import BaseModel, ConfigDict, Field

    from mem0 import Memory as Mem0Memory
    from mem0.memory.main import _safe_deepcopy_config
    from mem0.utils.scoring import ENTITY_BOOST_WEIGHT
    # S3VectorsSimilarity, BedrockOpenAILLM, VectorStoreFactory already available at module scope

    # ------------------------------------------------------------------
    # Langfuse helpers
    # ------------------------------------------------------------------
    try:
        from langfuse import get_client as get_langfuse_client
    except ImportError:
        get_langfuse_client = None

    _langfuse_auth_checked = False
    _langfuse_auth_ok = False

    def langfuse_is_configured() -> bool:
        nonlocal _langfuse_auth_checked, _langfuse_auth_ok
        if not (get_langfuse_client
                and os.getenv("LANGFUSE_PUBLIC_KEY")
                and os.getenv("LANGFUSE_SECRET_KEY")):
            return False
        if _langfuse_auth_checked:
            return _langfuse_auth_ok
        _langfuse_auth_checked = True
        try:
            _langfuse_auth_ok = bool(get_langfuse_client().auth_check())
        except Exception as exc:
            logger.warning("Langfuse auth check failed: %s", exc)
            _langfuse_auth_ok = False
        return _langfuse_auth_ok

    @contextmanager
    def langfuse_observation(name: str, as_type: str = "span", **kwargs):
        if not langfuse_is_configured():
            yield None
            return
        try:
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
                yield obs
        except Exception as exc:
            logger.warning("Langfuse observation '%s' failed: %s", name, exc)
            yield None

    def flush_langfuse() -> None:
        if not langfuse_is_configured():
            return
        try:
            get_langfuse_client().flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)

    # ------------------------------------------------------------------
    # S3-safe Memory subclass (separate entity store collection)
    # ------------------------------------------------------------------
    raw_collection = os.getenv("COLLECTION", "orgmem-vector-ix")
    collection_name = re.sub(r"[^A-Za-z0-9-]", "-", raw_collection.strip()).strip("-")
    entity_collection_name = (collection_name + "-entities")[:63].rstrip("-")

    class S3SafeMemory(Mem0Memory):
        @property
        def entity_store(self):
            if self._entity_store is None:
                entity_config = _safe_deepcopy_config(self.config.vector_store.config)
                if hasattr(entity_config, "collection_name"):
                    entity_config.collection_name = entity_collection_name
                elif isinstance(entity_config, dict):
                    entity_config["collection_name"] = entity_collection_name
                self._entity_store = VectorStoreFactory.create(
                    self.config.vector_store.provider, entity_config
                )
            return self._entity_store

        def _compute_entity_boosts(self, query_entities, filters):
            seen, deduped = set(), []
            for entity_type, entity_text in query_entities[:8]:
                key = entity_text.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append((entity_type, entity_text))
            if not deduped:
                return {}
            search_filters = {
                k: v for k, v in filters.items()
                if k in ("user_id", "agent_id", "run_id") and v
            }
            memory_boosts = {}
            try:
                for _, entity_text in deduped:
                    entity_embedding = self.embedding_model.embed(entity_text, "search")
                    matches = self.entity_store.search(
                        query=entity_text, vectors=entity_embedding,
                        top_k=100, filters=search_filters,
                    )
                    for match in matches:
                        similarity = match.score if hasattr(match, "score") else 0.0
                        if similarity < 0.5:
                            continue
                        payload = match.payload if hasattr(match, "payload") else {}
                        linked = payload.get("linked_memory_ids", [])
                        if not isinstance(linked, list):
                            continue
                        weight = 1.0 / (1.0 + 0.001 * ((max(len(linked), 1) - 1) ** 2))
                        boost = similarity * ENTITY_BOOST_WEIGHT * weight
                        for mid in linked:
                            if mid:
                                memory_boosts[str(mid)] = max(memory_boosts.get(str(mid), 0.0), boost)
            except Exception as e:
                logger.warning("Entity boost computation failed: %s", e)
            return memory_boosts

    # ------------------------------------------------------------------
    # Build mem0 config
    # ------------------------------------------------------------------
    bedrock_region = (
        os.getenv("BEDROCK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )
    bedrock_model = os.getenv("BEDROCK_MODEL", "openai.gpt-oss-20b-1:0")

    mem0_config = {
        "vector_store": {
            "provider": "s3_vectors",
            "config": {
                "vector_bucket_name": os.getenv("VECTOR_BUCKET"),
                "collection_name": collection_name,
                "embedding_model_dims": TITAN_EMBED_DIMS,
                "distance_metric": "cosine",
                "region_name": os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
            },
        },
        "llm": {
            "provider": "aws_bedrock",
            "config": {
                "model": bedrock_model,
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
                "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
                "aws_region": bedrock_region,
            },
        },
        "embedder": {
            "provider": "aws_bedrock",
            "config": {
                "model": TITAN_EMBED_MODEL,
                "embedding_dims": TITAN_EMBED_DIMS,
                "aws_region": bedrock_region,
            },
        },
    }

    _memory = S3SafeMemory.from_config(mem0_config)

    # ------------------------------------------------------------------
    # MCP tool models
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # MCP server
    # ------------------------------------------------------------------
    mcp = FastMCP("lastmem MCP Message Store")

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

    # ------------------------------------------------------------------
    # FastAPI + streamable-HTTP MCP transport (stateless)
    # ------------------------------------------------------------------
    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)
    fastapi_app = FastAPI(lifespan=mcp_app.router.lifespan_context)
    fastapi_app.mount("/", mcp_app, "mcp")
    return fastapi_app


# ---------------------------------------------------------------------------
# Modal web endpoint
# ---------------------------------------------------------------------------
@app.function(
    cpu=1,
    memory=512,
    timeout=300,
)
@modal.asgi_app()
def web():
    return _make_app()
