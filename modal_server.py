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

-------------------------------------------------------------------------------
Scoped Memory — Identity Header & Scope Levels
-------------------------------------------------------------------------------

X-Memory-Context header
    Every MCP request that targets org-, team-, or project-scoped memories MUST
    include this HTTP header.  The header carries the calling identity in one of
    three formats:

        org:<org_id>
        org:<org_id>:team:<team_id>
        org:<org_id>:team:<team_id>:project:<project_id>

    Each component (org_id, team_id, project_id) must contain only characters
    from [A-Za-z0-9_-].  User-scoped operations supply a user_id directly in
    the tool-call body instead and do not require this header.

Scope levels and their Scope_Key formats
    The scope parameter on store_memory / retrieve_memory selects one of four
    hierarchical levels.  The resolved Scope_Key is stored as metadata on every
    Memory_Entry and used as the partition filter in Memory_Backend queries.

    Scope       Requires in header          Scope_Key format
    -------     --------------------------  ------------------------------------------
    org         org_id                      org:<org_id>
    team        org_id, team_id             team:<org_id>:<team_id>
    project     org_id, team_id, project_id project:<org_id>:<team_id>:<project_id>
    user        (none — pass user_id in     user:<user_id>
                 the tool call instead)

MCP tools
    store_memory(memory, scope, team_id=None, project_id=None,
                 user_id=None, metadata=None) -> {"memory_id": ..., "scope_key": ...}
        Store a single memory string under the resolved scope.  Returns the
        memory_id assigned by the backend and the Scope_Key used.

    update_memory(memory_id, data, metadata=None) -> {"memory_id": ..., "message": ...}
        Update an existing memory by ID while preserving its stored scope metadata.

    delete_memory(memory_id) -> {"memory_id": ..., "message": ...}
        Delete one memory by ID.

    delete_all_memories(scope, team_id=None, project_id=None, user_id=None)
        Delete all memories under the resolved scope.

    retrieve_memory(query, scope=None, team_id=None, project_id=None,
                    user_id=None, top_k=10) -> list[MemoryResult]
        Search memories by semantic similarity.  When scope is given the search
        is filtered to that Scope_Key; when omitted all entries whose scope_key
        is accessible from the calling org are searched.

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
    from datetime import datetime, timezone
    from typing import Any

    from fastapi import FastAPI
    from fastmcp import FastMCP
    from fastmcp.server.dependencies import get_http_request
    from pydantic import BaseModel, Field

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
    # Context_Parser — parse and validate X-Memory-Context header
    # ------------------------------------------------------------------
    _COMPONENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")

    def parse_memory_context(header_value: str | None) -> dict | None:
        """Parse and validate the ``X-Memory-Context`` HTTP header.

        Args:
            header_value: Raw value of the ``X-Memory-Context`` header, or
                ``None`` when the header is absent.

        Returns:
            ``None`` when the header is absent (callers decide how to handle
            that case).  Otherwise a dict with the following keys:

            * ``org_id`` (str) — always present.
            * ``team_id`` (str) — present only when the header includes a team
              component.
            * ``project_id`` (str) — present only when the header includes a
              project component.

        Raises:
            ValueError: If the header value does not match one of the three
                recognised formats, or if any identity component contains
                characters outside ``[A-Za-z0-9_-]``.

        Valid formats::

            org:<org_id>
            org:<org_id>:team:<team_id>
            org:<org_id>:team:<team_id>:project:<project_id>
        """
        if header_value is None:
            return None

        parts = header_value.split(":")

        # --- structural validation ---
        valid = False
        if len(parts) == 2 and parts[0] == "org":
            valid = True
        elif len(parts) == 4 and parts[0] == "org" and parts[2] == "team":
            valid = True
        elif len(parts) == 6 and parts[0] == "org" and parts[2] == "team" and parts[4] == "project":
            valid = True

        if not valid:
            raise ValueError(
                "Invalid X-Memory-Context header. Expected one of:\n"
                "  org:<org_id>\n"
                "  org:<org_id>:team:<team_id>\n"
                "  org:<org_id>:team:<team_id>:project:<project_id>"
            )

        # --- character validation on each identity component ---
        # component positions: org_id=1, team_id=3 (if present), project_id=5 (if present)
        component_positions = {1: "org_id"}
        if len(parts) >= 4:
            component_positions[3] = "team_id"
        if len(parts) == 6:
            component_positions[5] = "project_id"

        for idx, name in component_positions.items():
            value = parts[idx]
            if not value:
                raise ValueError(
                    f"Invalid X-Memory-Context header: '{name}' component is empty."
                )
            if not _COMPONENT_RE.match(value):
                raise ValueError(
                    f"Invalid X-Memory-Context header: '{name}' component '{value}' "
                    f"contains characters outside [A-Za-z0-9_-]."
                )

        # --- build result dict ---
        result: dict = {"org_id": parts[1]}
        if len(parts) >= 4:
            result["team_id"] = parts[3]
        if len(parts) == 6:
            result["project_id"] = parts[5]
        return result

    # ------------------------------------------------------------------
    # Scope_Resolver — map scope enum + parsed context to Scope_Key
    # ------------------------------------------------------------------

    def resolve_scope_key(scope: str, ctx: dict | None, user_id: str | None = None) -> str:
        """Resolve a scope enum value and parsed context to a concrete Scope_Key.

        Args:
            scope: One of ``"org"``, ``"team"``, ``"project"``, or ``"user"``.
            ctx: Parsed context dict returned by ``parse_memory_context()`` — contains
                ``org_id`` (always present), ``team_id`` (optional), and ``project_id``
                (optional).  May be ``None`` only when ``scope="user"`` where no
                Identity_Header is required.
            user_id: The user identifier; required when ``scope="user"``.

        Returns:
            A Scope_Key string in one of the following formats:

            * ``org:<org_id>``
            * ``team:<org_id>:<team_id>``
            * ``project:<org_id>:<team_id>:<project_id>``
            * ``user:<user_id>``

        Raises:
            ValueError: When required identifiers are missing for the requested scope,
                or when ``scope`` is not one of the four recognised values.
        """
        if scope == "org":
            if ctx is None or not ctx.get("org_id"):
                raise ValueError(
                    "scope 'org' requires 'org_id' from the X-Memory-Context header."
                )
            return f"org:{ctx['org_id']}"

        elif scope == "team":
            if ctx is None or not ctx.get("org_id"):
                raise ValueError(
                    "scope 'team' requires 'org_id' from the X-Memory-Context header."
                )
            if not ctx.get("team_id"):
                raise ValueError(
                    "scope 'team' requires 'team_id' in the X-Memory-Context header "
                    "(expected format: org:<org_id>:team:<team_id>)."
                )
            return f"team:{ctx['org_id']}:{ctx['team_id']}"

        elif scope == "project":
            if ctx is None or not ctx.get("org_id"):
                raise ValueError(
                    "scope 'project' requires 'org_id' from the X-Memory-Context header."
                )
            if not ctx.get("team_id"):
                raise ValueError(
                    "scope 'project' requires 'team_id' in the X-Memory-Context header "
                    "(expected format: org:<org_id>:team:<team_id>:project:<project_id>)."
                )
            if not ctx.get("project_id"):
                raise ValueError(
                    "scope 'project' requires 'project_id' in the X-Memory-Context header "
                    "(expected format: org:<org_id>:team:<team_id>:project:<project_id>)."
                )
            return f"project:{ctx['org_id']}:{ctx['team_id']}:{ctx['project_id']}"

        elif scope == "user":
            if not user_id:
                raise ValueError(
                    "scope 'user' requires a 'user_id' to be supplied in the tool call."
                )
            return f"user:{user_id}"

        else:
            raise ValueError(
                f"Unrecognised scope '{scope}'. "
                "Valid scopes are: 'org', 'team', 'project', 'user'."
            )

    def normalize_metadata(metadata: dict[str, Any] | str | None) -> dict[str, Any]:
        """Accept object metadata or JSON-encoded object metadata from MCP clients."""
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise ValueError("metadata must be a JSON object or JSON-encoded object string") from exc
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("metadata must be an object")

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
    class MemoryResult(BaseModel):
        id: str
        score: float | None = None
        memory: str
        created_at: str | None = None
        updated_at: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # MCP server
    # ------------------------------------------------------------------
    mcp = FastMCP(
        "StrategyShifu",
        instructions=(
            "Use this server to store and retrieve scoped organizational memory, "
            "including project decisions, architecture choices, events, team facts, "
            "and org knowledge. Prefer retrieve_memory for questions asking what the "
            "organization, team, or project remembers."
        ),
    )

    @mcp.tool()
    async def store_memory(
        memory: str,
        scope: str,
        team_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | str | None = None,
    ) -> dict:
        """Store a scoped organization memory fact.

        Use this when the user asks to remember, save, record, or persist
        organizational knowledge such as project decisions, architecture choices,
        recurring events, team conventions, requirements, incidents, or user notes.
        Store project-specific decisions with scope='project', team-wide facts with
        scope='team', org-wide facts with scope='org', and personal notes with
        scope='user'. Add metadata such as {"type": "decision"} or {"type": "event"}
        when the memory category is clear.

        Args:
            memory: The memory text to store. Must be a non-empty string.
            scope: The scope level — one of 'org', 'team', 'project', or 'user'.
            team_id: Override the team_id from the Identity Header (only used when scope='team' or 'project').
            project_id: Override the project_id from the Identity Header (only used when scope='project').
            user_id: Required when scope='user'; ignored for all other scopes.
            metadata: Optional key-value map of extra metadata to attach to the memory.
        """
        # Reject empty memory before touching the backend
        if not memory or not memory.strip():
            return {"error": "memory must be a non-empty string"}

        # Read and parse the X-Memory-Context header
        try:
            request = get_http_request()
            header_value = request.headers.get("x-memory-context") or request.headers.get("X-Memory-Context")
            parsed_ctx = parse_memory_context(header_value)
        except ValueError as exc:
            return {"error": str(exc)}

        # Allow tool-call team_id / project_id to override what's in the header (req 3.4)
        if parsed_ctx is not None:
            if team_id is not None:
                parsed_ctx = {**parsed_ctx, "team_id": team_id}
            if project_id is not None:
                parsed_ctx = {**parsed_ctx, "project_id": project_id}

        # Resolve the scope key
        try:
            scope_key = resolve_scope_key(scope, parsed_ctx, user_id=user_id)
        except ValueError as exc:
            return {"error": str(exc)}

        try:
            memory_metadata = normalize_metadata(metadata)
        except ValueError as exc:
            return {"error": str(exc)}

        now_iso = datetime.now(timezone.utc).isoformat()
        memory_metadata.setdefault("created_at", now_iso)
        memory_metadata.setdefault("updated_at", now_iso)

        # Store to backend
        result = _memory.add(
            [{"role": "user", "content": memory}],
            user_id=scope_key,
            metadata={**memory_metadata, "scope": scope, "scope_key": scope_key},
        )

        # Extract memory_id from the add result
        memory_id = None
        events = []
        if isinstance(result, dict):
            # mem0 returns {"results": [{"id": ..., "event": "ADD", ...}, ...]}
            results_list = result.get("results", [])
            if results_list and isinstance(results_list, list):
                memory_id = results_list[0].get("id")
                events = [
                    {
                        "id": item.get("id"),
                        "event": item.get("event"),
                        "memory": item.get("memory"),
                    }
                    for item in results_list
                    if isinstance(item, dict)
                ]
            if memory_id is None:
                memory_id = result.get("id") or result.get("memory_id")
        elif isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                memory_id = first.get("id") or first.get("memory_id")
                events = [
                    {
                        "id": item.get("id"),
                        "event": item.get("event"),
                        "memory": item.get("memory"),
                    }
                    for item in result
                    if isinstance(item, dict)
                ]

        return {
            "memory_id": memory_id,
            "scope_key": scope_key,
            "created_at": memory_metadata.get("created_at"),
            "updated_at": memory_metadata.get("updated_at"),
            "events": events,
        }

    @mcp.tool()
    async def retrieve_memory(
        query: str,
        scope: str | None = None,
        team_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[MemoryResult] | dict:
        """Retrieve scoped organization memories by semantic search.

        Use this when the user asks what has been remembered, asks to find stored
        decisions, events, architecture choices, team facts, project context, or
        user notes. For prompts like "bring all decisions made in this project",
        call this tool with query terms such as "decisions architecture choices
        agreed approaches", scope='project', and a larger top_k. Include
        created_at and updated_at in the answer when comparing duplicates or
        conflicting memories.

        Args:
            query: Natural language search query. Must be a non-empty string.
            scope: Optional scope level — one of 'org', 'team', 'project', or 'user'.
                   When omitted, searches all entries accessible from the calling org.
            team_id: Override the team_id from the Identity Header (used when scope='team' or 'project').
            project_id: Override the project_id from the Identity Header (used when scope='project').
            user_id: Required when scope='user'; ignored for all other scopes.
            top_k: Maximum number of results to return (default 10). Must be >= 1.
        """
        # Reject empty query before touching the backend (req 4.1)
        if not query or not query.strip():
            return {"error": "query must be a non-empty string"}

        # Reject invalid top_k before touching the backend (req 4.6)
        if top_k < 1:
            return {"error": "top_k must be a positive integer"}

        # Read and parse the X-Memory-Context header
        try:
            request = get_http_request()
            header_value = request.headers.get("x-memory-context") or request.headers.get("X-Memory-Context")
            parsed_ctx = parse_memory_context(header_value)
        except ValueError as exc:
            return {"error": str(exc)}

        # Allow tool-call team_id / project_id to override what's in the header
        if parsed_ctx is not None:
            if team_id is not None:
                parsed_ctx = {**parsed_ctx, "team_id": team_id}
            if project_id is not None:
                parsed_ctx = {**parsed_ctx, "project_id": project_id}

        if scope is not None:
            # Scoped search: resolve the scope key and filter to exact match (req 4.2, 8.4)
            try:
                scope_key = resolve_scope_key(scope, parsed_ctx, user_id=user_id)
            except ValueError as exc:
                return {"error": str(exc)}

            # mem0 uses user_id as the partition filter; pass scope_key as user_id (req 4.7)
            response = _memory.search(query, filters={"user_id": scope_key}, top_k=top_k)
            responses = [response]
        else:
            # Unscoped search: search all entries for the org (req 4.3)
            # The header must be present so we can derive org_id for future prefix-based
            # filtering. mem0 requires an exact user_id/agent_id/run_id filter, so query
            # each concrete scope available from the header and merge the results.
            if parsed_ctx is None:
                return {"error": "X-Memory-Context header is required when scope is not specified"}
            scope_keys = [f"org:{parsed_ctx['org_id']}"]
            if parsed_ctx.get("team_id"):
                scope_keys.append(f"team:{parsed_ctx['org_id']}:{parsed_ctx['team_id']}")
            if parsed_ctx.get("team_id") and parsed_ctx.get("project_id"):
                scope_keys.append(
                    f"project:{parsed_ctx['org_id']}:{parsed_ctx['team_id']}:{parsed_ctx['project_id']}"
                )
            if user_id:
                scope_keys.append(f"user:{user_id}")

            responses = [
                _memory.search(query, filters={"user_id": scope_key}, top_k=top_k)
                for scope_key in scope_keys
            ]

        # Normalise response shape (mem0 may return a dict or a list)
        items = []
        for response in responses:
            response_items = response.get("results", response) if isinstance(response, dict) else response
            if isinstance(response_items, list):
                items.extend(response_items)

        deduped_items = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
            existing = deduped_items.get(item_id)
            if existing is None or (item.get("score") or 0) > (existing.get("score") or 0):
                deduped_items[item_id] = item
        items = sorted(
            deduped_items.values(),
            key=lambda item: item.get("score") or 0,
            reverse=True,
        )[:top_k]

        # Map to MemoryResult; pull scope_key from stored metadata (req 4.4)
        results = [
            MemoryResult(
                id=r.get("id", ""),
                score=r.get("score"),
                memory=r.get("memory", ""),
                created_at=r.get("created_at") or (r.get("metadata") or {}).get("created_at"),
                updated_at=r.get("updated_at") or (r.get("metadata") or {}).get("updated_at"),
                metadata={
                    **(r.get("metadata") or {}),
                    # surface scope_key at the top level of metadata for convenience
                    "scope_key": (r.get("metadata") or {}).get("scope_key", ""),
                },
            )
            for r in items
            if isinstance(r, dict)
        ]

        # Return empty list when no results without raising an error (req 4.5)
        return results

    @mcp.tool()
    async def update_memory(
        memory_id: str,
        data: str,
        metadata: dict[str, Any] | str | None = None,
    ) -> dict:
        """Update an existing memory by ID.

        Use this when the user asks to correct, revise, replace, or update an
        already stored memory and provides the memory ID. The memory text is
        replaced with data. Existing scope metadata is preserved so the memory
        stays in the same org/team/project/user partition.

        Args:
            memory_id: ID of the memory to update.
            data: New memory text. Must be a non-empty string.
            metadata: Optional custom metadata to merge into the memory.
        """
        if not memory_id or not memory_id.strip():
            return {"error": "memory_id must be a non-empty string"}
        if not data or not data.strip():
            return {"error": "data must be a non-empty string"}

        try:
            custom_metadata = normalize_metadata(metadata)
        except ValueError as exc:
            return {"error": str(exc)}

        try:
            existing_memory = _memory.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                return {"error": f"Memory with id {memory_id} not found"}

            existing_payload = existing_memory.payload or {}
            update_metadata = {**existing_payload, **custom_metadata}
            for protected_key in (
                "user_id",
                "agent_id",
                "run_id",
                "scope",
                "scope_key",
                "created_at",
            ):
                if protected_key in existing_payload:
                    update_metadata[protected_key] = existing_payload[protected_key]

            result = _memory.update(
                memory_id=memory_id,
                data=data,
                metadata=update_metadata,
            )
        except ValueError as exc:
            return {"error": str(exc)}

        response = result if isinstance(result, dict) else {"result": result}
        return {
            "memory_id": memory_id,
            **response,
        }

    @mcp.tool()
    async def delete_memory(memory_id: str) -> dict:
        """Delete a single memory by ID.

        Use this when the user asks to forget or delete a specific remembered
        item and provides its memory ID.

        Args:
            memory_id: ID of the memory to delete.
        """
        if not memory_id or not memory_id.strip():
            return {"error": "memory_id must be a non-empty string"}

        try:
            result = _memory.delete(memory_id=memory_id)
        except ValueError as exc:
            return {"error": str(exc)}

        response = result if isinstance(result, dict) else {"result": result}
        return {
            "memory_id": memory_id,
            **response,
        }

    @mcp.tool()
    async def delete_all_memories(
        scope: str,
        team_id: str | None = None,
        project_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Delete all memories under a resolved scope.

        Use this only when the user explicitly asks to delete or forget all
        memories for an org, team, project, or user scope. This maps the scope
        to the same internal scope_key used by store_memory and deletes every
        memory in that partition.

        Args:
            scope: The scope level - one of 'org', 'team', 'project', or 'user'.
            team_id: Override the team_id from the Identity Header (only used when scope='team' or 'project').
            project_id: Override the project_id from the Identity Header (only used when scope='project').
            user_id: Required when scope='user'; ignored for all other scopes.
        """
        try:
            request = get_http_request()
            header_value = request.headers.get("x-memory-context") or request.headers.get("X-Memory-Context")
            parsed_ctx = parse_memory_context(header_value)
        except ValueError as exc:
            return {"error": str(exc)}

        if parsed_ctx is not None:
            if team_id is not None:
                parsed_ctx = {**parsed_ctx, "team_id": team_id}
            if project_id is not None:
                parsed_ctx = {**parsed_ctx, "project_id": project_id}

        try:
            scope_key = resolve_scope_key(scope, parsed_ctx, user_id=user_id)
            result = _memory.delete_all(user_id=scope_key)
        except ValueError as exc:
            return {"error": str(exc)}

        response = result if isinstance(result, dict) else {"result": result}
        return {
            "scope_key": scope_key,
            **response,
        }

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
