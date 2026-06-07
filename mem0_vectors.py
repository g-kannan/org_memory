import argparse
import json
import os
import re
import logging
from contextlib import contextmanager

import boto3
from botocore.exceptions import ClientError
from mem0.configs.llms.aws_bedrock import AWSBedrockConfig
from mem0 import Memory as Mem0Memory
from mem0.llms.aws_bedrock import AWSBedrockLLM
from mem0.memory.main import _safe_deepcopy_config
from mem0.utils.scoring import ENTITY_BOOST_WEIGHT
from mem0.utils.factory import LlmFactory, VectorStoreFactory
from mem0.vector_stores.s3_vectors import S3Vectors, OutputData
from ollama import Client
from dotenv import load_dotenv

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

load_dotenv()
logger = logging.getLogger(__name__)
_langfuse_auth_checked = False
_langfuse_auth_ok = False

# Ensure your AWS credentials are configured in your environment
# e.g., by setting AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.1:8b"
DEFAULT_BEDROCK_MODEL = "openai.gpt-oss-20b-1:0"


def langfuse_is_configured() -> bool:
    global _langfuse_auth_checked, _langfuse_auth_ok

    if not (
        get_langfuse_client
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return False

    if _langfuse_auth_checked:
        return _langfuse_auth_ok

    _langfuse_auth_checked = True
    try:
        _langfuse_auth_ok = bool(get_langfuse_client().auth_check())
    except Exception as exc:
        logger.warning("Langfuse auth check failed: %s", exc)
        _langfuse_auth_ok = False

    if not _langfuse_auth_ok:
        logger.warning("Langfuse is configured but authentication failed; traces will be skipped.")

    return _langfuse_auth_ok


@contextmanager
def langfuse_observation(name: str, as_type: str = "span", **kwargs):
    if not langfuse_is_configured():
        yield None
        return

    try:
        langfuse = get_langfuse_client()
        with langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs) as observation:
            yield observation
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


class BedrockOpenAILLM(AWSBedrockLLM):
    """Bedrock OpenAI model adapter using AWS's documented chat-completion body."""

    def _build_openai_messages(self, messages):
        openai_messages = []
        for message in messages:
            role = message.get("role", "user")
            if role == "developer":
                role = "system"

            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            openai_messages.append({"role": role, "content": content})

        return openai_messages

    def _parse_openai_response(self, response):
        response_body = response.get("body").read().decode("utf-8")
        response_json = json.loads(response_body)
        choices = response_json.get("choices", [])
        if not choices:
            return str(response_json)

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return content

    def _extract_json_response(self, text: str) -> str:
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
        decoder = json.JSONDecoder()
        decoded_values = []

        for index, char in enumerate(text):
            if char not in "[{":
                continue

            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue

            decoded_values.append(value)

        if not decoded_values:
            return text

        for value in reversed(decoded_values):
            if isinstance(value, dict) and isinstance(value.get("memory"), list):
                return json.dumps(value)

        value = decoded_values[-1]
        if isinstance(value, list):
            return json.dumps({"memory": value})

        if isinstance(value, dict):
            for nested_value in value.values():
                if isinstance(nested_value, list):
                    return json.dumps({"memory": nested_value})

        return json.dumps(value)

    def generate_response(
        self,
        messages,
        response_format=None,
        tools=None,
        tool_choice="auto",
        stream=False,
        **kwargs,
    ):
        if tools:
            logger.warning("Ignoring tools for Bedrock OpenAI model '%s'", self.config.model)

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

        with langfuse_observation(
            name="bedrock-openai-generate",
            as_type="generation",
            input=request_body["messages"],
            model=self.config.model,
            model_parameters={
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
            metadata={"provider": "aws_bedrock"},
        ) as generation:
            response = self.client.invoke_model(
                body=json.dumps(request_body),
                modelId=self.config.model,
                accept="application/json",
                contentType="application/json",
            )
            parsed_response = self._parse_openai_response(response)
            if generation:
                generation.update(output=parsed_response)

        if response_format:
            return self._extract_json_response(parsed_response)

        return parsed_response


def s3_safe_index_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]", "-", name.strip()).strip("-")
    if not cleaned:
        raise ValueError("S3 Vectors index name cannot be empty")
    if len(cleaned) > 63:
        cleaned = cleaned[:63].rstrip("-")
    if len(cleaned) < 3:
        raise ValueError(f"S3 Vectors index name '{cleaned}' must be at least 3 characters")
    return cleaned


def get_embedding_dims(model: str) -> int:
    client = Client()
    response = client.embed(model=model, input="dimension probe")
    embeddings = response.get("embeddings") or []
    if not embeddings:
        raise ValueError(f"Ollama returned no embeddings for model '{model}'")
    return len(embeddings[0])


def validate_s3_vector_index_dims(expected_dims: int, collection: str) -> None:
    vector_bucket = os.getenv("VECTOR_BUCKET")
    region = os.getenv("AWS_DEFAULT_REGION")

    client = boto3.client("s3vectors", region_name=region)
    try:
        response = client.get_index(vectorBucketName=vector_bucket, indexName=collection)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NotFoundException":
            return
        raise

    index = response.get("index", {})
    actual_dims = index.get("dimension")

    if actual_dims != expected_dims:
        existing_vectors = client.list_vectors(
            vectorBucketName=vector_bucket,
            indexName=collection,
            maxResults=1,
            returnData=False,
            returnMetadata=False,
        ).get("vectors", [])

        if not existing_vectors:
            client.delete_index(vectorBucketName=vector_bucket, indexName=collection)
            return

        raise ValueError(
            f"S3 Vectors index '{collection}' has dimension {actual_dims}, but "
            f"Ollama model '{EMBEDDING_MODEL}' returns {expected_dims}. "
            "Create a new COLLECTION or delete/recreate the existing index with "
            f"dimension {expected_dims}."
        )


class S3VectorsSimilarity(S3Vectors):
    def _distance_to_similarity(self, distance: float | None) -> float | None:
        if distance is None:
            return None

        if self.distance_metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))

        if self.distance_metric == "euclidean":
            return 1.0 / (1.0 + distance)

        return distance

    def _parse_output(self, vectors):
        results = []
        for vector in vectors:
            payload = vector.get("metadata", {})
            if isinstance(payload, str):
                import json

                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata for key {vector.get('key')}")
                    payload = {}

            results.append(
                OutputData(
                    id=vector.get("key"),
                    score=self._distance_to_similarity(vector.get("distance")),
                    payload=payload,
                )
            )
        return results


VectorStoreFactory.provider_to_class["s3_vectors"] = f"{__name__}.S3VectorsSimilarity"
DEFAULT_AWS_BEDROCK_FACTORY = LlmFactory.provider_to_class["aws_bedrock"]


embedding_dims = get_embedding_dims(EMBEDDING_MODEL)
collection_name = s3_safe_index_name(os.getenv("COLLECTION", "mem0"))
entity_collection_name = s3_safe_index_name(f"{collection_name}-entities")
validate_s3_vector_index_dims(embedding_dims, collection_name)
validate_s3_vector_index_dims(embedding_dims, entity_collection_name)


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
                self.config.vector_store.provider,
                entity_config,
            )
        return self._entity_store

    def _compute_entity_boosts(self, query_entities, filters):
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            for _, entity_text in deduped:
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                matches = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=100,
                    filters=search_filters,
                )

                for match in matches:
                    similarity = match.score if hasattr(match, "score") else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, "payload") else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts


Memory = S3SafeMemory


def build_llm_config(provider: str | None = None, model: str | None = None) -> dict:
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()

    if provider in ("bedrock", "aws_bedrock"):
        bedrock_model = model or os.getenv("BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
        if bedrock_model.startswith("openai."):
            LlmFactory.provider_to_class["aws_bedrock"] = (f"{__name__}.BedrockOpenAILLM", AWSBedrockConfig)
        else:
            LlmFactory.provider_to_class["aws_bedrock"] = DEFAULT_AWS_BEDROCK_FACTORY

        return {
            "provider": "aws_bedrock",
            "config": {
                "model": bedrock_model,
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
                "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
                "aws_region": (
                    os.getenv("BEDROCK_AWS_REGION")
                    or os.getenv("AWS_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "us-west-2"
                ),
            },
        }

    if provider != "ollama":
        raise ValueError("LLM_PROVIDER must be 'ollama' or 'aws_bedrock'")

    return {
        "provider": "ollama",
        "config": {
            "model": model or os.getenv("OLLAMA_LLM_MODEL", DEFAULT_OLLAMA_LLM_MODEL),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
        },
    }


def build_memory_config(llm_provider: str | None = None, llm_model: str | None = None) -> dict:
    return {
        "vector_store": {
            "provider": "s3_vectors",
            "config": {
                "vector_bucket_name": os.getenv("VECTOR_BUCKET"),
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dims,
                "distance_metric": "cosine",
                "region_name": os.getenv("AWS_DEFAULT_REGION"),
            },
        },
        "llm": build_llm_config(provider=llm_provider, model=llm_model),
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBEDDING_MODEL,
                "embedding_dims": embedding_dims,
            },
        },
    }


config = build_memory_config()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        choices=("ollama", "aws_bedrock", "bedrock"),
        default=os.getenv("LLM_PROVIDER", "ollama"),
        help="LLM backend for Mem0 extraction; embeddings remain local via Ollama.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model ID for the selected LLM backend. For Bedrock OpenAI GPT OSS, "
            "use openai.gpt-oss-20b-1:0."
        ),
    )
    args = parser.parse_args()

    selected_config = build_memory_config(llm_provider=args.llm, llm_model=args.llm_model)
    messages = [
        {
            "role": "user",
            "content": (
                "We need an architecture decision for querying existing BigQuery datasets "
                "from our Databricks analytics workspace without copying data into Delta first."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Decision: use Databricks Lakehouse Federation as the integration pattern "
                "for BigQuery. It lets Databricks users query BigQuery tables through "
                "Unity Catalog foreign catalogs while keeping BigQuery as the system of record."
            ),
        },
        {
            "role": "user",
            "content": (
                "Capture the rationale: we want governed cross-platform access, fewer ETL "
                "pipelines, centralized discovery in Unity Catalog, and the option to migrate "
                "hot datasets into Delta later if performance or cost requires it."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Recorded ADR: choose Databricks Lakehouse Federation for BigQuery access. "
                "Consequences include managing BigQuery connection credentials, monitoring "
                "federated query cost and latency, and documenting when datasets should be "
                "materialized into Delta Lake instead."
            ),
        },
    ]
    query = "What architecture decision was made for accessing BigQuery from Databricks?"
    filters = {"user_id": "architecture"}
    metadata = {"category": "adr", "decision": "bigquery-federation"}

    try:
        with langfuse_observation(
            name="mem0-vectors-run",
            input={"messages": messages, "query": query, "filters": filters},
            metadata={
                "llm_provider": selected_config["llm"]["provider"],
                "llm_model": selected_config["llm"]["config"]["model"],
                "embedder_provider": selected_config["embedder"]["provider"],
                "embedder_model": selected_config["embedder"]["config"]["model"],
                "vector_store_provider": selected_config["vector_store"]["provider"],
                "collection": collection_name,
            },
        ) as run_span:
            with langfuse_observation(name="mem0-create-memory", metadata={"collection": collection_name}):
                m = Memory.from_config(selected_config)

            with langfuse_observation(name="mem0-add", input=messages, metadata=metadata) as add_span:
                add_result = m.add(messages, user_id="architecture", metadata=metadata)
                if add_span:
                    add_span.update(output=add_result)

            with langfuse_observation(name="mem0-search", input={"query": query, "filters": filters}) as search_span:
                results = m.search(query, filters=filters)
                if search_span:
                    search_span.update(output=results)

            if run_span:
                run_span.update(output=results)

            print(results)
    finally:
        flush_langfuse()


if __name__ == "__main__":
    main()
