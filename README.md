# lastmem

Mem0 experiment using:

- S3 Vectors as the vector store
- Ollama for local embeddings
- Ollama or Amazon Bedrock for Mem0 LLM extraction
- Langfuse for optional observability

The default run uses local Ollama for both LLM extraction and embeddings. Bedrock is optional and only replaces the LLM; embeddings stay local.

## Requirements

- Python managed by `uv`
- Ollama running locally
- Ollama models pulled:

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull llama3.1:8b
```

- AWS credentials with access to:
  - S3 Vectors bucket/index operations
  - Bedrock model invocation, only when using `--llm aws_bedrock`
- Langfuse project keys, only when observability is enabled

## Environment

Create a local `.env` file. It is ignored by git.

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-south-1
VECTOR_BUCKET=orgmem-vector
COLLECTION=orgmem-vector-ix
```

Optional LLM settings:

```env
LLM_PROVIDER=ollama
OLLAMA_LLM_MODEL=llama3.1:8b
BEDROCK_MODEL=openai.gpt-oss-20b-1:0
BEDROCK_AWS_REGION=us-west-2
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
LLM_TOP_P=0.9
```

Optional Langfuse settings:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Defaults

- Embedding provider: `ollama`
- Embedding model: `qwen3-embedding:0.6b`
- LLM provider: `ollama`
- Ollama LLM model: `llama3.1:8b`
- Bedrock LLM model, when Bedrock is selected: `openai.gpt-oss-20b-1:0`
- Vector store provider: `s3_vectors`
- Distance metric: `cosine`
- Collection: `COLLECTION` env var, or `mem0`
- Entity collection: `<collection>-entities`
- Langfuse: enabled automatically when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present

## Run With Defaults

Uses local Ollama for the LLM and local Ollama for embeddings.

```powershell
uv run python mem0_vectors.py
```

Equivalent:

```powershell
uv run python mem0_vectors.py --llm ollama
```

## Run With Bedrock

Uses Amazon Bedrock for the Mem0 LLM and keeps embeddings local through Ollama.

```powershell
uv run python mem0_vectors.py --llm aws_bedrock
```

The short alias also works:

```powershell
uv run python mem0_vectors.py --llm bedrock
```

Specify a Bedrock model:

```powershell
uv run python mem0_vectors.py --llm aws_bedrock --llm-model openai.gpt-oss-20b-1:0
```

For non-OpenAI Bedrock models, pass the Bedrock model ID. The script uses Mem0's built-in `aws_bedrock` provider for those models.

```powershell
uv run python mem0_vectors.py --llm aws_bedrock --llm-model anthropic.claude-3-5-haiku-20241022-v1:0
```

## CLI Options

```text
--llm {ollama,aws_bedrock,bedrock}
    LLM backend for Mem0 extraction. Embeddings remain local via Ollama.

--llm-model MODEL_ID
    Model ID for the selected LLM backend.
```

## Observability

When Langfuse keys are present, the script emits spans for:

- The full `mem0_vectors.py` run
- Memory client creation
- `m.add(...)`
- `m.search(...)`
- Bedrock OpenAI generation calls, when using `openai.*` Bedrock models

The script calls `flush()` before exit so traces are sent for short CLI runs.
It also runs `auth_check()` once; if the keys do not match `LANGFUSE_BASE_URL`, traces are skipped and the memory flow continues.

## Notes

- `gpt-oss-20b` on Bedrock uses model ID `openai.gpt-oss-20b-1:0`.
- The script includes a local adapter for Bedrock OpenAI models because they use AWS's OpenAI-style chat-completion request and response body.
- If an existing S3 Vectors index has the wrong embedding dimension and contains vectors, the script raises an error instead of deleting data.
- Never commit `.env` or AWS credentials.
