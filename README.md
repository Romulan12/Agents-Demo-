# Agentic RAG

A multi-agent research assistant built on **LangGraph**. A router supervises three specialized agents — Q&A, blog writer, and academic paper writer — each running a ReAct loop over a whitelisted toolset (vector search, arXiv, web). Guardrails, query expansion, and human-in-the-loop are first-class graph nodes. Fully traced via LangSmith.

---

## Features

- **Multi-agent orchestration** — router dispatches to `qa` / `blog` / `academic` based on intent.
- **ReAct inner loops** with autonomous tool selection inside each agent.
- **RAG over local PDFs** via Chroma (one collection per paper + a shared cross-paper collection).
- **Auto-ingest from arXiv** — the `arxiv_download` tool indexes new papers on the fly.
- **Web search + scrape** fallback when indexed papers don't cover the topic.
- **Guardrails as graph nodes** — prompt-injection / scope check on input; hallucination check against actual tool observations on output.
- **Query expansion** — structured rewrite (intent + key concepts + search terms + scope constraints) before blog/academic; skipped for Q&A to keep simple queries fast.
- **Human-in-the-loop** — the academic agent uses LangGraph `interrupt()` to pause after producing an outline, then resumes once the user approves or revises.
- **Persistent per-thread memory** — every conversation is checkpointed to SQLite (`SqliteSaver`), so sessions resume across restarts by `thread_id`.
- **Streaming UI** — Gradio renders tokens as they arrive.
- **Two-tier model routing** — lightweight nodes (guards, router, expansion) use a Haiku-tier model; agents use Sonnet-tier. ~30–50% lower end-to-end latency.
- **LangSmith Hub prompt versioning** — prompts pull from the hub at startup with in-code fallback if hub is unreachable.
- **Provider-agnostic LLM layer** — works against any OpenAI-compatible endpoint (real OpenAI, LiteLLM gateways, Azure proxies, local vLLM). Set `OPENAI_BASE_URL`.

---

## Architecture

```
                         ┌───────────────────────────────────────┐
                         │   app.py — Gradio UI (port 7860)      │
                         │   • Chat tab • Papers tab             │
                         │   • Streaming • Threaded memory       │
                         └────────────────┬──────────────────────┘
                                          │ user message
                                          ▼
                         ┌───────────────────────────────────────┐
                         │   src/graph.py — LangGraph compiled   │
                         │   StateGraph[AgentState]              │
                         │   Checkpoints: SqliteSaver            │
                         │   Tracing: LangSmith (auto)           │
                         └────────────────┬──────────────────────┘
                                          │
                                          ▼
                          ┌─────────────────────────┐
                          │  input_guard            │  fast model
                          └────────┬────────────────┘
                            blocked│        passed
                          ┌────────┘                ▼
                          ▼                ┌─────────────────────┐
                         END               │  router             │  fast model
                                           └────┬────────────────┘
                                                │
                          ┌─────────────────────┼────────────────────────┐
                          │ qa                  │ blog | academic        │
                          ▼                     ▼                        │
                  ┌──────────────┐    ┌─────────────────────┐             │
                  │ qa_agent     │    │  query_expansion    │  fast model │
                  │ ReAct        │    └──────────┬──────────┘             │
                  └──────┬───────┘               ▼                        │
                         │            ┌─────────────────────┐             │
                         │            │ blog_agent  /       │             │
                         │            │ academic_agent      │             │
                         │            │ (HITL via interrupt)│             │
                         │            └──────────┬──────────┘             │
                         └────────────┬──────────┘                        │
                                      ▼                                   │
                          ┌─────────────────────────┐                     │
                          │  output_guard           │  fast model         │
                          └────────┬────────────────┘                     │
                                   ▼                                      │
                                  END  ◄──────────────────────────────────┘
```

Tools live in `src/tools/` and are whitelisted per agent (`QA_TOOLS`, `BLOG_TOOLS`, `ACADEMIC_TOOLS`):

| Tool | Source |
|---|---|
| `vector_search`, `summarize_paper`, `list_indexed_papers` | Chroma |
| `arxiv_search`, `arxiv_download` (auto-ingests) | arXiv |
| `web_search`, `scrape_url` | DuckDuckGo + requests |

---

## Quickstart

```bash
# 1. Create venv and install deps
python -m venv venv_agentic
source venv_agentic/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, FAST_LLM_MODEL,
# EMBEDDING_MODEL, and (optionally) LANGSMITH_* for tracing/prompts.

# 3. (Optional) Pre-index any PDFs in datasets/ into Chroma
python -m src.rag.ingest

# 4. Run
python app.py
```

Open http://127.0.0.1:7860 in a browser.

---

## Configuration

All config is loaded from `.env` at startup. The key vars:

| Var | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | Bearer token for the OpenAI-compatible endpoint. |
| `OPENAI_BASE_URL` | no | Gateway URL. Blank → hits `api.openai.com` directly. |
| `LLM_MODEL` | yes | Main agent model (e.g. `openai:claude-sonnet-4-6`). Format is `provider:model`. |
| `FAST_LLM_MODEL` | no | Cheaper/faster model for guards, router, expansion (e.g. `openai:claude-haiku-4-5`). Blank → reuses `LLM_MODEL`. |
| `EMBEDDING_MODEL` | yes | Embedding model recognized by your endpoint (e.g. `amazon.titan-embed-text-v2:0`, `text-embedding-3-small`). |
| `LANGSMITH_TRACING` | no | `true` to enable LangSmith trace upload. |
| `LANGSMITH_API_KEY` | only if tracing/prompts | LangSmith key. |
| `LANGSMITH_PROJECT` | no | Tracing project name. |
| `LANGSMITH_PROMPTS` | no | `true` to pull prompts from LangSmith Hub at startup. |
| `LANGSMITH_PROMPTS_OWNER` | only if pulling prompts | Your workspace handle/UUID. |
| `LANGSMITH_PROMPTS_REPO` | no | Hub repo prefix for the three prompt entries. |
| `PROMPT_REVISION` | no | `latest` or a specific commit hash. |
| `DATASETS_DIR` / `CHROMA_DIR` / `CHECKPOINT_DB` | no | Filesystem paths. |
| `MAX_REACT_ITERATIONS` | no | Cap on ReAct loop iterations per agent (default 4). |

---

## Project structure

```
app.py                          Gradio UI entrypoint
requirements.txt
.env.example                    Template — copy to .env

src/
  config.py                     Settings dataclass, loads .env
  state.py                      AgentState (TypedDict shared across nodes)
  graph.py                      Main LangGraph wiring
  llm_client.py                 ChatOpenAI / OpenAIEmbeddings factory (provider-agnostic)
  llm.py                        Backward-compat re-exports
  prompts.py                    LangSmith Hub pull with fallback
  tracing.py                    LangSmith trace init

  agents/
    router.py                   Picks qa | blog | academic
    qa_agent.py                 ReAct over QA_TOOLS
    blog_agent.py               ReAct over BLOG_TOOLS (incl. BLOG_SYSTEM_PROMPT)
    academic_agent.py           Subgraph: research → outline → (interrupt) → paper
    query_expansion.py          Structured intent/concepts/terms rewrite

  guardrails/
    input_guard.py              Prompt injection + scope check
    output_guard.py             Hallucination check vs tool observations

  rag/
    ingest.py                   PDF → chunks → Chroma (idempotent per paper)
    store.py                    Chroma collection factory + retriever helpers

  tools/
    __init__.py                 Per-agent tool whitelists
    retrieval.py                vector_search, summarize_paper, list_indexed_papers
    arxiv.py                    arxiv_search, arxiv_download (auto-ingests)
    web.py                      web_search, scrape_url

datasets/                       Drop PDFs here for ingestion (gitignored content)
data/                           Chroma store + checkpoint sqlite (gitignored)
```

---

## How prompts work

Each agent owns its system prompt as a module-level constant in its own file (so the prompt is reviewable in code). At runtime, `pull_prompt_or(<hub-name>, <local-constant>)` tries to pull from LangSmith Hub; if the hub is unreachable, the prompt isn't there, or `LANGSMITH_PROMPTS=false`, it falls back to the local constant. Result is cached per-process.

To version a prompt:

1. Edit the constant in code (e.g. `BLOG_SYSTEM_PROMPT` in `src/agents/blog_agent.py`).
2. Either run `python scripts/push_prompts.py` (if your key has write permissions) or paste the new text into the corresponding hub entry in the LangSmith UI.
3. Restart the app — the new prompt is loaded.

---

## How the academic HITL flow works

1. `router` picks `academic`.
2. `query_expansion` enriches the request.
3. Inside the academic subgraph: `research` (ReAct over arXiv + vector search) → `outline` (structured Pydantic outline) → `interrupt`.
4. The graph **pauses**. Gradio surfaces the outline and an approval/revision form.
5. User approves → graph resumes → `paper` node generates section-by-section.
6. User revises → graph resumes with feedback → `outline` regenerates and pauses again.
7. Final paper goes through `output_guard` then back to the user.

State is checkpointed throughout, so you can close the browser and resume from the same outline next time you load the thread.

---

## Adding a new agent

1. Write `src/agents/my_agent.py` exposing `my_node(state) -> dict`.
2. Add a tool whitelist in `src/tools/__init__.py` (or reuse an existing one).
3. Register it in `src/graph.py`: `g.add_node("my", my_node)` + edge from router.
4. Add `"my"` to `Route` in `src/state.py` and to the router's `RouteDecision` enum in `src/agents/router.py`.
5. Update the router system prompt to know when to pick `my`.

---

## Tracing & debugging

If `LANGSMITH_TRACING=true`, every graph run produces a hierarchical trace in your LangSmith project (`LANGSMITH_PROJECT`). Each node is a child run; tool calls, prompts, and token usage are all captured. Useful for:

- Seeing which agent the router picked and why.
- Inspecting the query-expansion output that fed retrieval.
- Diagnosing why the output guard blocked a response.
- Comparing latency across model tiers.

