"""Gradio UI for the Agentic RAG system.

Two tabs:
  - Chat: send messages to the graph (with persistent thread_id memory).
           Supports interrupt-based outline approval for the academic agent.
  - Papers: ingest PDFs into Chroma, list indexed papers.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import gradio as gr
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from src.config import SETTINGS
from src.graph import get_graph
from src.rag.ingest import ingest_directory, ingest_pdf
from src.rag.store import list_papers
from src.tracing import init_tracing

# Nodes whose AIMessage tokens should be streamed into the UI. Excludes the
# guards/router/expansion (their LLM output is structured JSON, not user-facing
# prose). Subgraph node names (research, outline, paper) are included so the
# academic agent streams during its interior phases too.
_STREAMING_NODES = {"qa", "blog", "academic", "research", "outline", "paper"}


def _new_thread_id() -> str:
    return f"chat-{uuid.uuid4().hex[:8]}"


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _pending_interrupt(graph, thread_id: str):
    """Return the pending interrupt payload, or None if the graph is not paused."""
    snapshot = graph.get_state(_config(thread_id))
    interrupts = getattr(snapshot, "interrupts", None) or []
    if interrupts:
        return interrupts[0].value
    if snapshot.next:
        tasks = snapshot.tasks or ()
        for task in tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


def _last_assistant_text(graph, thread_id: str) -> str:
    snapshot = graph.get_state(_config(thread_id))
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return snapshot.values.get("final_answer", "") if snapshot.values else ""


def _stream_into_history(graph, payload, thread_id, history, user_label):
    """Stream tokens from the graph into the last history slot.

    Yields the running history after each token so Gradio can re-render.
    Returns the final assistant text streamed (caller uses it as a fallback if
    snapshot doesn't yet reflect the new message).
    """
    history.append((user_label, ""))
    yield history, ""

    streamed = ""
    cfg = _config(thread_id)
    for chunk, metadata in graph.stream(payload, config=cfg, stream_mode="messages"):
        node = (metadata or {}).get("langgraph_node", "")
        if node not in _STREAMING_NODES:
            continue
        if not isinstance(chunk, (AIMessage, AIMessageChunk)):
            continue
        content = chunk.content if isinstance(chunk.content, str) else ""
        if not content:
            continue
        streamed += content
        history[-1] = (user_label, streamed)
        yield history, streamed


def chat_handler(message: str, history: list, thread_id: str, style: str, length: str):
    """Generator — streams tokens into the chat as the graph runs."""
    if not message.strip():
        yield history, thread_id, gr.update(visible=False), ""
        return

    graph = get_graph()
    if not thread_id:
        thread_id = _new_thread_id()

    word_count = {"Short (300)": 300, "Medium (500)": 500, "Long (800)": 800}.get(length, 500)
    initial = {
        "messages": [HumanMessage(content=message)],
        "style": style.lower(),
        "word_count": word_count,
    }

    try:
        streamed = ""
        for hist, partial in _stream_into_history(graph, initial, thread_id, history, message):
            streamed = partial
            yield hist, thread_id, gr.update(visible=False), ""
    except Exception as e:
        history[-1] = (message, f"Error: {e}") if history else (message, f"Error: {e}")
        yield history, thread_id, gr.update(visible=False), ""
        return

    # Final state from the checkpointer may have the canonical text (e.g. the
    # outline node's formatted markdown). Prefer it over raw streamed tokens.
    interrupt_payload = _pending_interrupt(graph, thread_id)
    final_text = _last_assistant_text(graph, thread_id) or streamed or "(no response)"
    history[-1] = (message, final_text)

    if interrupt_payload:
        yield (
            history,
            thread_id,
            gr.update(visible=True),
            json.dumps(interrupt_payload, indent=2, default=str),
        )
    else:
        yield history, thread_id, gr.update(visible=False), ""


def approve_outline(history: list, thread_id: str):
    graph = get_graph()
    streamed = ""
    for hist, partial in _stream_into_history(
        graph, Command(resume={"approved": True}), thread_id, history, "[approved outline]"
    ):
        streamed = partial
        yield hist, gr.update(visible=False), ""

    final_text = _last_assistant_text(graph, thread_id) or streamed or "(paper generated)"
    history[-1] = ("[approved outline]", final_text)
    yield history, gr.update(visible=False), ""


def revise_outline(history: list, thread_id: str, feedback: str):
    if not feedback.strip():
        feedback = "Please revise to be more specific and concrete."
    graph = get_graph()
    label = f"[revise: {feedback}]"
    streamed = ""
    for hist, partial in _stream_into_history(
        graph, Command(resume={"approved": False, "feedback": feedback}),
        thread_id, history, label,
    ):
        streamed = partial
        yield hist, gr.update(visible=False), ""

    interrupt_payload = _pending_interrupt(graph, thread_id)
    final_text = _last_assistant_text(graph, thread_id) or streamed or "(revised outline)"
    history[-1] = (label, final_text)

    if interrupt_payload:
        yield (
            history,
            gr.update(visible=True),
            json.dumps(interrupt_payload, indent=2, default=str),
        )
    else:
        yield history, gr.update(visible=False), ""


def reset_thread():
    return [], _new_thread_id(), gr.update(visible=False), ""


def ingest_existing():
    results = ingest_directory()
    if not results:
        return "No PDFs found in datasets/."
    return "\n".join(
        f"[{r['status']:>7}] {r['paper']}  chunks={r['chunks']}" for r in results
    )


def upload_and_ingest(files):
    if not files:
        return "No files uploaded."
    out = []
    for f in files:
        src = Path(f.name)
        dest = SETTINGS.datasets_dir / src.name
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        result = ingest_pdf(dest)
        out.append(f"[{result['status']:>7}] {result['paper']}  chunks={result['chunks']}")
    return "\n".join(out)


def refresh_papers():
    papers = list_papers()
    return "\n".join(f"- {p}" for p in papers) if papers else "(none indexed yet)"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Agentic RAG\nMulti-agent research assistant powered by LangGraph + Chroma.")

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=520, show_copy_button=True)
                        msg = gr.Textbox(
                            placeholder="Ask a question, request a blog, or generate an academic paper...",
                            label="Message",
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            reset_btn = gr.Button("New thread")

                    with gr.Column(scale=1):
                        thread_id = gr.Textbox(
                            value=_new_thread_id(),
                            label="Thread ID",
                            info="Reuse to continue a session.",
                        )
                        style = gr.Radio(
                            ["Technical", "Professional", "Casual"],
                            value="Professional",
                            label="Writing style",
                        )
                        length = gr.Radio(
                            ["Short (300)", "Medium (500)", "Long (800)"],
                            value="Medium (500)",
                            label="Length (for blogs)",
                        )

                with gr.Group(visible=False) as approval_box:
                    gr.Markdown("### Outline approval required")
                    interrupt_view = gr.Code(language="json", label="Interrupt payload")
                    feedback = gr.Textbox(label="Revision feedback (optional)")
                    with gr.Row():
                        approve_btn = gr.Button("Approve & generate paper", variant="primary")
                        revise_btn = gr.Button("Revise outline")

            with gr.Tab("Papers"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Ingest")
                        ingest_btn = gr.Button("Ingest all PDFs from datasets/")
                        upload = gr.File(label="Or upload PDFs", file_count="multiple", file_types=[".pdf"])
                        upload_btn = gr.Button("Upload & ingest")
                        ingest_log = gr.Textbox(label="Output", lines=10)
                    with gr.Column():
                        gr.Markdown("### Indexed papers")
                        refresh_btn = gr.Button("Refresh")
                        paper_list = gr.Textbox(label="Papers", lines=10, value=refresh_papers())

        send_btn.click(
            chat_handler,
            [msg, chatbot, thread_id, style, length],
            [chatbot, thread_id, approval_box, interrupt_view],
        ).then(lambda: "", None, msg)

        msg.submit(
            chat_handler,
            [msg, chatbot, thread_id, style, length],
            [chatbot, thread_id, approval_box, interrupt_view],
        ).then(lambda: "", None, msg)

        reset_btn.click(
            reset_thread, None, [chatbot, thread_id, approval_box, interrupt_view]
        )

        approve_btn.click(
            approve_outline,
            [chatbot, thread_id],
            [chatbot, approval_box, interrupt_view],
        )
        revise_btn.click(
            revise_outline,
            [chatbot, thread_id, feedback],
            [chatbot, approval_box, interrupt_view],
        ).then(lambda: "", None, feedback)

        ingest_btn.click(ingest_existing, None, ingest_log).then(
            refresh_papers, None, paper_list
        )
        upload_btn.click(upload_and_ingest, upload, ingest_log).then(
            refresh_papers, None, paper_list
        )
        refresh_btn.click(refresh_papers, None, paper_list)

    return demo


if __name__ == "__main__":
    init_tracing()
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
