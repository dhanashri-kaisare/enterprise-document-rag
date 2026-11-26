from pathlib import Path
from uuid import uuid4
import gradio as gr

from app.services.document_pipeline import process_document
from app.services.rag_service import answer_question, reset_caches
from app.services.validation_service import validate_answer
from app.services.vector_store import delete_document_from_store, load_vector_store

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_indexed_documents() -> list[str]:
    vector_store = load_vector_store()
    if vector_store is None:
        return []
    docs = {
        doc.metadata.get("filename")
        for doc in vector_store.docstore._dict.values()
        if doc.metadata.get("filename")
    }
    return sorted(docs)


def upload_file(file):
    if file is None:
        return "⚠️ Please select a PDF or TXT file first.", gr.update(), gr.update()

    file_path = Path(file.name)
    if file_path.suffix.lower() not in {".pdf", ".txt"}:
        return "❌ Only PDF and TXT files are supported.", gr.update(), gr.update()

    saved_path = DATA_DIR / f"{uuid4().hex}{file_path.suffix.lower()}"
    saved_path.write_bytes(file_path.read_bytes())

    try:
        result = process_document(str(saved_path), original_filename=file_path.name)
        reset_caches()
        docs = get_indexed_documents()
        status = (
            f"✅ **{file_path.name}** indexed successfully!\n\n"
            f"- **Chunks created:** {len(result['chunks'])}\n"
            f"- **Characters extracted:** {len(result['text'])}"
        )
        return (
            status,
            gr.update(choices=["All Documents"] + docs, value="All Documents"),
            gr.update(choices=docs, value=None),
        )
    except Exception as exc:
        return f"❌ Upload failed: {exc}", gr.update(), gr.update()


def remove_document(filename):
    if not filename:
        return "⚠️ Select a document to delete.", gr.update(), gr.update()

    deleted = delete_document_from_store(filename)
    reset_caches()
    docs = get_indexed_documents()
    return (
        f"🗑️ Deleted **{filename}** ({deleted} chunks removed).",
        gr.update(choices=["All Documents"] + docs, value="All Documents"),
        gr.update(choices=docs, value=None),
    )


def ask_bot(message, history, scope):
    if not message.strip():
        return history, ""

    filename_filter = None if scope == "All Documents" else scope

    try:
        result = answer_question(message, filename_filter=filename_filter)
        if not result.get("context"):
            val_badge = "⚠️ **Answer Validation:** `NOT_SUPPORTED`"
        else:
            val = validate_answer(context=result["context"], answer=result["answer"])
            status = val.get("validation", "NOT_SUPPORTED")
            icon = "✅" if val.get("is_supported") else "⚠️"
            val_badge = f"{icon} **Answer Validation:** `{status}`"

        sources_md = ""
        if result["sources"]:
            lines = ["\n\n---\n**📚 Retrieved & Reranked Sources:**"]
            for s in result["sources"]:
                lines.append(
                    f"- **{s['filename']}** *(Cosine: `{s['cosine_similarity']}` | Rerank: `{s['rerank_score']}`)*\n"
                    f"  > {s['content'][:220].strip()}..."
                )
            sources_md = "\n".join(lines)

        full_reply = f"{result['answer']}\n\n{val_badge}{sources_md}"
    except Exception as exc:
        full_reply = f"❌ Error: {exc}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full_reply},
    ]
    return history, ""


initial_docs = get_indexed_documents()

with gr.Blocks(title="Enterprise Document RAG Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📄 Enterprise Document Retrieval & Context Validation Assistant\n"
        "Powered by **Hybrid Search (FAISS + BM25 via RRF)**, **Cross-Encoder Reranking**, and **LLM Groundedness Validation**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Document Management")
            file_input = gr.File(label="Upload PDF or TXT", file_types=[".pdf", ".txt"])
            upload_btn = gr.Button("Upload & Index Document", variant="primary")
            upload_status = gr.Markdown()

            gr.Markdown("### 🔍 Search Scope & Delete")
            scope_dropdown = gr.Dropdown(
                choices=["All Documents"] + initial_docs,
                value="All Documents",
                label="Filter Queries by Document",
            )
            delete_dropdown = gr.Dropdown(
                choices=initial_docs,
                value=None,
                label="Select Document to Delete",
            )
            delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=480, label="Grounded Document Q&A")
            msg_box = gr.Textbox(
                placeholder="Ask a question about your uploaded documents...",
                label="Your Question",
            )
            with gr.Row():
                send_btn = gr.Button("Ask Question", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    upload_btn.click(
        upload_file,
        inputs=[file_input],
        outputs=[upload_status, scope_dropdown, delete_dropdown],
    )
    delete_btn.click(
        remove_document,
        inputs=[delete_dropdown],
        outputs=[upload_status, scope_dropdown, delete_dropdown],
    )
    send_btn.click(ask_bot, inputs=[msg_box, chatbot, scope_dropdown], outputs=[chatbot, msg_box])
    msg_box.submit(ask_bot, inputs=[msg_box, chatbot, scope_dropdown], outputs=[chatbot, msg_box])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()