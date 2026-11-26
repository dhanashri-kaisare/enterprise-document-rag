import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="📄",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 Enterprise Document Assistant")
st.caption("Upload documents, filter by file, and receive validated answers powered by Hybrid Search + Cross-Encoder Reranking.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if st.button("Upload Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a document first.")
        else:
            try:
                with st.spinner("Processing and indexing document..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream"
                        )
                    }
                    response = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Document uploaded successfully!")
                    st.write(f"**Filename:** {result.get('filename')}")
                    st.write(f"**Chunks created:** {result.get('chunks_created')}")
                    st.write(f"**Characters extracted:** {result.get('characters_extracted')}")
                else:
                    st.error(f"Upload failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Start the FastAPI server first.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

    # ---------------- INDEXED DOCUMENTS & FILTER ----------------
    st.divider()
    st.subheader("📚 Indexed Documents")

    documents = []
    try:
        response = requests.get(f"{API_URL}/documents/list", timeout=10)
        if response.status_code == 200:
            documents = response.json().get("documents", [])
    except requests.exceptions.RequestException:
        st.caption("Start the API to view indexed documents.")

    selected_scope = st.selectbox(
        "🔍 Search Scope",
        options=["All Documents"] + documents,
        index=0,
    )

    if documents:
        for idx, filename in enumerate(documents):
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {filename}")
            if col2.button("🗑️", key=f"del_{idx}_{filename}", help=f"Delete {filename}"):
                del_resp = requests.delete(f"{API_URL}/documents/{filename}", timeout=30)
                if del_resp.status_code == 200:
                    st.toast(f"Deleted {filename}")
                    st.rerun()
                else:
                    st.error(f"Failed to delete {filename}")
    else:
        st.caption("No indexed documents yet.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            for source in message.get("sources", []):
                with st.expander(f"📄 Source: {source.get('filename', 'Unknown')}"):
                    st.write(source.get("content", ""))
                    st.caption(
                        f"Cosine similarity: {source.get('cosine_similarity', 'N/A')} | "
                        f"Cross-Encoder rerank score: {source.get('rerank_score', 'N/A')}"
                    )

            validation = message.get("validation")
            if validation:
                if validation.get("is_supported"):
                    st.success("Answer validation: SUPPORTED")
                else:
                    st.warning("Answer validation: " + validation.get("validation", "NOT_SUPPORTED"))

# ---------------- QUESTION INPUT ----------------
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            payload = {"question": question}
            if selected_scope != "All Documents":
                payload["filename"] = selected_scope

            with st.spinner("Running Hybrid Search + Cross-Encoder Reranking..."):
                response = requests.post(f"{API_URL}/questions/ask", json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer was returned.")
                sources = result.get("sources", [])
                validation = result.get("validation", {})

                st.markdown(answer)

                if sources:
                    st.subheader("📚 Retrieved Sources")
                    for source in sources:
                        with st.expander(f"📄 {source.get('filename', 'Unknown source')}"):
                            st.write(source.get("content", ""))
                            st.caption(
                                f"Cosine similarity: {source.get('cosine_similarity', 'N/A')} | "
                                f"Cross-Encoder rerank score: {source.get('rerank_score', 'N/A')}"
                            )
                else:
                    st.info("No relevant document sources were retrieved.")

                if validation.get("is_supported"):
                    st.success("Answer validation: SUPPORTED")
                else:
                    st.warning("Answer validation: " + validation.get("validation", "NOT_SUPPORTED"))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "validation": validation,
                })
            else:
                error_message = f"API error ({response.status_code}): {response.text}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": [], "validation": None})

        except requests.exceptions.ConnectionError:
            error_message = "Cannot connect to FastAPI. Please start the backend server."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": [], "validation": None})
        except requests.exceptions.RequestException as exc:
            error_message = f"Request failed: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": [], "validation": None})