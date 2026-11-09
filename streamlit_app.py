import streamlit as st
import requests

# ---------------- CONFIGURATION ----------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- HEADER ----------------
st.title("📄 Enterprise Document Assistant")
st.caption(
    "Upload documents, ask questions, and receive answers "
    "grounded in your documents."
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document",
        type=["pdf", "txt"]
    )

    if st.button("Upload Document", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a document first.")

        else:
            try:
                with st.spinner("Processing document..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream"
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        files=files,
                        timeout=120
                    )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Document uploaded successfully!")

                    st.write(
                        f"**Filename:** {result.get('filename')}"
                    )
                    st.write(
                        f"**Chunks created:** "
                        f"{result.get('chunks_created')}"
                    )
                    st.write(
                        f"**Characters extracted:** "
                        f"{result.get('characters_extracted')}"
                    )

                else:
                    st.error(f"Upload failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    "Start the FastAPI server first."
                )

            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

    # ---------------- INDEXED DOCUMENTS ----------------
    st.divider()
    st.subheader("📚 Indexed Documents")

    try:
        response = requests.get(
            f"{API_URL}/documents/list",
            timeout=10
        )

        if response.status_code == 200:
            documents = response.json().get("documents", [])

            if documents:
                for filename in documents:
                    st.write(f"📄 {filename}")
            else:
                st.caption("No indexed documents yet.")

        else:
            st.warning("Could not load indexed documents.")

    except requests.exceptions.RequestException:
        st.caption("Start the API to view indexed documents.")

    # ---------------- CLEAR CHAT ----------------
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
                with st.expander(
                    f"📄 Source: {source.get('filename', 'Unknown')}"
                ):
                    st.write(source.get("content", ""))

                    if "cosine_similarity" in source:
                          st.caption(
                             f"Cosine similarity: {source['cosine_similarity']}"
                          )
            validation = message.get("validation")

            if validation:
                if validation.get("is_supported"):
                    st.success("Answer validation: SUPPORTED")
                else:
                    st.warning(
                        "Answer validation: "
                        + validation.get(
                            "validation",
                            "NOT_SUPPORTED"
                        )
                    )


# ---------------- QUESTION INPUT ----------------
question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching documents and generating an answer..."
            ):
                response = requests.post(
                    f"{API_URL}/questions/ask",
                    json={"question": question},
                    timeout=120
                )

            if response.status_code == 200:

                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer was returned."
                )

                sources = result.get("sources", [])
                validation = result.get("validation", {})

                # Display answer
                st.markdown(answer)

                # Display retrieved sources
                if sources:
                    st.subheader("📚 Retrieved Sources")

                    for source in sources:
                        with st.expander(
                            f"📄 {source.get('filename', 'Unknown source')}"
                        ):
                            st.write(source.get("content", ""))

                            st.caption(
                             f"Cosine similarity: "
                             f"{source.get('cosine_similarity', 'N/A')}"
)

                else:
                    st.info(
                        "No relevant document sources were retrieved."
                    )

                # Display validation status
                if validation.get("is_supported"):
                    st.success("Answer validation: SUPPORTED")
                else:
                    st.warning(
                        "Answer validation: "
                        + validation.get(
                            "validation",
                            "NOT_SUPPORTED"
                        )
                    )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "validation": validation
                })

            else:
                error_message = (
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                    "validation": None
                })

        except requests.exceptions.ConnectionError:

            error_message = (
                "Cannot connect to FastAPI. "
                "Please start the backend server."
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "validation": None
            })

        except requests.exceptions.RequestException as exc:

            error_message = f"Request failed: {exc}"

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "validation": None
            })