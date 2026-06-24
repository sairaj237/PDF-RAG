import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

DEFAULT_SYSTEM_PROMPT = """You are a helpful PDF Q&A assistant. Follow these rules:

1. Answer ONLY using the provided context below. Never use outside knowledge.
2. Synthesize information across all context chunks to give a complete answer.
3. Use bullet points or numbered lists when the answer has multiple items.
4. Only say you don't have enough information if the context genuinely contains nothing relevant.
5. When referencing specific content, cite the page number if available.
6. Never repeat the question. Never add disclaimers or filler text.

Context from PDF:
{context}"""

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AskThePDF", page_icon="📄", layout="centered")
st.title("📄 AskThePDF")

# ─── Session state ────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Embedding PDF…"):
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
            )
        if resp.ok:
            st.success(resp.json().get("message", "Uploaded!"))
            st.session_state.pdf_uploaded = True
            st.session_state.messages = []
        else:
            st.error(f"Upload failed: {resp.text}")

    if st.session_state.pdf_uploaded:
        st.info("PDF is ready. Ask away!")

    st.divider()

    with st.expander("⚙️ System Prompt", expanded=False):
        st.caption("Customize how the assistant behaves. Use `{context}` as a placeholder for retrieved PDF content.")
        edited_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=300,
            label_visibility="collapsed",
        )
        col1, col2 = st.columns(2)
        if col1.button("Apply", use_container_width=True):
            st.session_state.system_prompt = edited_prompt
            st.success("Prompt updated.")
        if col2.button("Reset", use_container_width=True):
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear & Reset", use_container_width=True):
        with st.spinner("Clearing…"):
            requests.delete(f"{API_BASE}/clear")
        st.session_state.pdf_uploaded = False
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ─── Chat area ────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.pdf_uploaded:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    if prompt := st.chat_input("Ask a question about your PDF…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                        "system_prompt": st.session_state.system_prompt,
                    },
                )
            if resp.ok:
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources", expanded=False):
                        for i, src in enumerate(sources, 1):
                            page = f" (page {src['page']})" if src.get("page") is not None else ""
                            st.markdown(f"**Chunk {i}{page}**")
                            st.caption(src["content"])
            else:
                answer = f"Error: {resp.text}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
