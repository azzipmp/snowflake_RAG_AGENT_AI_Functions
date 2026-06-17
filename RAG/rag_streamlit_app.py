import streamlit as st
import json
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="RAG – Cortex Search", page_icon="❄️", layout="wide")

# ── Active Snowpark session ───────────────────────────────────────────────────
session = get_active_session()

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; color:#1a1a2e; }
    .sub-header  { color:#666; font-size:0.95rem; margin-bottom:1.5rem; }
    .file-card {
        background:#f8f9fa; border:1px solid #e0e0e0;
        border-radius:10px; padding:1rem 1.4rem;
        margin-bottom:0.8rem; border-left:5px solid #29b5e8;
    }
    .file-title { font-size:1rem; font-weight:600; color:#1a1a2e; }
    .badge {
        display:inline-block; background:#e0f4fb; color:#0e7fa8;
        border-radius:6px; padding:2px 9px; font-size:0.78rem;
        font-weight:600; margin-right:5px;
    }
    .indexed { color:#16a34a; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">❄️ RAG — Snowflake Cortex Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload any file (PDF, TXT, CSV, DOCX, MD, JSON) → auto-index into Snowflake → ask questions</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM Model", [
        "mistral-large2", "snowflake-arctic", "llama3.1-70b",
        "llama3.1-8b", "mixtral-8x7b"
    ])
    chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 300, 150, step=50)
    st.markdown("---")
    st.caption("Supported formats: TXT, PDF, CSV, DOCX, MD, JSON, HTML")


# ══════════════════════════════════════════════════════════════════════════════
# Helper – extract text from uploaded file based on format
# ══════════════════════════════════════════════════════════════════════════════
def extract_text_from_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if file_name.endswith('.txt') or file_name.endswith('.md'):
        return content.decode('utf-8', errors='ignore')

    elif file_name.endswith('.csv'):
        text = content.decode('utf-8', errors='ignore')
        lines = text.strip().split('\n')
        # Convert CSV rows into readable text
        return '\n'.join(lines)

    elif file_name.endswith('.json'):
        text = content.decode('utf-8', errors='ignore')
        try:
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            return text

    elif file_name.endswith('.html') or file_name.endswith('.htm'):
        text = content.decode('utf-8', errors='ignore')
        # Basic HTML tag removal
        import re
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    elif file_name.endswith('.pdf'):
        # Upload PDF to stage and use AI_PARSE_DOCUMENT
        return extract_pdf_via_stage(uploaded_file.name, content)

    elif file_name.endswith('.docx'):
        # Extract text from DOCX (ZIP with XML)
        import zipfile
        import io
        import re
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                xml_content = z.read('word/document.xml').decode('utf-8')
                text = re.sub(r'<[^>]+>', ' ', xml_content)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception:
            return content.decode('utf-8', errors='ignore')

    else:
        # Fallback: try to read as plain text
        return content.decode('utf-8', errors='ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Helper – extract PDF via Snowflake stage + AI_PARSE_DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════
def extract_pdf_via_stage(file_name: str, content: bytes) -> str:
    import os

    # Use SSE-encrypted stage (AI_PARSE_DOCUMENT doesn't support client-side encryption)
    session.sql("""
        CREATE STAGE IF NOT EXISTS TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE
        ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    """).collect()

    # Write to temp file and PUT to stage
    safe_name = file_name.replace(" ", "_").replace("'", "").replace(",", "")
    tmp_path = f"/tmp/{safe_name}"
    with open(tmp_path, 'wb') as f:
        f.write(content)

    session.file.put(tmp_path, "@TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE", auto_compress=False, overwrite=True)
    os.remove(tmp_path)

    # Use SNOWFLAKE.CORTEX.PARSE_DOCUMENT with stage reference (not TO_FILE)
    try:
        result = session.sql(f"""
            SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
                @TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE,
                '{safe_name}',
                {{'mode': 'OCR'}}
            ):content::STRING AS doc_text
        """).collect()
        if result and result[0]["DOC_TEXT"]:
            text = result[0]["DOC_TEXT"]
            if len(text.strip()) > 50:
                return text
    except Exception as e:
        st.warning(f"PARSE_DOCUMENT (OCR) failed: {e}. Trying LAYOUT mode...")

    # Try LAYOUT mode as fallback
    try:
        result = session.sql(f"""
            SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
                @TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE,
                '{safe_name}',
                {{'mode': 'LAYOUT'}}
            ):content::STRING AS doc_text
        """).collect()
        if result and result[0]["DOC_TEXT"]:
            text = result[0]["DOC_TEXT"]
            if len(text.strip()) > 50:
                return text
    except Exception as e:
        st.warning(f"PARSE_DOCUMENT (LAYOUT) failed: {e}")

    st.error(f"Could not extract readable text from {file_name}. The PDF may be image-only or encrypted.")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Helper – chunk text
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Helper – insert chunks
# ══════════════════════════════════════════════════════════════════════════════
def insert_chunks(file_name: str, chunks: list):
    from snowflake.snowpark.functions import lit, current_timestamp
    from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
    # Use batch insert with createDataFrame to avoid SQL injection issues
    rows = [(file_name, i, chunk) for i, chunk in enumerate(chunks)]
    schema = StructType([
        StructField("FILE_NAME", StringType()),
        StructField("CHUNK_INDEX", IntegerType()),
        StructField("CHUNK", StringType())
    ])
    df = session.create_dataframe(rows, schema=schema)
    # Add UPLOADED_AT column to match the 4-column table schema
    df = df.with_column("UPLOADED_AT", current_timestamp())
    df.write.mode("append").save_as_table("TXT_RAG_DB.DATA.TXT_CHUNKS")


# ══════════════════════════════════════════════════════════════════════════════
# Helper – delete chunks for a file (re-indexing)
# ══════════════════════════════════════════════════════════════════════════════
def delete_chunks(file_name: str):
    safe_name = file_name.replace("'", "''")
    session.sql(f"DELETE FROM TXT_RAG_DB.DATA.TXT_CHUNKS WHERE FILE_NAME = '{safe_name}'").collect()


# ══════════════════════════════════════════════════════════════════════════════
# Helper – Cortex Search
# ══════════════════════════════════════════════════════════════════════════════
def cortex_search(question: str, limit: int = 5):
    from snowflake.snowpark.functions import call_function, lit, parse_json
    payload_dict = {
        "query": question,
        "columns": ["CHUNK", "FILE_NAME"],
        "limit": limit
    }
    payload_str = json.dumps(payload_dict)
    # Use Snowpark function call with lit() to safely pass the payload
    df = session.create_dataframe([{"dummy": 1}])
    result_df = df.select(
        call_function("SNOWFLAKE.CORTEX.SEARCH_PREVIEW",
                      lit("TXT_RAG_DB.DATA.TXT_SEARCH_SERVICE"),
                      lit(payload_str)).alias("results")
    )
    rows = result_df.collect()
    if rows and rows[0]["RESULTS"]:
        raw = rows[0]["RESULTS"]
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return data.get("results", data.get("RESULTS", []))
        elif isinstance(data, list):
            return data
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Helper – Cortex Complete
# ══════════════════════════════════════════════════════════════════════════════
def cortex_complete(model_name: str, question: str, context_chunks: list) -> str:
    from snowflake.snowpark.functions import call_function, lit
    context = "\n\n---\n\n".join(
        f"[From: {c.get('FILE_NAME', 'unknown')}]\n{c.get('CHUNK', '')}"
        for c in context_chunks
    )
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
        "If the answer is not in the context, say \"I couldn't find that in the files.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    # Use Snowpark call_function with lit() to safely pass the prompt
    df = session.create_dataframe([{"dummy": 1}])
    result_df = df.select(
        call_function("SNOWFLAKE.CORTEX.COMPLETE",
                      lit(model_name),
                      lit(prompt)).alias("answer")
    )
    rows = result_df.collect()
    return rows[0]["ANSWER"] if rows else "No response from LLM."


# ══════════════════════════════════════════════════════════════════════════════
# File Upload Section
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Upload Files")

uploaded_files = st.file_uploader(
    "Drop files here to upload and index",
    type=["txt", "pdf", "csv", "docx", "md", "json", "html", "htm"],
    accept_multiple_files=True,
    help="Supported: TXT, PDF, CSV, DOCX, MD, JSON, HTML"
)

if uploaded_files:
    if st.button("Index All Uploaded Files", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            status_text.text(f"Processing {file_name}...")

            # Extract text based on file type
            text = extract_text_from_file(uploaded_file)

            if not text.strip():
                st.warning(f"No text extracted from {file_name}. Skipping.")
                continue

            # Delete old chunks if re-indexing
            if file_name in st.session_state.indexed_files:
                delete_chunks(file_name)

            # Chunk and insert
            chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
            insert_chunks(file_name, chunks)
            st.session_state.indexed_files.add(file_name)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()
        st.success(f"Indexed {len(uploaded_files)} file(s)!")
        st.rerun()

# ── Show indexed files ────────────────────────────────────────────────────────
if st.session_state.indexed_files:
    st.markdown("#### Indexed Files")
    for fname in sorted(st.session_state.indexed_files):
        ext = fname.rsplit('.', 1)[-1].upper() if '.' in fname else 'FILE'
        st.markdown(f"""
        <div class="file-card">
            <div class="file-title">{fname}</div>
            <div style="margin-top:0.3rem;">
                <span class="badge">{ext}</span>
                <span class="indexed">Indexed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Option to check chunk count
    if st.button("Show chunk count"):
        result = session.sql("SELECT COUNT(*) AS cnt FROM TXT_RAG_DB.DATA.TXT_CHUNKS").collect()
        st.info(f"Total chunks in database: {result[0]['CNT']}")

    # Clear all uploaded documents and chunks
    if st.button("Clear All Documents & Chunks", type="secondary", use_container_width=True):
        session.sql("TRUNCATE TABLE TXT_RAG_DB.DATA.TXT_CHUNKS").collect()
        st.session_state.indexed_files = set()
        st.session_state.chat_messages = []
        st.success("All documents and chunks cleared!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Chat Window
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### Ask your files")

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_messages:
        st.markdown("""
        <div style='text-align:center;color:#bbb;padding:4rem 0;'>
            <div style='font-size:2rem;'>💬</div>
            <div style='font-size:0.9rem;margin-top:0.4rem;'>Upload and index files above, then ask anything about them...</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-end;margin-bottom:0.5rem;'>
                    <div style='background:#29b5e8;color:white;padding:0.55rem 1rem;
                                border-radius:16px 16px 4px 16px;max-width:72%;font-size:0.88rem;'>
                        {msg["content"]}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-start;margin-bottom:0.5rem;'>
                    <div style='background:#f1f1f1;color:#333;padding:0.55rem 1rem;
                                border-radius:16px 16px 16px 4px;max-width:72%;font-size:0.88rem;'>
                        {msg["content"]}
                    </div>
                </div>""", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([9, 1])
with col_input:
    user_input = st.text_input("Ask a question...", key="chat_input", label_visibility="collapsed")
with col_btn:
    send_clicked = st.button("Send", use_container_width=True)

if send_clicked and user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    if not st.session_state.indexed_files:
        answer = "Please upload and index at least one file first."
    else:
        with st.spinner("Searching & generating answer..."):
            try:
                chunks = cortex_search(user_input)
                if chunks:
                    answer = cortex_complete(model, user_input, chunks)
                else:
                    answer = "I couldn't find relevant content in the indexed files."
            except Exception as e:
                answer = f"Error: {e}"

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.rerun()
