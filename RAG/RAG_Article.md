# Building a RAG Chatbot with Snowflake Cortex Search

**A Complete Guide to Retrieval-Augmented Generation on Snowflake**

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Architecture Overview](#architecture-overview)
3. [How It Works — Step by Step](#how-it-works)
4. [Project Setup](#project-setup)
5. [SQL Setup Script](#sql-setup-script)
6. [Streamlit App Code Walkthrough](#streamlit-app-code-walkthrough)
7. [Key Snowflake Services Used](#key-snowflake-services-used)
8. [Lessons Learned](#lessons-learned)
9. [How to Run This Project](#how-to-run-this-project)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by providing them with relevant context from your own data before generating a response. Instead of relying solely on the LLM's training data, RAG retrieves the most relevant documents from a knowledge base and feeds them as context to the model.

```
Traditional LLM:
  User Question ──► LLM ──► Answer (from training data only, may hallucinate)

RAG-Enhanced LLM:
  User Question ──► Search Knowledge Base ──► Relevant Context + Question ──► LLM ──► Grounded Answer
```

**Why RAG?**
- Answers are grounded in YOUR data (no hallucination)
- No need to fine-tune or retrain the LLM
- Data stays private in Snowflake (never leaves your account)
- Always up-to-date — just add new documents

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SNOWFLAKE ACCOUNT                                     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    STREAMLIT APP (Frontend)                            │  │
│  │                                                                       │  │
│  │   ┌─────────────┐   ┌──────────────┐   ┌────────────────────────┐  │  │
│  │   │  File Upload │   │  Chat Window │   │  Settings (Model,      │  │  │
│  │   │  (PDF, TXT,  │   │  (Q&A with   │   │  Chunk Size, Overlap)  │  │  │
│  │   │  CSV, DOCX)  │   │   documents) │   │                        │  │  │
│  │   └──────┬───────┘   └──────┬───────┘   └────────────────────────┘  │  │
│  └──────────┼──────────────────┼────────────────────────────────────────┘  │
│             │                  │                                             │
│             ▼                  ▼                                             │
│  ┌─────────────────┐   ┌─────────────────────────────────────────────┐     │
│  │  INGESTION       │   │  RETRIEVAL & GENERATION                     │     │
│  │  PIPELINE        │   │                                             │     │
│  │                  │   │  1. User asks a question                    │     │
│  │  1. Upload file  │   │  2. Cortex Search finds relevant chunks    │     │
│  │  2. Parse (OCR)  │   │  3. Top-K chunks become context            │     │
│  │  3. Chunk text   │   │  4. Context + Question → LLM               │     │
│  │  4. Store chunks │   │  5. LLM generates grounded answer          │     │
│  │                  │   │                                             │     │
│  └────────┬─────────┘   └───────────┬─────────────────────────────────┘     │
│           │                         │                                        │
│           ▼                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     SNOWFLAKE DATA LAYER                              │  │
│  │                                                                       │  │
│  │   ┌─────────────────┐    ┌────────────────────────────────────────┐ │  │
│  │   │  TXT_CHUNKS      │    │  CORTEX SEARCH SERVICE                 │ │  │
│  │   │  (Table)         │    │  (Managed Hybrid Index)                │ │  │
│  │   │                  │    │                                        │ │  │
│  │   │  FILE_NAME       │───►│  • Semantic embeddings (auto)         │ │  │
│  │   │  CHUNK_INDEX     │    │  • Keyword index (BM25)               │ │  │
│  │   │  CHUNK           │    │  • Auto-refresh every 1 minute        │ │  │
│  │   │  UPLOADED_AT     │    │  • Hybrid ranking                     │ │  │
│  │   └─────────────────┘    └────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │   ┌─────────────────┐    ┌────────────────────────────────────────┐ │  │
│  │   │  STAGE (SSE)     │    │  CORTEX AI FUNCTIONS                   │ │  │
│  │   │  (PDF Storage)   │    │                                        │ │  │
│  │   │                  │    │  • PARSE_DOCUMENT (PDF → Text)         │ │  │
│  │   │  uploaded PDFs    │    │  • COMPLETE (LLM Generation)           │ │  │
│  │   │  for OCR parsing  │    │  • SEARCH_PREVIEW (Query Index)        │ │  │
│  │   └─────────────────┘    └────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works — Step by Step

### Phase 1: Document Ingestion

```
                    ┌───────────────┐
                    │  User Uploads │
                    │  a PDF File   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │  Detect File Type │
                    │  (.pdf? .txt?     │
                    │   .csv? .docx?)   │
                    └───────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
     ┌────────────┐  ┌──────────┐  ┌──────────┐
     │  PDF:      │  │  TXT/MD: │  │  DOCX:   │
     │  Upload to │  │  Read as │  │  Extract  │
     │  Stage →   │  │  UTF-8   │  │  from XML │
     │  OCR Parse │  │          │  │  in ZIP   │
     └─────┬──────┘  └────┬─────┘  └────┬─────┘
           │               │              │
           └───────────────┼──────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  CHUNKING       │
                  │                 │
                  │  Split text into│
                  │  overlapping    │
                  │  chunks of      │
                  │  ~1000 chars    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  INSERT INTO    │
                  │  TXT_CHUNKS     │
                  │  table          │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────────────┐
                  │  Cortex Search Service  │
                  │  AUTO-INDEXES within    │
                  │  1 minute (embeddings   │
                  │  + keyword index)       │
                  └─────────────────────────┘
```

### Phase 2: Question Answering (RAG)

```
    ┌──────────────────┐
    │  User asks:      │
    │  "What is the    │
    │   program about?"│
    └────────┬─────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  CORTEX SEARCH_PREVIEW  │
    │                         │
    │  Hybrid search:         │
    │  • Semantic similarity  │
    │  • Keyword matching     │
    │  Returns Top-5 chunks   │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────┐
    │  BUILD PROMPT                                │
    │                                             │
    │  "You are a helpful assistant.              │
    │   Use ONLY the context below...            │
    │                                             │
    │   Context:                                  │
    │   [Chunk 1 from file_a.pdf]                │
    │   [Chunk 2 from file_a.pdf]                │
    │   [Chunk 3 from file_b.txt]                │
    │   ...                                       │
    │                                             │
    │   Question: What is the program about?     │
    │   Answer:"                                  │
    └────────┬────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  CORTEX.COMPLETE        │
    │  (mistral-large2)       │
    │                         │
    │  LLM generates answer   │
    │  grounded in the chunks │
    └────────┬────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  Display answer in      │
    │  chat interface         │
    └─────────────────────────┘
```

---

## Project Setup

### Project Structure

```
RAG/
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── rag_streamlit_app.py      # Main Streamlit app (upload, chat, RAG logic)
├── rag_snoflake.sql          # SQL setup script (run once)
├── snowflake.yml             # Snowflake app deployment configuration
├── pyproject.toml            # Python dependencies
└── Veltrix_Dynamics_Company_Report.txt  # Sample test document
```

### Prerequisites

| Requirement | Details |
|---|---|
| Snowflake Account | Enterprise or higher (Cortex AI functions required) |
| Role | ACCOUNTADMIN or role with CREATE DATABASE, CREATE STAGE, CREATE CORTEX SEARCH SERVICE privileges |
| Warehouse | COMPUTE_WH (or any active warehouse) |

---

## SQL Setup Script

Run this **once** in a Snowflake SQL worksheet before launching the app:

```sql
-- 1. Create database & schema
CREATE DATABASE IF NOT EXISTS TXT_RAG_DB;
CREATE SCHEMA IF NOT EXISTS TXT_RAG_DB.DATA;

USE DATABASE TXT_RAG_DB;
USE SCHEMA DATA;

-- 2. Table to store text chunks
CREATE OR REPLACE TABLE TXT_RAG_DB.DATA.TXT_CHUNKS (
    FILE_NAME   VARCHAR,        -- original filename
    CHUNK_INDEX INT,            -- chunk number within the file
    CHUNK       TEXT,           -- actual text chunk (used for search)
    UPLOADED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- 3. Cortex Search Service (auto-indexes chunks for hybrid search)
CREATE OR REPLACE CORTEX SEARCH SERVICE TXT_RAG_DB.DATA.TXT_SEARCH_SERVICE
    ON CHUNK                    -- column to index for search
    ATTRIBUTES FILE_NAME        -- filterable metadata column
    WAREHOUSE = COMPUTE_WH
    TARGET_LAG = '1 minute'     -- re-index frequency
    AS (
        SELECT CHUNK, FILE_NAME
        FROM TXT_RAG_DB.DATA.TXT_CHUNKS
    );

-- 4. SSE-encrypted stage for PDF uploads (required for PARSE_DOCUMENT)
CREATE STAGE IF NOT EXISTS TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- 5. Verify
SHOW CORTEX SEARCH SERVICES;
SELECT COUNT(*) AS total_chunks FROM TXT_RAG_DB.DATA.TXT_CHUNKS;
```

### Why SSE Encryption for the Stage?

`SNOWFLAKE.CORTEX.PARSE_DOCUMENT` does **not** support client-side encrypted stages (the default). You must create a stage with `ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')` for PDF/image parsing to work.

---

## Streamlit App Code Walkthrough

### Core Components

#### 1. File Parser (Multi-Format Support)

```python
def extract_text_from_file(uploaded_file) -> str:
    """Routes to the correct parser based on file extension."""
    file_name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if file_name.endswith('.pdf'):
        return extract_pdf_via_stage(uploaded_file.name, content)
    elif file_name.endswith('.txt') or file_name.endswith('.md'):
        return content.decode('utf-8', errors='ignore')
    elif file_name.endswith('.docx'):
        # Extract from ZIP > word/document.xml
        ...
    elif file_name.endswith('.csv'):
        return content.decode('utf-8', errors='ignore')
    ...
```

#### 2. PDF Parsing via Cortex

```python
def extract_pdf_via_stage(file_name: str, content: bytes) -> str:
    """Uploads PDF to SSE stage, then uses PARSE_DOCUMENT for OCR extraction."""
    # 1. Upload to SSE-encrypted stage
    session.file.put(tmp_path, "@TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE", ...)

    # 2. Parse with OCR
    result = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
            @TXT_RAG_DB.DATA.RAG_UPLOAD_STAGE_SSE,
            '{safe_name}',
            {{'mode': 'OCR'}}
        ):content::STRING AS doc_text
    """).collect()
    return result[0]["DOC_TEXT"]
```

#### 3. Text Chunking (Overlapping Windows)

```python
def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    """Split text into overlapping chunks for better retrieval."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap  # overlap ensures context continuity
    return chunks
```

```
  Document Text:
  ┌────────────────────────────────────────────────────────────────┐
  │  The Post Graduate Program in Generative AI and ML, a          │
  │  collaborative offering from Edureka and Illinois Tech...      │
  └────────────────────────────────────────────────────────────────┘

  After chunking (size=1000, overlap=150):

  Chunk 0: ┌──────────────────────────────────┐
            │  "The Post Graduate Program..."  │  (chars 0-999)
            └──────────────────────────────────┘
                                  ▲ overlap
  Chunk 1:            ┌──────────────────────────────────┐
                      │  "...offering from Edureka..."   │  (chars 850-1849)
                      └──────────────────────────────────┘
                                            ▲ overlap
  Chunk 2:                        ┌──────────────────────────────────┐
                                  │  "...Illinois Tech stands..."    │  (chars 1700-2699)
                                  └──────────────────────────────────┘
```

#### 4. Cortex Search (Hybrid Retrieval)

```python
def cortex_search(question: str, limit: int = 5):
    """Search indexed chunks using Cortex Search (semantic + keyword hybrid)."""
    from snowflake.snowpark.functions import call_function, lit
    payload_str = json.dumps({"query": question, "columns": ["CHUNK", "FILE_NAME"], "limit": limit})

    df = session.create_dataframe([{"dummy": 1}])
    result_df = df.select(
        call_function("SNOWFLAKE.CORTEX.SEARCH_PREVIEW",
                      lit("TXT_RAG_DB.DATA.TXT_SEARCH_SERVICE"),
                      lit(payload_str)).alias("results")
    )
    return json.loads(result_df.collect()[0]["RESULTS"])["results"]
```

#### 5. LLM Answer Generation

```python
def cortex_complete(model_name: str, question: str, context_chunks: list) -> str:
    """Generate answer using LLM with retrieved context."""
    context = "\n\n---\n\n".join(
        f"[From: {c['FILE_NAME']}]\n{c['CHUNK']}" for c in context_chunks
    )
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
    Context:
    {context}

    Question: {question}
    Answer:"""

    df = session.create_dataframe([{"dummy": 1}])
    result_df = df.select(
        call_function("SNOWFLAKE.CORTEX.COMPLETE", lit(model_name), lit(prompt)).alias("answer")
    )
    return result_df.collect()[0]["ANSWER"]
```

---

## Key Snowflake Services Used

| Service | Purpose | Why We Used It |
|---|---|---|
| **Cortex Search Service** | Hybrid semantic + keyword search index | Auto-manages embeddings, no manual vector DB needed |
| **SNOWFLAKE.CORTEX.PARSE_DOCUMENT** | PDF/image OCR extraction | Extracts text from PDFs server-side in Snowflake |
| **SNOWFLAKE.CORTEX.COMPLETE** | LLM text generation | Generates answers using retrieved context |
| **SNOWFLAKE.CORTEX.SEARCH_PREVIEW** | Query the search service | Returns ranked results from the search index |
| **Internal Stage (SSE)** | File storage for PDFs | Required for PARSE_DOCUMENT compatibility |
| **Streamlit in Snowflake** | Web UI | No infrastructure needed, runs natively |

---

## Lessons Learned

### 1. Stage Encryption Matters
`PARSE_DOCUMENT` does NOT work with client-side encrypted stages (the default). Always create stages with `ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')` for document parsing.

### 2. Use Snowpark `call_function` + `lit()` for Safe SQL
Never use f-string interpolation or `$$` dollar-quoting when passing user content to Cortex functions. Use Snowpark's `call_function` with `lit()` to avoid SQL injection and escaping issues.

### 3. Chunk Overlap is Critical
Without overlap, search results may miss context that spans chunk boundaries. A 15% overlap (150 chars on 1000-char chunks) ensures continuity.

### 4. Cortex Search Service is Fully Managed
- You don't see the embeddings (they're internal)
- It auto-refreshes based on `TARGET_LAG`
- It combines semantic AND keyword search (hybrid ranking)
- No need to manage a separate vector database

### 5. The `ATTRIBUTES` Clause Enables Filtering
Adding `ATTRIBUTES FILE_NAME` lets you filter search results by source file without affecting ranking — useful for scoping answers to specific documents.

---

## How to Run This Project

### Step 1: Run the SQL Setup

Open a Snowflake SQL Worksheet and execute `rag_snoflake.sql` (all statements).

### Step 2: Deploy the Streamlit App

In Snowsight:
1. Navigate to **Projects > Streamlit**
2. Click **+ Streamlit App**
3. Upload the `RAG/` folder contents
4. Set `rag_streamlit_app.py` as the main file

Or use the Snowflake CLI:
```bash
snow streamlit deploy --project RAG/
```

### Step 3: Use the App

1. Open the Streamlit app in Snowsight
2. Upload PDF, TXT, CSV, or DOCX files
3. Click "Index All Uploaded Files"
4. Wait ~1 minute for Cortex Search to index
5. Ask questions in the chat!

---

## Cost Estimate

| Operation | Cost |
|---|---|
| PARSE_DOCUMENT (PDF OCR) | ~970 tokens/page |
| CORTEX.COMPLETE (per question) | Depends on model + context size |
| Cortex Search Service | Serverless compute (auto-scales) |
| Storage (chunks table) | Standard Snowflake storage rates |

For a typical 10-page PDF with 5 questions/day, expect < 0.1 credits/day.

---

## Extending This Project

Ideas for next steps:
- Add **file-specific filtering** (ask questions about only one document)
- Add **citation links** (show which chunk the answer came from)
- Support **image-based Q&A** with AI_EXTRACT on images
- Add **conversation memory** (multi-turn chat with thread context)
- Build an **evaluation pipeline** to measure answer accuracy

---

*Built with Snowflake Cortex Search, Cortex AI Functions, and Streamlit in Snowflake.*
