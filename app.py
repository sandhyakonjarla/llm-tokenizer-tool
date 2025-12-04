import math
from io import StringIO

import streamlit as st
import tiktoken
from pypdf import PdfReader
import pandas as pd


# ---------- CONFIG: MODELS & PRICES (you can tweak later) ----------

MODELS = {
    "GPT-4 style (cl100k_base)": {
        "tokenizer_family": "cl100k_base",   # for tiktoken
        "context": 128_000,                  # max tokens
        # Example prices — change later from docs if you want
        "input_per_1k": 0.01,                # $ per 1K input tokens
        "output_per_1k": 0.03,               # $ per 1K output tokens
    },
    "GPT-3.5 style (cl100k_base)": {
        "tokenizer_family": "cl100k_base",
        "context": 16_000,
        "input_per_1k": 0.0015,
        "output_per_1k": 0.002,
    },
    "Gemini-style (approx)": {
        "tokenizer_family": "cl100k_base",
        "context": 1_000_000,   # very large context, just an example
        "input_per_1k": 0.00125,
        "output_per_1k": 0.00375,
    },
    "Claude-style (approx)": {
        "tokenizer_family": "cl100k_base",
        "context": 200_000,
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
}


# ---------- TOKENIZER HELPERS ----------

@st.cache_resource
def get_encoding(tokenizer_family: str):
    """
    Returns a tiktoken encoding.
    cl100k_base works for GPT-4, GPT-3.5 and is a
    reasonable approximation for many modern models.
    """
    return tiktoken.get_encoding(tokenizer_family)


def count_tokens(text: str, encoding) -> int:
    if not text:
        return 0
    return len(encoding.encode(text))


def chunk_text_by_tokens(text: str, encoding, chunk_size: int):
    """
    Splits text into chunks of `chunk_size` tokens using the encoding.
    """
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(
            {
                "chunk_index": len(chunks) + 1,
                "tokens": len(chunk_tokens),
                "text": chunk_text,
            }
        )
    return chunks, len(tokens)


def estimate_cost(num_tokens: int, input_price_per_1k: float, output_price_per_1k: float):
    """
    Very rough cost estimation:
    - assume output tokens ≈ 30% of input, max 1K.
    """
    if num_tokens == 0:
        return 0.0, 0.0, 0.0
    input_cost = (num_tokens / 1000) * input_price_per_1k
    est_output_tokens = min(int(num_tokens * 0.3), 1000)
    output_cost = (est_output_tokens / 1000) * output_price_per_1k
    total = input_cost + output_cost
    return input_cost, output_cost, total


# ---------- FILE READING (txt / pdf / csv / excel) ----------

def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    # 1) Plain text
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        return stringio.read()

    # 2) PDF
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        pages_text = []
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
        return "\n\n".join(pages_text)

    # 3) CSV
    if uploaded_file.type in ["text/csv", "application/vnd.ms-excel"]:
        df = pd.read_csv(uploaded_file)
        # keep only text-like columns
        text_cols = df.select_dtypes(include=["object"]).columns
        text_data = df[text_cols].astype(str).fillna("").values.flatten()
        return "\n".join(text_data)

    # 4) Excel (.xlsx)
    if uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        df = pd.read_excel(uploaded_file)
        text_cols = df.select_dtypes(include=["object"]).columns
        text_data = df[text_cols].astype(str).fillna("").values.flatten()
        return "\n".join(text_data)

    # 5) Fallback: try decode as text
    try:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="LLM Token & Cost Analyzer", layout="wide")

st.title("🧮 LLM Token & Cost Analyzer")
st.caption(
    "Paste text or upload a file to see token counts, suggested RAG chunks, "
    "and approximate cost for different LLM families (GPT, Gemini, Claude)."
)

# Sidebar config
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox("Model family (for tokenizer & cost estimate)", list(MODELS.keys()))
model_cfg = MODELS[model_name]
encoding = get_encoding(model_cfg["tokenizer_family"])

chunk_size = st.sidebar.slider("Chunk size (tokens)", min_value=128, max_value=2048, value=400, step=32)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 Prices above are **examples**. Update them in the code to match the latest from "
    "OpenAI / Google / Anthropic dashboards."
)

# Main input area
st.subheader("1️⃣ Input Text / File")

col1, col2 = st.columns(2)

with col1:
    text_input = st.text_area(
        "Paste text here:",
        height=250,
        placeholder="Paste any article, CRM export text, documentation, or notes here...",
    )

with col2:
    uploaded_file = st.file_uploader(
        "Or upload a .txt / .pdf / .csv / .xlsx file",
        type=["txt", "pdf", "csv", "xlsx"],
    )

file_text = read_uploaded_file(uploaded_file)

# Combine text from box and file if both given
if text_input and file_text:
    st.info("You provided BOTH manual text and a file. We'll analyze them **together**.")
    combined_text = text_input + "\n\n" + file_text
elif file_text:
    combined_text = file_text
else:
    combined_text = text_input

if not combined_text:
    st.warning("Please paste some text or upload a file to begin.")
    st.stop()

# Analysis button
if st.button("🔍 Analyze Tokens & Cost"):
    # TOKEN COUNTS
    total_tokens = count_tokens(combined_text, encoding)
    context_limit = model_cfg["context"]

    chunks, encoded_total_tokens = chunk_text_by_tokens(combined_text, encoding, chunk_size)
    num_chunks = len(chunks)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Total tokens", f"{total_tokens:,}")
    with colB:
        st.metric("Suggested chunk size", f"{chunk_size} tokens")
    with colC:
        st.metric("Number of chunks", f"{num_chunks}")

    # CONTEXT WINDOW WARNING
    if total_tokens > context_limit:
        st.error(
            f"Text is **larger than the context window** for {model_name} "
            f"({total_tokens:,} tokens > {context_limit:,} tokens). "
            "You MUST use RAG / chunking."
        )
    else:
        st.success(
            f"Text fits inside the context window for {model_name} "
            f"({total_tokens:,} ≤ {context_limit:,})."
        )

    # COST ESTIMATE
    input_cost, output_cost, total_cost = estimate_cost(
        total_tokens,
        model_cfg["input_per_1k"],
        model_cfg["output_per_1k"],
    )

    st.subheader("2️⃣ Approximate Cost (Example Numbers)")
    st.markdown(
        f"""
        - **Input cost**: `${input_cost:.6f}`  
        - **Estimated output cost** (rough): `${output_cost:.6f}`  
        - **Total per call (approx)**: `${total_cost:.6f}`  

        ⚠️ These are *illustrative* values.  
        Please update the prices in the code with the latest from provider dashboards.
        """
    )

    # RAG vs full-context comparison
    st.subheader("3️⃣ RAG vs Full-Context (Token Saving Logic)")

    # Full-context = send everything once (if it fits)
    if total_tokens <= context_limit:
        full_context_tokens = total_tokens
    else:
        # If it doesn't fit, conceptually you'd need multiple passes.
        full_context_tokens = total_tokens  # for comparison only

    # RAG: imagine per question you send only 3 chunks
    retrieved_chunks_per_query = 3
    avg_chunk_tokens = math.ceil(total_tokens / num_chunks) if num_chunks > 0 else 0
    rag_tokens_per_query = avg_chunk_tokens * retrieved_chunks_per_query

    saving_factor = (full_context_tokens / rag_tokens_per_query) if rag_tokens_per_query else 0.0

    st.markdown(
        f"""
        - **If you naively send full text** each time: ~**{full_context_tokens:,} tokens**  
        - **If you use RAG (e.g., 3 chunks/query)**: ~**{rag_tokens_per_query:,} tokens per query**  

        💡 **Token saving factor** (rough):  
        `full / rag ≈ {saving_factor:.1f}x`
        """
    )

    # CHUNK PREVIEW
    st.subheader("4️⃣ Sample Chunks (for RAG)")

    st.markdown(
        "These are the first few chunks your RAG system would store and retrieve from a vector DB."
    )

    for ch in chunks[:5]:
        st.markdown(f"**Chunk {ch['chunk_index']}** — `{ch['tokens']}` tokens")
        st.code(ch["text"][:800] + ("..." if len(ch["text"]) > 800 else ""), language="text")

    # TOKENS PER PARAGRAPH
    st.subheader("5️⃣ Tokens per Paragraph")

    paragraphs = [p for p in combined_text.split("\n\n") if p.strip()]
    data = []
    for i, p in enumerate(paragraphs, start=1):
        t = count_tokens(p, encoding)
        data.append((i, t))

    if data:
        st.markdown("Paragraph index → token count")
        st.table(
            {"Paragraph": [d[0] for d in data], "Tokens": [d[1] for d in data]}
        )
    else:
        st.write("No clear paragraph separation found.")
