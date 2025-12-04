##############################################################
# LLM Token & Cost Analyzer  (OpenAI GPT models only)
#
# What this app does
# ------------------
# 1. Lets a user:
#    - Paste text in a textbox  OR
#    - Upload a file (txt, pdf, csv, xlsx)
#
# 2. Converts that text into tokens using tiktoken.
#
# 3. Lets the user choose one of several OpenAI GPT models
#    that all use the same tokenizer family (o200k_base).
#
# 4. Shows:
#    - Total token count
#    - Approximate input cost per model (USD per 1M tokens)
#    - Colored token visualization + token IDs
#
# NOTE:
#   This app does *not* call the OpenAI API.
#   It only uses the same tokenizer family as the models,
#   so you can estimate length + cost before actually calling any real API in a production system.
##############################################################

import streamlit as st
import tiktoken
from pypdf import PdfReader
import pandas as pd


MODEL_OPTIONS = [
    "GPT-4.1",
    "GPT-4.1 mini",
    "GPT-4o",
    "GPT-4o mini",
]


ENCODING_NAME = "o200k_base"

# OFFICIAL *input* prices per 1M tokens from OpenAI pricing page.
PRICES_PER_MILLION_INPUT = {
    "GPT-4.1": 2.00,       # $2.00 / 1M input tokens
    "GPT-4.1 mini": 0.40,  # $0.40 / 1M input tokens
    "GPT-4o": 2.50,        # $2.50 / 1M input tokens
    "GPT-4o mini": 0.15,   # $0.15 / 1M input tokens
}



def read_file_to_text(uploaded_file) -> str:
    """Convert various file types to a single big text string."""
    if uploaded_file is None:
        return ""

    # Plain text
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # PDF
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        pages_text = []
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
        return "\n\n".join(pages_text)

    # CSV
    if uploaded_file.type in ["text/csv", "application/vnd.ms-excel"]:
        df = pd.read_csv(uploaded_file)
        text_cols = df.select_dtypes(include=["object"]).columns
        text_data = df[text_cols].astype(str).fillna("").values.flatten()
        return "\n".join(text_data)

    # XLSX (Excel)
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        text_cols = df.select_dtypes(include=["object"]).columns
        text_data = df[text_cols].astype(str).fillna("").values.flatten()
        return "\n".join(text_data)

    # Fallback: unknown type
    return ""



def build_colored_tokens_html(tokens, token_ids):
    """
    Turn tokens + token IDs into a scrollable colored HTML block.

    - Each token is a span with a background color.
    - Hover tooltip shows the token ID.
    - We only visualize the *first N* tokens for performance.
    """
    colors = [
        "#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9",
        "#C5CAE9", "#BBDEFB", "#B3E5FC", "#B2EBF2",
        "#B2DFDB", "#C8E6C9",
    ]

    html = """
    <div style="
        max-height:400px;
        overflow-y:auto;
        overflow-x:auto;
        padding:10px;
        background:#000000;
        border-radius:8px;
        border:1px solid #ddd;
        font-family:monospace;
        white-space:pre-wrap;
    ">
    """

    for i, tok in enumerate(tokens):
        color = colors[i % len(colors)]
        tid = token_ids[i]
        html += (
            f"<span title='Token ID: {tid}' "
            f"style='background:{color}; color:black; padding:2px 4px; margin:2px; "
            f"border-radius:4px; display:inline-block; cursor:pointer;'>"
            f"{tok}"
            f"</span>"
        )

    html += "</div>"
    return html



st.set_page_config(page_title="LLM Token & Cost Analyzer", layout="wide")

st.title("LLM Token & Cost Analyzer (GPT-only)")
st.caption(
    "Hey, I'm Sandhya Konjarla. This LLM Token & Cost Analyzer was built to explore how GPT models tokenize text, "
    "estimate input costs, and visualize RAG-style chunking — all using Python, Streamlit, and tiktoken."
)


left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("Input Text or File")

    uploaded_file = st.file_uploader(
        "Upload a .txt / .pdf / .csv / .xlsx file (optional)",
        type=["txt", "pdf", "csv", "xlsx"],
    )

    # If a file is uploaded, we disable manual text typing so they don't clash.
    disable_textbox = uploaded_file is not None

    user_text = st.text_area(
        "Type / Paste text here:",
        value="Hey!😊 type or paste any text here to see how GPT models tokenize it.",
        height=200,
        disabled=disable_textbox,
    )

    if uploaded_file is not None:
        file_text = read_file_to_text(uploaded_file)
        if file_text.strip():
            user_text = file_text
            st.info("File detected → using file contents instead of textbox.")
        else:
            st.warning("Uploaded file seems empty or unreadable. Falling back to textbox.")

with right_col:
    st.subheader("Choose GPT model")

    with st.container():
        st.markdown("**Model (all use the o200k_base tokenizer family):**")

        chosen_model = st.radio(
            "Select a model:",
            MODEL_OPTIONS,
            horizontal=False,  
            label_visibility="collapsed"
        )

    st.markdown(
        "- **GPT-4.1** → flagship, long context, higher quality\n"
        "- **GPT-4.1 mini** → cheaper, good for high-volume usage\n"
        "- **GPT-4o** → omni model (text+vision), very capable\n"
        "- **GPT-4o mini** → ultra-cheap, good for bulk CRM / logs"
    )

    analyze_clicked = st.button("Analyze tokens & cost", type="primary")



if analyze_clicked:
    if not user_text.strip():
        st.error("Please paste some text or upload a file first.")
        st.stop()

    
    try:
        encoding = tiktoken.get_encoding(ENCODING_NAME)
        encoding_used = ENCODING_NAME
    except KeyError:
        # Safety fallback: older tiktoken versions may not know o200k_base.
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding_used = "cl100k_base (fallback)"

   
    token_ids = encoding.encode(user_text)
    total_tokens = len(token_ids)

 
    model_key = chosen_model  
    price_per_million = PRICES_PER_MILLION_INPUT.get(model_key, 0.0)

    # Cost = (tokens / 1M) * price_per_million
    estimated_cost = (total_tokens / 1_000_000) * price_per_million

 
    suggested_chunk_size = 600
    approx_chunks = max(1, round(total_tokens / suggested_chunk_size))

  
    st.markdown("---")
    st.subheader("Token & Cost Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total tokens", total_tokens)
    with c2:
        st.metric("Encoding used", encoding_used)
    with c3:
        st.metric(
            "Est. input cost (USD)",
            f"${estimated_cost:.6f}",
            help=(
                f"Price used: ${price_per_million:.2f} per 1M input tokens "
                f"for {chosen_model}"
            ),
        )

    st.info(
        "All listed GPT models currently share the same tokenizer family "
        "(o200k_base), so **token counts are identical** for a given text. "
        "What changes between models is **price and capability**, "
        "not how they break text into tokens."
    )

    st.subheader("RAG-style chunk suggestion")
    st.write(
        f"- Suggested chunk size: **{suggested_chunk_size} tokens**\n"
        f"- Approximate number of chunks: **{approx_chunks}**"
    )

  
    st.subheader("Token visualization (first 2,000 tokens)")

    max_visual_tokens = 2000
    visual_token_ids = token_ids[:max_visual_tokens]

    # Decode each token ID back to text; this is what Karpathy's tool shows.
    tokens_as_strings = [encoding.decode([tid]) for tid in visual_token_ids]

    html = build_colored_tokens_html(tokens_as_strings, visual_token_ids)
    st.markdown(html, unsafe_allow_html=True)

    st.caption(
        f"Showing first {len(visual_token_ids)} tokens out of {total_tokens}. "
        "Scroll horizontally and vertically inside the box."
    )


    st.subheader("Raw token IDs (first 5,000)")
    st.code(", ".join(str(tid) for tid in token_ids[:5000]), language="text")

