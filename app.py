import streamlit as st
import time
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="📱", layout="wide")

# 2. LOAD MODEL FROM HUGGING FACE
# We use @st.cache_resource so it only downloads once
@st.cache_resource
def load_pipeline():
    model_id = "AishaniS/text_summarizer"  # Your specific HF repository
    
    try:
        # Load directly from the Hub
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

summarizer = load_pipeline()

# 3. PREPROCESSING FUNCTION (Corrected for your Date/Time format)
def clean_whatsapp_log(text):
    """
    Parses WhatsApp chat.
    Target format: "24/12/25, 09:38 - Name: Message"
    """
    # Regex Breakdown:
    # \d{1,2}/\d{1,2}/\d{2,4}  -> Date (e.g., 24/12/25)
    # ,\s                      -> Comma and space
    # \d{1,2}:\d{2}            -> Time (e.g., 09:38 or 20:43)
    # \s-\s                    -> " - " separator
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    
    clean_lines = []
    lines = text.split('\n')
    
    for line in lines:
        # Filter system messages
        if "<Media omitted>" in line or "Messages and calls are end-to-end encrypted" in line:
            continue
            
        # Remove timestamp
        cleaned_line = re.sub(pattern, '', line).strip()
        
        # Only add if text remains
        if cleaned_line:
            clean_lines.append(cleaned_line)
            
    return "\n".join(clean_lines)

# 4. CHUNKING FUNCTION (To handle long chats)
def chunk_text(text, max_chars=2000):
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) < max_chars:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 5. MAIN UI
st.title("📱 Real-Time WhatsApp Summarizer")
st.markdown(f"**Model:** `AishaniS/text_summarizer` | **Status:** {'✅ Loaded' if summarizer else '❌ Error'}")
st.markdown("Upload your exported `_chat.txt` file to analyze conversation.")

uploaded_file = st.file_uploader("Choose a file", type=['txt'])

if uploaded_file and summarizer:
    raw_text = uploaded_file.getvalue().decode("utf-8")
    
    # Preprocess
    clean_text = clean_whatsapp_log(raw_text)
    
    # Layout: Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📜 Processed Chat")
        st.text_area("Cleaned Input", clean_text, height=400)
    
    with col2:
        st.subheader("🤖 AI Summary")
        if st.button("Generate Summary"):
            if not clean_text:
                st.warning("Chat is empty after cleaning. Check the file format.")
            else:
                with st.spinner("Analyzing..."):
                    start_time = time.time() # Latency Timer Start
                    
                    # Generate
                    chunks = chunk_text(clean_text)
                    summary_parts = []
                    
                    # Summarize first 3 chunks to keep it fast
                    for i, chunk in enumerate(chunks[:3]):
                        try:
                            res = summarizer(chunk, max_length=128, min_length=30, do_sample=False)
                            summary_parts.append(res[0]['summary_text'])
                        except Exception as e:
                            st.warning(f"Could not summarize chunk {i+1}: {e}")
                    
                    final_summary = " ".join(summary_parts)
                    
                    end_time = time.time() # Latency Timer End
                    latency = end_time - start_time
                    
                    st.success(final_summary)
                    st.info(f"⏱️ Model Latency: {latency:.2f} seconds")