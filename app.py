import streamlit as st
from transformers import pipeline

# Streamlit App
st.set_page_config(page_title="T5 FineTuning Summarizer", layout="centered")
# Load the summarization pipeline
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="trohith89/KDTS_T5_Summary_FineTune")

pipe = load_model()


# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextArea textarea {
            height: 300px !important;
            font-size: 16px;
        }
        .headline {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4B8BBE;
            padding: 20px;
        }
        .stButton>button {
            background-color: #4B8BBE;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Headline
st.markdown('<div class="headline">T5 FineTuning Summarizer</div>', unsafe_allow_html=True)

# Text input
user_input = st.text_area("Enter your long text below:", height=300, placeholder="Paste or type your content here...")

# Summarize button
if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Generating summary..."):
            summary = pipe(user_input, max_length=150, min_length=30, do_sample=False)[0]['generated_text']
        st.subheader("📝 Summary:")
        st.success(summary)
    else:
        st.warning("Please enter some text to summarize.")
