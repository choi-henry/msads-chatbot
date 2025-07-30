import streamlit as st
from rag_pipeline import generate_answer

st.set_page_config(page_title="MSADS Assistant", page_icon="🎓", layout="centered")

st.sidebar.image("https://www.uchicago.edu/assets/images/logos/primary-logo.svg", width=200)
st.sidebar.markdown("### MSADS Chatbot Assistant")
st.sidebar.markdown("Built by Group1 (2025)")

if "preset_question" not in st.session_state:
    st.session_state.preset_question = ""

st.title("🎓 MSADS Chatbot Assistant")
st.markdown("Ask me anything about the [UChicago MSADS program](https://ms-ads.datascience.uchicago.edu/).")

# FAQ
col1, col2 = st.columns(2)
with col1:
    if st.button(" What are the core courses?"):
        st.session_state.preset_question = "What are the core courses of the MSADS program?"
        st.experimental_rerun()
with col2:
    if st.button(" How do I apply?"):
        st.session_state.preset_question = "How do I apply to the MSADS program?"
        st.experimental_rerun()

user_question = st.text_input("💡 Enter your question below:", value=st.session_state.preset_question)

if user_question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_question)
    st.success(" Answer")
    st.markdown(f"> {answer}")
