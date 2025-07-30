import streamlit as st
from rag_pipeline import generate_answer


st.set_page_config(page_title="MSADS Assistant", page_icon="🎓", layout="centered")


st.sidebar.image("https://www.uchicago.edu/assets/images/logos/primary-logo.svg", width=200)
st.sidebar.markdown("### MSADS Chatbot Assistant")
st.sidebar.markdown("Built by Group1 (2025)")

st.title("🎓 MSADS Chatbot Assistant")
st.markdown("Ask me anything about the [UChicago MSADS program](https://ms-ads.datascience.uchicago.edu/).")

user_question = st.text_input("Enter your question below:")

if user_question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_question)

    st.success("Answer")
    st.markdown(f"> {answer}")

# FAQ
st.markdown("----")
st.markdown("Try one of these example questions:")

col1, col2 = st.columns(2)
with col1:
    if st.button(" What are the core courses?"):
        st.experimental_set_query_params(question="What are the core courses of the MSADS program?")
with col2:
    if st.button(" How do I apply?"):
        st.experimental_set_query_params(question="How do I apply to the MSADS program?")


