import streamlit as st

st.title("MSADS Chatbot Assistant")
st.markdown("Ask me anything about the MS in Applied Data Science program!")

user_question = st.text_input("Enter your question here:")

if user_question:
    st.write("Answer will appear here... (RAG response placeholder)")
