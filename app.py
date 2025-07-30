import streamlit as st
from rag_pipeline import generate_answer

# 🖼️ 페이지 세팅
st.set_page_config(page_title="MSADS Assistant", page_icon="🎓", layout="centered")

# 🎓 사이드바 로고와 소개
st.sidebar.image("https://www.uchicago.edu/assets/images/logos/primary-logo.svg", width=200)
st.sidebar.markdown("### MSADS Chatbot Assistant")
st.sidebar.markdown("Built with ❤️ by Group1 (2025)")

# 🏷️ 상단 타이틀
st.title("🎓 MSADS Chatbot Assistant")
st.markdown("Ask me anything about the [UChicago MSADS program](https://professional.uchicago.edu/find-your-fit/masters/master-science-applied-data-science).")

# 💬 사용자 질문 입력
user_question = st.text_input("💡 Enter your question below:")

# 🔍 질문이 있을 경우
if user_question:
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_question)

    st.success("✅ Answer")
    st.markdown(f"> {answer}")

# 📌 FAQ 추천 (선택)
st.markdown("----")
st.markdown("Try one of these example questions:")

col1, col2 = st.columns(2)
with col1:
    if st.button("📚 What are the core courses?"):
        st.experimental_set_query_params(question="What are the core courses of the MSADS program?")
with col2:
    if st.button("🎓 How do I apply?"):
        st.experimental_set_query_params(question="How do I apply to the MSADS program?")


