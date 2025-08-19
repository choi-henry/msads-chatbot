import sys
import os
import streamlit as st
from datetime import datetime

# ===== Paths & Imports =====
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from rag_pipeline import generate_answer  # 기존 함수 그대로 사용

# ===== Build FAISS if missing =====
faiss_path = os.path.join(base_dir, "faiss_index", "index.faiss")
if not os.path.exists(faiss_path):
    exit_code = os.system(f"python {os.path.join(base_dir, 'build_faiss.py')}")
    if exit_code != 0:
        st.error("⚠️ Failed to build FAISS index. Check logs for details.")

# ===== Page Config =====
st.set_page_config(page_title="MSADS Assistant", page_icon="🎓", layout="wide")

# ===== Sidebar (Logo & Intro) =====
st.sidebar.markdown("## 🎓 MSADS Chatbot Assistant")
# 로고: 로컬 파일 우선, 없으면 외부 URL 폴백
local_logo = os.path.join(base_dir, "assets", "uchicago_logo.png")
if os.path.exists(local_logo):
    st.sidebar.image(local_logo, use_column_width=True)
else:
    # 외부 URL (폴백)
    st.sidebar.image("https://www.uchicago.edu/assets/images/logos/primary-logo.svg", use_column_width=True)

st.sidebar.markdown("Built by Group1 (2025)")
st.sidebar.markdown("---")

# ===== Session State =====
if "preset_question" not in st.session_state:
    st.session_state.preset_question = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "answer" not in st.session_state:
    st.session_state.answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ===== Header (Top Banner) =====
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; padding:8px 0 2px 0;">
      <span style="font-size:28px;">🎓</span>
      <div style="font-size:26px; font-weight:700; color:#800000;">
        MSADS Chatbot Assistant
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Ask me anything about the **UChicago MSADS program**.")

# ===== Layout: Left (Input/FAQ) | Right (Answer/History) =====
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("❓ Ask a Question")

    # 입력/버튼을 하나의 form으로 묶어 Enter 제출 지원
    with st.form("qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "💡 Enter your question below:",
            value=st.session_state.preset_question or st.session_state.last_question,
            placeholder="e.g., What are the core courses of the MSADS program?"
        )
        col_a, col_b = st.columns(2)
        submit = col_a.form_submit_button("🔎 Get Answer", use_container_width=True)
        clear_q = col_b.form_submit_button("🧹 Clear", use_container_width=True)

    if clear_q:
        st.session_state.preset_question = ""
        st.session_state.last_question = ""
        st.session_state.answer = None
        st.rerun()

    st.markdown("---")
    st.subheader("📌 Quick FAQs")

    # FAQ 버튼 4개 (원래 2개 → 확장 가능)
    faq_col1, faq_col2 = st.columns(2)
    with faq_col1:
        if st.button("What are the core courses?", use_container_width=True):
            st.session_state.preset_question = "What are the core courses of the MSADS program?"
            st.rerun()
        if st.button("How do I apply?", use_container_width=True):
            st.session_state.preset_question = "How do I apply to the MSADS program?"
            st.rerun()
    with faq_col2:
        if st.button("How is the program structured?", use_container_width=True):
            st.session_state.preset_question = "How is the program structured (in-person vs online)?"
            st.rerun()
        if st.button("What is the capstone project?", use_container_width=True):
            st.session_state.preset_question = "What is the capstone project about?"
            st.rerun()

with right:
    st.subheader("💡 Answer")

    # 답변 생성
    if submit and user_question.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(user_question.strip())
            except Exception as e:
                st.error("⚠️ An error occurred while generating the answer. Check logs for details.")
                answer = None

        if answer:
            st.session_state.answer = answer
            st.session_state.last_question = user_question.strip()
            st.session_state.history.append({
                "q": user_question.strip(),
                "a": answer,
                "t": datetime.now().strftime("%H:%M")
            })

    # 답변 표시 (카드 스타일)
    if st.session_state.answer:
        st.success("Answer")
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #eaeaea;
                border-radius:14px;
                padding:18px;
                box-shadow:0 4px 16px rgba(0,0,0,0.06);
                line-height:1.65;
                ">
                {st.session_state.answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        # UX: 복사/다운로드/피드백
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ Download",
                data=str(st.session_state.answer).encode("utf-8"),
                file_name="answer.txt",
                mime="text/plain",
                use_container_width=True
            )
        with c2:
            st.button("👍 Helpful", use_container_width=True)
        with c3:
            st.button("👎 Not helpful", use_container_width=True)

    # 히스토리(최근 8개)
    st.markdown("### 🗂 History")
    if len(st.session_state.history) == 0:
        st.caption("No history yet. Try asking a question!")
    else:
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(f"**You** ({item['t']}): {item['q']}")
            st.markdown(
                f"<div style='background:#f7f7f9; padding:12px; border-radius:10px; border:1px solid #eee;'>{item['a']}</div>",
                unsafe_allow_html=True
            )
            st.divider()





