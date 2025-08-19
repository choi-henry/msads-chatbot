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

# ===== Sidebar =====
st.sidebar.markdown("## 🎓 MSADS Chatbot Assistant")
logo_path = os.path.join(base_dir, "assets", "uchicago_logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path)
else:
    st.sidebar.image("https://www.uchicago.edu/assets/images/logos/primary-logo.svg")
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

# ===== Header =====
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; padding:4px 0;">
      <span style="font-size:28px;">🎓</span>
      <div style="font-size:26px; font-weight:700; color:#800000;">
        MSADS Chatbot Assistant
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Ask me anything about the **UChicago MSADS program**.")

# ===== Layout (2 cols) =====
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("❓ Ask a Question")

    with st.form("qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "💡 Enter your question below:",
            value=st.session_state.preset_question or st.session_state.last_question,
            placeholder="e.g., What are the core courses?"
        )
        colA, colB = st.columns(2)
        submit = colA.form_submit_button("🔎 Get Answer")
        clear_q = colB.form_submit_button("🧹 Clear")

    if clear_q:
        st.session_state.preset_question = ""
        st.session_state.last_question = ""
        st.session_state.answer = None
        st.rerun()

    st.markdown("---")
    st.subheader("📌 Quick FAQs")
    faq_qs = [
        "What are the core courses of the MSADS program?",
        "How do I apply to the MSADS program?",
        "How is the program structured (in-person vs online)?",
        "What is the capstone project about?"
    ]
    for q in faq_qs:
        if st.button(q, use_container_width=True):
            st.session_state.preset_question = q
            st.rerun()

with right:
    st.subheader("💡 Answer")

    if submit and user_question.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(user_question.strip())
            except Exception as e:
                st.error("⚠️ Error while generating the answer.")
                answer = None

        if answer:
            st.session_state.answer = answer
            st.session_state.last_question = user_question.strip()
            st.session_state.history.append({
                "q": user_question.strip(),
                "a": answer,
                "t": datetime.now().strftime("%H:%M")
            })

    if st.session_state.answer:
        st.markdown(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #eaeaea;
                border-radius:14px;
                padding:18px;
                box-shadow:0 2px 10px rgba(0,0,0,0.05);
                line-height:1.6;">
                {st.session_state.answer}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📂 History")
    if not st.session_state.history:
        st.caption("No history yet. Try asking a question!")
    else:
        for item in reversed(st.session_state.history[-6:]):
            st.markdown(f"**You** ({item['t']}): {item['q']}")
            st.markdown(
                f"<div style='background:#f7f7f9; padding:12px; border-radius:10px; border:1px solid #eee;'>{item['a']}</div>",
                unsafe_allow_html=True
            )
            st.divider()






