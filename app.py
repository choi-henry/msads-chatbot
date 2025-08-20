import os, sys
import streamlit as st
from datetime import datetime

# ---- load token from secrets -> env
for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
    if k in st.secrets:
        os.environ[k] = st.secrets[k].strip()

# collect candidates & validate
candidates = []
for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
    v = os.environ.get(k)
    if v:
        candidates.append(v.strip())

def _pick_valid_hf_token(cands):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for tok in cands:
            try:
                api.whoami(token=tok)  # valid?
                for kk in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
                    os.environ[kk] = tok
                return True
            except Exception:
                continue
        return False
    except Exception:
        if cands:
            tok = cands[0]
            for kk in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
                os.environ[kk] = tok
            return True

ok = _pick_valid_hf_token(candidates)
if not ok:
    for kk in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if kk in os.environ:
            del os.environ[kk]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== Paths & Imports =====
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from rag_pipeline import generate_answer 

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

logo_path = os.path.join(base_dir, "University Logo", "SVG_RGB_Digital", "University Logo_1Color_Maroon_RGB.svg")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.image("https://toppng.com/vector/the-university-of-chicago-logo-vector/460006")

st.sidebar.markdown("Built by Group1 (2025)")
st.sidebar.markdown("---")

with st.sidebar.expander("Dev · HF Auth Check", expanded=False):
    if st.button("Test HF auth (config only)"):
        try:
            from huggingface_hub import HfApi
            from transformers import AutoConfig
            api = HfApi()
            me = api.whoami(token=os.environ.get("HF_TOKEN"))
            _ = AutoConfig.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=True
            )
            st.success(f"✅ Auth OK, @{me.get('name') or me.get('email')}")
        except Exception as e:
            st.exception(e)
            
# ===== Session State =====
if "history" not in st.session_state:
    st.session_state.history = []
if "answer" not in st.session_state:
    st.session_state.answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ===== Helper: run QA =====
def run_qa(q: str):
    q = (q or "").strip()
    if not q:
        return
    with st.spinner("⏳ Generating with Mistral-7B…"):
        try:
            ans = generate_answer(q)
        except Exception as e:
            st.exception(e)   # 전체 에러 스택까지 출력
            st.stop()
    st.session_state.answer = ans
    st.session_state.last_question = q
    st.session_state.history.append(
        {"q": q, "a": ans, "t": datetime.now().strftime("%H:%M")}
    )

# ===== Header =====
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; padding:4px 0 2px 0;">
      <span style="font-size:28px;">🎓</span>
      <div style="font-size:26px; font-weight:700; color:#800000;">
        MSADS Chatbot Assistant
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Ask me anything about the **UChicago MSADS program**.")

# ===== Layout: Left (Quick FAQs) | Right (Ask/Answer/History) =====
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("📌 Quick FAQs")
    faq_questions = [
        "What are the core courses in the MS in Applied Data Science program?",
        "What are some elective courses in the MS in Applied Data Science program?",
        "How do I apply to the MSADS program?",
        "Can you provide information on Generative AI Principles course?",
        "What is the cost per course for the MS in Applied Data Science program?",
        "How do I apply to the MBA/MS joint degree program?",
    ]
    for i, q in enumerate(faq_questions):
        if st.button(q, key=f"faq_{i}", use_container_width=True):
            run_qa(q)

with right:
    # ---- Ask a Question ----
    st.subheader("Inquire About the MSADS Program at UChicago")
    with st.form("qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "💡 Enter your question below:",
            value=st.session_state.last_question,
            placeholder="e.g., What are the core courses in the MSADS program?"
        )
        colA, colB = st.columns(2)
        submit = colA.form_submit_button("🔎 Get Answer")
        clear_q = colB.form_submit_button("🧹 Clear")

    if submit:
        run_qa(user_question)

    if clear_q:
        st.session_state.answer = None
        st.session_state.last_question = ""
        st.rerun()

    # ---- Answer ----
    st.subheader("💡 Answer")
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
        # optional: 다운로드/피드백 버튼 원하면 여기에 추가

    # ---- History ----
    st.subheader("📂 History")
    if not st.session_state.history:
        st.caption("No history yet. Try a FAQ or ask a question!")
    else:
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(f"**You** ({item['t']}): {item['q']}")
            st.markdown(
                f"<div style='background:#f7f7f9; padding:12px; border-radius:10px; border:1px solid #eee;'>{item['a']}</div>",
                unsafe_allow_html=True
            )
            st.divider()

import base64

# ===== Footer (precise center) =====
st.markdown("---")

logo_path = os.path.join(
    base_dir, "University Logo", "SVG_RGB_Digital",
    "University Logo_1Color_Maroon_RGB.svg"
)

with open(logo_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <div style="text-align:center; padding:6px 0 0 0;">
      <img src="data:image/svg+xml;base64,{img_b64}" style="display:block; margin:0 auto;" width="220" />
      <div style="color:grey; font-size:14px; margin-top:6px;">
        MSADS Chatbot Assistant · Midterm Project · Built by Group1 (2025)
      </div>
    </div>
    """,
    unsafe_allow_html=True
)






























