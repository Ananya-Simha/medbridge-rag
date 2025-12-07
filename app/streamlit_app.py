import streamlit as st
import requests

API_URL = "http://34.26.106.102:8000/answer"  # replace with your VM external IP

st.set_page_config(page_title="MedBridge", layout="wide")
st.title("MedBridge: Patient-Friendly Medical Answers")

st.write(
    "Ask a medical question. The system retrieves evidence from MedQuAD and "
    "generates a simplified, cited answer."
)

question = st.text_area("Your question", height=100)

if st.button("Get answer") and question.strip():
    with st.spinner("Generating answer..."):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
        else:
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            passages = data.get("passages", [])
            if passages:
                st.subheader("Sources")
                for i, p in enumerate(passages, start=1):
                    url = p.get("url") or "No URL"
                    topic = p.get("topic") or ""
                    st.markdown(f"{i}. [{url}]({url}) — *{topic}*")
                with st.expander("Show retrieved passages"):
                    for i, p in enumerate(passages, start=1):
                        st.markdown(f"**[{i}]** {p['answer_chunk']}")
