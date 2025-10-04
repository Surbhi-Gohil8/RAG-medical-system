import streamlit as st

def render_history_download():
    messages = st.session_state.get("messages", [])
    if messages:
        chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        st.download_button(
            "Download Chat History",
            chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )