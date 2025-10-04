import streamlit as st
from utils.api import ask_question


def render_chat():
    st.subheader("💬 Chat with your assistant")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    # render existing chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # input and response
    user_input=st.chat_input("Type your question....")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role":"user","content":user_input})

        with st.spinner("Generating answer..."):
            response = ask_question(user_input)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "")
            sources = data.get("sources", [])

            st.chat_message("assistant").markdown(answer)
            # Show sources under the assistant message
            if sources:
                with st.expander("📄 Sources"):
                    for i, src in enumerate(sources, start=1):
                        src_file = src.get("source", "")
                        page = src.get("page")
                        page_str = f" (page {page})" if page is not None else ""
                        st.markdown(f"- `{src_file}`{page_str}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.text}")