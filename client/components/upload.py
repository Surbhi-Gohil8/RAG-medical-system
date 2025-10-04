import streamlit as st
from utils.api import upload_pdfs_api


def render_uploader():
    st.sidebar.header("Upload Medical documents (.PDFs)")
    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple PDFs", type="pdf", accept_multiple_files=True
    )

    if st.sidebar.button("Upload to Vector DB"):
        if not uploaded_files:
            st.sidebar.warning("Please select at least one PDF.")
            return
        with st.sidebar.status("Uploading and indexing...", expanded=True) as status:
            with st.spinner("Processing documents (split, embed, upsert)..."):
                response = upload_pdfs_api(uploaded_files)
            if response.status_code == 200:
                st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) successfully.")
                st.toast("Vector store updated.")
            else:
                st.sidebar.error(f"Upload failed: {response.text}")