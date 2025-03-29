import streamlit as st
import os
import traceback
import json
from config import INSURANCE_SCHEMA
from extractors.pdf_extractor import extract_text_from_pdf
from extractors.gemini_extractor import GeminiFlashExtractor

# Ensure the GEMINI_API_KEY is available
if not os.environ.get('GEMINI_API_KEY') and os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                api_key = line.strip().split('=', 1)[1]
                if api_key.startswith('"') and api_key.endswith('"'):
                    api_key = api_key[1:-1]
                os.environ['GEMINI_API_KEY'] = api_key

# Main application
def main():
    st.set_page_config(page_title="Insurance Policy Data Extractor", layout="wide")

    st.title("Insurance Policy Data Extractor (Gemini 2.0 Flash)")
    st.write("Upload any insurance policy PDF to extract structured data using Google's Gemini 2.0 Flash AI model.")

    # Sidebar
    with st.sidebar:
        st.subheader("About")
        st.write("This app extracts standardized data from insurance policy PDFs.")
        api_key_status = "✅ API Key Found" if os.environ.get('GEMINI_API_KEY') else "❌ API Key Missing"
        st.info(f"Gemini API Status: {api_key_status}")

    # Upload section
    st.subheader("Upload Policy PDF")
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

    if uploaded_file and os.environ.get('GEMINI_API_KEY'):
        with st.spinner("Processing PDF..."):
            pdf_text, page_texts, document_hash = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.success("PDF processed successfully")
            # Add scrollable text area for full PDF content
            with st.expander("View Full PDF Text"):
                st.text_area("PDF Content", pdf_text, height=400)
          
            if st.button("Extract Policy Data", type="primary"):
                try:
                    with st.spinner("Extracting data with Gemini 2.0 Flash..."):
                        extractor = GeminiFlashExtractor()
                        extracted_data = extractor.extract_data(pdf_text, page_texts, document_hash)
                        
                        if extracted_data:
                            st.session_state.extracted_data = extracted_data
                            st.session_state.show_editor = True
                        else:
                            st.error("Failed to extract data")
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.error("Failed to process PDF")
    elif uploaded_file and not os.environ.get('GEMINI_API_KEY'):
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())