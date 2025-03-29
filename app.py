import streamlit as st
import os
import traceback
import json
import datetime
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

def get_json_download(data, filename):
    """Create downloadable JSON content"""
    json_str = json.dumps(data, indent=2)
    return json_str, filename

# Main application
def main():
    st.set_page_config(page_title="Insurance Policy Data Extractor", layout="wide")

    st.title("Insurance Policy AI Data Extractor ")
    st.write("Upload any insurance policy PDF to extract structured data using a custom trained insurance policy AI model.")

    # Sidebar
    with st.sidebar:
        st.subheader("About")
        st.write("This app extracts standardized data from insurance policy PDFs.")
        api_key_status = "✅ API Key Found" if os.environ.get('GEMINI_API_KEY') else "❌ API Key Missing"
        st.info(f"Gemini API Status: {api_key_status}")

    # Initialize session state
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = {}
    if "edited_data" not in st.session_state:
        st.session_state.edited_data = {}
    if "show_editor" not in st.session_state:
        st.session_state.show_editor = False

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
                    with st.spinner("Model extracting data..."):
                        extractor = GeminiFlashExtractor()
                        extracted_data = extractor.extract_data(pdf_text, page_texts, document_hash)
                        
                        if extracted_data:
                            st.session_state.extracted_data = extracted_data
                            st.session_state.edited_data = json.loads(json.dumps(extracted_data))  # Deep copy
                            st.session_state.show_editor = True
                        else:
                            st.error("Failed to extract data")
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
                    st.error(traceback.format_exc())
            
            # Display editable fields if data has been extracted
            if st.session_state.show_editor:
                st.subheader("Review and Edit Extracted Data")
                
                # Create form for editing
                with st.form(key="edit_form"):
                    # Top level fields (Company Name and Policy #)
                    col1, col2 = st.columns(2)
                    with col1:
                        company_name = st.session_state.edited_data.get("Company Name", "")
                        company_name_edited = st.text_input("Company Name", value=company_name)
                    with col2:
                        policy_num = st.session_state.edited_data.get("Policy #", "")
                        policy_num_edited = st.text_input("Policy #", value=policy_num)
                    
                    # Create tabs for categories
                    categories = [k for k in INSURANCE_SCHEMA.keys() if isinstance(INSURANCE_SCHEMA[k], dict)]
                    tabs = st.tabs(categories)
                    
                    # Dictionary to store form values
                    form_values = {}
                    
                    for i, (category, tab) in enumerate(zip(categories, tabs)):
                        with tab:
                            if category in st.session_state.edited_data:
                                # Create input fields for each item in the category
                                for field in INSURANCE_SCHEMA[category]:
                                    value = st.session_state.edited_data[category].get(field, "")
                                    key = f"{category}_{field}"
                                    form_values[key] = st.text_input(field, value=value, key=key)
                    
                    # Submit button
                    submit_button = st.form_submit_button("Save Changes")
                    
                    if submit_button:
                        # Update edited data structure with form inputs
                        st.session_state.edited_data["Company Name"] = company_name_edited
                        st.session_state.edited_data["Policy #"] = policy_num_edited
                        
                        for category in categories:
                            if category not in st.session_state.edited_data:
                                st.session_state.edited_data[category] = {}
                            
                            for field in INSURANCE_SCHEMA[category]:
                                key = f"{category}_{field}"
                                if key in form_values:
                                    st.session_state.edited_data[category][field] = form_values[key]
                        
                        st.success("Changes saved!")
                
                # Export section
                if st.session_state.edited_data:
                    st.subheader("Export Data")
                    
                    # Create JSON export button
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"insurance_data_{timestamp}.json"
                    
                    json_data, json_name = get_json_download(st.session_state.edited_data, filename)
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name=json_name,
                        mime="application/json"
                    )
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