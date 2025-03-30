import streamlit as st
import os
import traceback
import json
import datetime
import base64
import pandas as pd
import io
from config import INSURANCE_SCHEMA
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

def get_csv_download(data, filename):
    """Create downloadable CSV content from nested JSON - includes all fields"""
    # Flatten the nested JSON structure
    flattened_data = []
    
    # Process top-level fields
    row = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            row[key] = value
    
    # Process nested fields - include ALL fields from schema
    for category, category_schema in INSURANCE_SCHEMA.items():
        if isinstance(category_schema, dict):
            for field in category_schema:
                # Get value from data if it exists, otherwise use empty string
                value = ""
                if category in data and isinstance(data[category], dict) and field in data[category]:
                    value = data[category][field]
                row[f"{category} - {field}"] = value
    
    flattened_data.append(row)
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(flattened_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    return csv_string, filename

def create_csv_view(data):
    """Create a DataFrame view for CSV display - includes all fields"""
    flat_data = []
    
    # Include top-level fields
    for key, value in data.items():
        if not isinstance(value, dict):
            flat_data.append({
                "Category": "General",
                "Field": key,
                "Value": value
            })
    
    # Include ALL fields from schema
    for category, category_schema in INSURANCE_SCHEMA.items():
        if isinstance(category_schema, dict):
            for field in category_schema:
                # Get value from data if it exists, otherwise use "none" or "$0"
                value = "none"
                if field.lower().find("amount") >= 0:
                    value = "$0"
                if category in data and isinstance(data[category], dict) and field in data[category]:
                    value = data[category][field]
                
                flat_data.append({
                    "Category": category,
                    "Field": field,
                    "Value": value
                })
    
    return pd.DataFrame(flat_data)

def display_pdf(file_bytes):
    """Display PDF in Streamlit UI"""
    # Encode PDF as base64 string for embedding
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    
    # Create an HTML component to display the PDF
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="750" type="application/pdf"></iframe>
    """
    
    # Display the PDF using HTML component
    st.markdown(pdf_display, unsafe_allow_html=True)

def generate_document_hash(pdf_bytes):
    """Generate a simple hash for the document"""
    import hashlib
    return hashlib.md5(pdf_bytes).hexdigest()

# Main application
def main():
    st.set_page_config(page_title="Insurance Policy Data Extractor", layout="wide")

    st.title("Insurance Policy AI Data Extractor")
    st.write("Upload any insurance policy PDF to extract predefined structured data using a custom trained insurance policy AI model.")

    # Sidebar
    with st.sidebar:
 
        api_key_status = "✅ API Key Found" if os.environ.get('GEMINI_API_KEY') else "❌ API Key Missing"
        st.info(f"Model API Status: {api_key_status}")

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
        # Get raw PDF bytes
        pdf_bytes = uploaded_file.getvalue()
        
        # Generate document hash
        document_hash = generate_document_hash(pdf_bytes)
        
        # Display PDF preview in an expander
        with st.expander("Click to view PDF Preview", expanded=False):
            display_pdf(pdf_bytes)
        
        
        st.success("PDF loaded successfully")

        if st.button("Extract Policy Data", type="primary"):
            try:
                with st.spinner("Model extracting data from PDF..."):
                    extractor = GeminiFlashExtractor()
                    # Pass only the raw PDF bytes - no text extraction
                    extracted_data = extractor.extract_data(pdf_bytes, None, document_hash)
                    
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
                st.subheader("Export Structured Data")
                
                # Create timestamp for filenames
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                
                # JSON export button
                with col1:
                    json_filename = f"insurance_data_{timestamp}.json"
                    json_data, json_name = get_json_download(st.session_state.edited_data, json_filename)
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name=json_name,
                        mime="application/json"
                )
                
                                
                # Display outputs at the end (closed by default)
                st.subheader("Structured Data Output Preview")
                
                # CSV output - closed by default
                with st.expander("CSV Output", expanded=False):
                    # Create CSV view from current session state - include all fields
                    df = create_csv_view(st.session_state.edited_data)
                    st.dataframe(df)
                    
                        # JSON output - closed by default
                with st.expander("JSON Output", expanded=False):
                    # Format the JSON with indentation for better readability
                    formatted_json = json.dumps(st.session_state.edited_data, indent=2)
                    st.code(formatted_json, language="json")
                    
                
                # CSV export button
                with col2:
                    csv_filename = f"insurance_data_{timestamp}.csv"
                    csv_data, csv_name = get_csv_download(st.session_state.edited_data, csv_filename)
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name=csv_name,
                        mime="text/csv"
                    )

    elif uploaded_file and not os.environ.get('GEMINI_API_KEY'):
        st.error("Model API Key is missing. Please set the GEMINI_API_KEY environment variable.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())