import streamlit as st
import os
import traceback
import json
import datetime
import base64
import pandas as pd
import io
from config import INSURANCE_SCHEMA
from extractors.model_extractor import GeminiFlashExtractor

# Ensure the GEMINI_API_KEY is available
if not os.environ.get("GEMINI_API_KEY") and os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
                if api_key.startswith('"') and api_key.endswith('"'):
                    api_key = api_key[1:-1]
                os.environ["GEMINI_API_KEY"] = api_key


def get_json_download(data, filename):
    """Create downloadable JSON content with only values (no source fields)"""
    # Create a clean copy of the data with only values
    clean_data = {}
    
    # Handle top-level fields
    for key, value in data.items():
        if not isinstance(value, dict):
            clean_data[key] = value
        elif isinstance(value, dict) and "value" in value:
            clean_data[key] = value["value"]
        elif isinstance(value, dict) and all(isinstance(v, dict) and "value" in v for v in value.values()):
            # This is a category with fields that have value/source structure
            clean_data[key] = {}
            for field, field_data in value.items():
                clean_data[key][field] = field_data["value"]
    
    json_str = json.dumps(clean_data, indent=2)
    return json_str, filename


def get_csv_download(data, filename):
    """Create downloadable CSV content from nested JSON - includes only values, not sources"""
    # Flatten the nested JSON structure
    flattened_data = []

    # Process top-level fields
    row = {}
    for key, value in data.items():
        if key != "Company Name" and key != "Policy #" and not isinstance(value, dict):
            continue  # Skip categories, we'll handle them below
            
        if not isinstance(value, dict):
            row[key] = value
        elif isinstance(value, dict) and "value" in value:
            row[key] = value["value"]

    # Process nested fields - include ALL fields from schema
    for category, category_schema in INSURANCE_SCHEMA.items():
        if isinstance(category_schema, dict) and category != "Company Name" and category != "Policy #":
            for field in category_schema:
                # Get value from data if it exists, otherwise use empty string
                value = ""
                if (
                    category in data
                    and isinstance(data[category], dict)
                    and field in data[category]
                ):
                    if isinstance(data[category][field], dict) and "value" in data[category][field]:
                        value = data[category][field]["value"]
                    else:
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
    """Create a DataFrame view for CSV display - includes only values, not sources"""
    flat_data = []

    # Include top-level fields
    for key, value in data.items():
        if key not in ["Company Name", "Policy #"] and isinstance(value, dict) and not "value" in value:
            continue  # Skip categories, we'll handle them below
            
        if not isinstance(value, dict):
            flat_data.append({"Category": "General", "Field": key, "Value": value})
        elif isinstance(value, dict) and "value" in value:
            flat_data.append({"Category": "General", "Field": key, "Value": value["value"]})

    # Include ALL fields from schema
    for category, category_schema in INSURANCE_SCHEMA.items():
        if isinstance(category_schema, dict) and category != "Company Name" and category != "Policy #":
            for field in category_schema:
                # Get value from data if it exists, otherwise use "none" or "$0"
                value = "none"
                if field.lower().find("amount") >= 0:
                    value = "$0"
                if (
                    category in data
                    and isinstance(data[category], dict)
                    and field in data[category]
                ):
                    if isinstance(data[category][field], dict) and "value" in data[category][field]:
                        value = data[category][field]["value"]
                    else:
                        value = data[category][field]

                flat_data.append({"Category": category, "Field": field, "Value": value})

    return pd.DataFrame(flat_data)


def create_json_preview(data):
    """Create a clean JSON preview with only values (no source fields)"""
    # Create a clean copy of the data with only values
    clean_data = {}
    
    # Handle top-level fields
    for key, value in data.items():
        if not isinstance(value, dict):
            clean_data[key] = value
        elif isinstance(value, dict) and "value" in value:
            clean_data[key] = value["value"]
        elif isinstance(value, dict) and all(isinstance(v, dict) and "value" in v for v in value.values()):
            # This is a category with fields that have value/source structure
            clean_data[key] = {}
            for field, field_data in value.items():
                clean_data[key][field] = field_data["value"]
    
    return json.dumps(clean_data, indent=2)


def check_api_connection():
    """Test if the Gemini API connection is working"""
    import os
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return False, "No API key provided"

    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Simple test request - get available models
        model_list = genai.list_models()

        # If we get here without exception, the API is working
        return True, "Connection successful"
    except Exception as e:
        # Return the specific error message
        return False, str(e)


def display_pdf(file_bytes):
    """Display PDF in Streamlit UI"""
    # Encode PDF as base64 string for embedding
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")

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
    st.write(
        "Upload any insurance policy PDF to extract predefined structured data using a custom trained insurance policy AI model."
    )

    # Sidebar
    with st.sidebar:
        api_key_status = (
            "✅ API Key Found"
            if os.environ.get("GEMINI_API_KEY")
            else "❌ API Key Missing"
        )
        st.info(f"Model ENV API Status: {api_key_status}")

        # Add the connection test
        if os.environ.get("GEMINI_API_KEY"):
            if st.button("Test API Connection"):
                with st.spinner("Testing API connection..."):
                    is_connected, message = check_api_connection()

                    if is_connected:
                        st.success(f"✅ API Connection: {message}")
                    else:
                        st.error(f"❌ API Connection Failed: {message}")
        else:
            st.warning("Add API key to test connection")
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

    if uploaded_file and os.environ.get("GEMINI_API_KEY"):
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
                    extracted_data = extractor.extract_data(
                        pdf_bytes, None, document_hash
                    )

                    if extracted_data:
                        st.session_state.extracted_data = extracted_data
                        st.session_state.edited_data = json.loads(
                            json.dumps(extracted_data)
                        )  # Deep copy
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
                    company_name = st.session_state.edited_data.get("Company Name", {})
                    if isinstance(company_name, dict) and "value" in company_name:
                        company_name_value = company_name["value"]
                        company_source = company_name["source"]
                    else:
                        company_name_value = company_name
                        company_source = ""
                    
                    company_name_edited = st.text_input(
                        "Company Name", value=company_name_value
                    )
                    with st.expander("View Source", expanded=False):
                        st.text_area("Source Context", value=company_source, height=150, disabled=True)
                    
                with col2:
                    policy_num = st.session_state.edited_data.get("Policy #", {})
                    if isinstance(policy_num, dict) and "value" in policy_num:
                        policy_num_value = policy_num["value"]
                        policy_source = policy_num["source"]
                    else:
                        policy_num_value = policy_num
                        policy_source = ""
                        
                    policy_num_edited = st.text_input("Policy #", value=policy_num_value)
                    with st.expander("View Source", expanded=False):
                        st.text_area("Source Context", value=policy_source, height=150, disabled=True)

                # Create tabs for categories
                categories = [
                    k
                    for k in INSURANCE_SCHEMA.keys()
                    if isinstance(INSURANCE_SCHEMA[k], dict) and k != "Company Name" and k != "Policy #"
                ]
                tabs = st.tabs(categories)

                # Dictionary to store form values
                form_values = {}
                form_sources = {}

                for i, (category, tab) in enumerate(zip(categories, tabs)):
                    with tab:
                        if category in st.session_state.edited_data:
                            # Create input fields for each item in the category
                            for field in INSURANCE_SCHEMA[category]:
                                field_data = st.session_state.edited_data[category].get(field, {})
                                
                                if isinstance(field_data, dict) and "value" in field_data:
                                    value = field_data["value"]
                                    source = field_data["source"]
                                else:
                                    value = field_data
                                    source = ""
                                
                                st.subheader(field)
                                key_value = f"{category}_{field}_value"
                                key_source = f"{category}_{field}_source"
                                
                                # Input for the value
                                form_values[key_value] = st.text_input(
                                    "Value", value=value, key=key_value
                                )
                                
                                # Display source in a collapsible section (read-only)
                                with st.expander("View Source", expanded=False):
                                    form_sources[key_source] = st.text_area(
                                        "Source Context", value=source, height=150, key=key_source, disabled=True
                                    )
                                
                                st.divider()

                # Submit button
                submit_button = st.form_submit_button("Save Changes")

                if submit_button:
                    # Update edited data structure with form inputs
                    if isinstance(st.session_state.edited_data.get("Company Name", {}), dict) and "value" in st.session_state.edited_data.get("Company Name", {}):
                        st.session_state.edited_data["Company Name"]["value"] = company_name_edited
                    else:
                        st.session_state.edited_data["Company Name"] = {"value": company_name_edited, "source": company_source}
                    
                    if isinstance(st.session_state.edited_data.get("Policy #", {}), dict) and "value" in st.session_state.edited_data.get("Policy #", {}):
                        st.session_state.edited_data["Policy #"]["value"] = policy_num_edited
                    else:
                        st.session_state.edited_data["Policy #"] = {"value": policy_num_edited, "source": policy_source}

                    for category in categories:
                        if category not in st.session_state.edited_data:
                            st.session_state.edited_data[category] = {}

                        for field in INSURANCE_SCHEMA[category]:
                            key_value = f"{category}_{field}_value"
                            key_source = f"{category}_{field}_source"
                            
                            if key_value in form_values:
                                if isinstance(st.session_state.edited_data[category].get(field, {}), dict) and "source" in st.session_state.edited_data[category].get(field, {}):
                                    st.session_state.edited_data[category][field]["value"] = form_values[key_value]
                                else:
                                    # Get source from form or use empty string
                                    source = form_sources.get(key_source, "")
                                    st.session_state.edited_data[category][field] = {
                                        "value": form_values[key_value],
                                        "source": source
                                    }

                    st.success("Changes saved!")

            if st.session_state.edited_data:
                st.subheader("Structured Data Output Preview")

                with st.expander("CSV Output", expanded=False):
                    df = create_csv_view(st.session_state.edited_data)
                    st.dataframe(df)

                with st.expander("JSON Output", expanded=False):
                    formatted_json = create_json_preview(st.session_state.edited_data)
                    st.code(formatted_json, language="json")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                st.subheader("Export Structured Data")
                col1, col2 = st.columns(2)

                with col1:
                    json_filename = f"insurance_data_{timestamp}.json"
                    json_data, json_name = get_json_download(
                        st.session_state.edited_data, json_filename
                    )
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name=json_name,
                        mime="application/json",
                    )

                with col2:
                    csv_filename = f"insurance_data_{timestamp}.csv"
                    csv_data, csv_name = get_csv_download(
                        st.session_state.edited_data, csv_filename
                    )
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name=csv_name,
                        mime="text/csv",
                    )

    elif uploaded_file and not os.environ.get("GEMINI_API_KEY"):
        st.error(
            "Model ENV API Key is missing. Please set the GEMINI_API_KEY environment variable."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())