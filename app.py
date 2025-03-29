import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import traceback

from config import INSURANCE_SCHEMA, CACHE_DIR
from extractors.pdf_extractor import extract_text_from_pdf
from extractors.semantic_extractor import SemanticExtractor
from utils.export import get_csv_download, get_json_download, get_excel_download
from utils.feedback import save_feedback

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Main application
def main():
    st.set_page_config(page_title="Insurance Policy Data Extractor", layout="wide")
    
    st.title("Insurance Policy Data Extractor")
    st.write(
        "Upload any insurance policy PDF to extract standardized data using semantic search."
    )

    # Sidebar
    with st.sidebar:
        st.subheader("About")
        st.write("This app extracts standardized data from insurance policy PDFs.")
        st.write("How to use:")
        st.write("- Upload an insurance policy PDF")
        st.write("- Wait for extracted data using semantic search")
        st.write("- Finalize data and review into a standardized field formatting")
        st.write("- Download data in CSV, JSON, or Excel format for further use")

    # Main tabs
    tab1, tab2 = st.tabs(["Extract Data", "About"])

    with tab1:
        st.subheader("Upload Policy PDF")
        st.write("Upload any insurance policy PDF for standardized data extraction.")

        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

        if uploaded_file:
            with st.spinner("Processing PDF..."):
                pdf_text, page_texts, document_hash = extract_text_from_pdf(
                    uploaded_file
                )

            if pdf_text:
                st.success("PDF processed successfully")
                # Add scrollable text area for full PDF content
                with st.expander("View Full PDF Text"):
                    st.text_area("PDF Content", pdf_text, height=400)
              
                if st.button("Extract Policy Data", type="primary"):
                    with st.spinner("Extracting data with semantic search..."):
                        extractor = SemanticExtractor(pdf_text, page_texts, document_hash)
                        extracted_data = extractor.extract_data()

                        if extracted_data:
                            st.session_state.extracted_data = extracted_data
                            st.session_state.show_editor = True
                            st.session_state.original_values = {
                                category: {
                                    field: value for field, value in fields.items()
                                }
                                for category, fields in extracted_data.items()
                            }
                        else:
                            st.error("Failed to extract data")

                if (
                    hasattr(st.session_state, "show_editor")
                    and st.session_state.show_editor
                ):
                    st.subheader("Extracted Policy Data")

                    if "edited_data" not in st.session_state:
                        st.session_state.edited_data = (
                            st.session_state.extracted_data.copy()
                        )

                    # Statistics
                    total_fields = sum(
                        len(INSURANCE_SCHEMA[category]) for category in INSURANCE_SCHEMA
                    )
                    found_fields = sum(
                        1
                        for category in st.session_state.extracted_data
                        for field in st.session_state.extracted_data[category]
                        if st.session_state.extracted_data[category][field]
                        != "Not Found"
                    )

                    confidence = (
                        found_fields / total_fields * 100 if total_fields > 0 else 0
                    )

                    # Editor
                    with st.form("edit_form"):
                        # Use tabs for categories
                        category_tabs = st.tabs(list(INSURANCE_SCHEMA.keys()))

                        for i, (category, tab) in enumerate(
                            zip(INSURANCE_SCHEMA.keys(), category_tabs)
                        ):
                            with tab:
                                st.subheader(category)

                                for field in INSURANCE_SCHEMA[category]:
                                    value = "Not Found"
                                    if (
                                        category in st.session_state.extracted_data
                                        and field
                                        in st.session_state.extracted_data[category]
                                    ):
                                        value = st.session_state.extracted_data[
                                            category
                                        ][field]

                                    field_key = f"{category}_{field}"
                                    st.text_input(field, value=value, key=field_key)
                                    
                                    has_source = (
                                        "sources" in st.session_state.extracted_data
                                        and category in st.session_state.extracted_data["sources"]
                                        and field in st.session_state.extracted_data["sources"][category]
                                        and st.session_state.extracted_data["sources"][category][field]
                                        and value != "Not Found"
                                    )
                                    
                                    if has_source:
                                        with st.expander("View Source"):
                                            source_chunk = st.session_state.extracted_data["sources"][category][field]
                                            st.text_area(
                                                "Source Text", 
                                                source_chunk, 
                                                height=150, 
                                                key=f"source_{field_key}",
                                                disabled=True
                                            )

                        # Submit and feedback options
                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button("Save Changes")
                        with col2:
                            collect_feedback = st.checkbox(
                                "Submit corrections as feedback for model to learn from",
                                value=True,
                            )

                        if submitted:
                            # Update edited data
                            for category in INSURANCE_SCHEMA:
                                st.session_state.edited_data[category] = {}
                                for field in INSURANCE_SCHEMA[category]:
                                    field_key = f"{category}_{field}"
                                    if field_key in st.session_state:
                                        st.session_state.edited_data[category][
                                            field
                                        ] = st.session_state[field_key]
                                    else:
                                        st.session_state.edited_data[category][
                                            field
                                        ] = "Not Found"

                            # Store feedback
                            if collect_feedback and hasattr(
                                st.session_state, "document_hash"
                            ):
                                any_changes = False
                                for category in st.session_state.edited_data:
                                    if category == "sources":
                                        continue
                                        
                                    for field in st.session_state.edited_data[category]:
                                        if (
                                            category in st.session_state.original_values
                                            and field
                                            in st.session_state.original_values[
                                                category
                                            ]
                                            and st.session_state.original_values[
                                                category
                                            ][field]
                                            != st.session_state.edited_data[category][
                                                field
                                            ]
                                        ):
                                            any_changes = True
                                            break

                                if any_changes:
                                    feedback_saved = save_feedback(
                                        document_hash,
                                        st.session_state.original_values,
                                        st.session_state.edited_data,
                                    )

                                    if feedback_saved:
                                        st.success(
                                            "Changes saved and feedback collected"
                                        )
                                    else:
                                        st.success(
                                            "Changes saved (feedback not stored)"
                                        )
                                else:
                                    st.success("Changes saved (no changes detected)")
                            else:
                                st.success("Changes saved")

                    # Export options
                    st.subheader("Export Data")
                    st.write("All exports have the same standardized structure")

                    col1, col2, col3 = st.columns(3)

                    # Prepare data
                    export_data = {}
                    for category in INSURANCE_SCHEMA:
                        export_data[category] = {}
                        for field in INSURANCE_SCHEMA[category]:
                            value = "Not Found"
                            if (
                                category in st.session_state.edited_data
                                and field in st.session_state.edited_data[category]
                            ):
                                value = st.session_state.edited_data[category][field]
                            export_data[category][field] = value

                    # Create filenames with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d")
                    csv_filename = f"policy_data_{timestamp}.csv"
                    json_filename = f"policy_data_{timestamp}.json"
                    excel_filename = f"policy_data_{timestamp}.xlsx"

                    # Download buttons
                    with col1:
                        csv_data, csv_name = get_csv_download(export_data, csv_filename)
                        if csv_data is not None:
                            st.download_button(
                                "Download CSV",
                                data=csv_data,
                                file_name=csv_name,
                                mime="text/csv",
                            )

                    with col2:
                        json_data, json_name = get_json_download(
                            export_data, json_filename
                        )
                        if json_data is not None:
                            st.download_button(
                                "Download JSON",
                                data=json_data,
                                file_name=json_name,
                                mime="application/json",
                            )

                    with col3:
                        excel_data, excel_name = get_excel_download(
                            export_data, excel_filename
                        )
                        if excel_data is not None:
                            st.download_button(
                                "Download Excel",
                                data=excel_data,
                                file_name=excel_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    
            else:
                st.error("Failed to process PDF")

    with tab2:
        st.subheader("About This Extractor")

        st.write(
            "This tool uses semantic vector search to extract standardized insurance policy data."
        )

        st.write("**Key Features:**")

        st.write("1. **Semantic Vector Search**")
        st.write("- Uses TF-IDF vectorization to understand policy text")
        st.write("- Focuses on semantic meaning rather than exact patterns")
        st.write("- Works across different carriers and policy formats")

        st.write("2. **Standardized Output Structure**")
        st.write("- Consistent categories and fields for all policies")
        st.write("- Uniform value formatting")
        st.write("- Complete data structure (all fields present)")

        st.write("3. **Standardized Value Formatting**")
        st.write("- Percentages: '66.67% of Salary'")
        st.write("- Multipliers: '2x Annual Salary'")
        st.write("- Ages: 'Age 65'")
        st.write("- Time periods: '90 days', '24 months'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())

        # Create empty data structure for error case
        empty_data = {}
        for category in INSURANCE_SCHEMA:
            empty_data[category] = {}
            for field in INSURANCE_SCHEMA[category]:
                empty_data[category][field] = "Not Found"

        st.session_state.extracted_data = empty_data
        st.session_state.edited_data = empty_data