import pandas as pd
import json
import io
import streamlit as st
from config import INSURANCE_SCHEMA

def get_csv_download(data, filename="insurance_data.csv"):
    """
    Generate a CSV file containing extracted insurance data
    
    Args:
        data: Dictionary of extracted values by category and field
        filename: Name of the CSV file
        
    Returns:
        tuple: (csv_data, filename) or (None, None) on error
    """
    try:
        # Convert to DataFrame
        rows = []
        for category in INSURANCE_SCHEMA:
            for field in INSURANCE_SCHEMA[category]:
                value = "Not Found"
                if category in data and field in data[category]:
                    value = data[category][field]
                rows.append({"Category": category, "Field": field, "Value": value})

        df = pd.DataFrame(rows)

        # Convert to CSV
        csv = df.to_csv(index=False).encode("utf-8")
        return csv, filename
    except Exception as e:
        st.error(f"Error generating CSV: {e}")
        return None, None


def get_json_download(data, filename="insurance_data.json"):
    """
    Generate a JSON file containing extracted insurance data
    
    Args:
        data: Dictionary of extracted values by category and field
        filename: Name of the JSON file
        
    Returns:
        tuple: (json_data, filename) or (None, None) on error
    """
    try:
        # Ensure consistent structure
        structured_data = {}
        for category in INSURANCE_SCHEMA:
            structured_data[category] = {}
            for field in INSURANCE_SCHEMA[category]:
                value = "Not Found"
                if category in data and field in data[category]:
                    value = data[category][field]
                structured_data[category][field] = value

        # Convert to JSON
        json_str = json.dumps(structured_data, indent=4).encode("utf-8")
        return json_str, filename
    except Exception as e:
        st.error(f"Error generating JSON: {e}")
        return None, None


def get_excel_download(data, filename="insurance_data.xlsx"):
    """
    Generate an Excel file containing extracted insurance data
    
    Args:
        data: Dictionary of extracted values by category and field
        filename: Name of the Excel file
        
    Returns:
        tuple: (excel_data, filename) or (None, None) on error
    """
    try:
        # Create Excel file
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            # Summary sheet
            summary_rows = []
            for category in INSURANCE_SCHEMA:
                for field in INSURANCE_SCHEMA[category]:
                    value = "Not Found"
                    if category in data and field in data[category]:
                        value = data[category][field]
                    summary_rows.append(
                        {"Category": category, "Field": field, "Value": value}
                    )

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Category sheets
            for category in INSURANCE_SCHEMA:
                category_rows = []
                for field in INSURANCE_SCHEMA[category]:
                    value = "Not Found"
                    if category in data and field in data[category]:
                        value = data[category][field]
                    category_rows.append({"Field": field, "Value": value})

                category_df = pd.DataFrame(category_rows)
                sheet_name = category[:31]  # Excel limit
                category_df.to_excel(writer, sheet_name=sheet_name, index=False)

        buffer.seek(0)
        return buffer, filename
    except Exception as e:
        st.error(f"Error generating Excel: {e}")
        return None, None