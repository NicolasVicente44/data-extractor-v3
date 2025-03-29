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
        # Convert to DataFrame with flattened structure
        rows = []
        
        # Process top-level fields
        for key in data:
            if key == "sources":
                continue
                
            if isinstance(data[key], dict):
                # Process nested fields
                for subkey, value in data[key].items():
                    rows.append({
                        "Category": key,
                        "Field": subkey,
                        "Value": value
                    })
            else:
                # Process non-nested fields
                rows.append({
                    "Category": "",
                    "Field": key,
                    "Value": data[key]
                })

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
        data: Dictionary of extracted values
        filename: Name of the JSON file
        
    Returns:
        tuple: (json_data, filename) or (None, None) on error
    """
    try:
        # Create a clean copy without sources
        clean_data = {}
        for key, value in data.items():
            if key != "sources":
                clean_data[key] = value

        # Convert to JSON
        json_str = json.dumps(clean_data, indent=4).encode("utf-8")
        return json_str, filename
    except Exception as e:
        st.error(f"Error generating JSON: {e}")
        return None, None


def get_excel_download(data, filename="insurance_data.xlsx"):
    """
    Generate an Excel file containing extracted insurance data
    
    Args:
        data: Dictionary of extracted values
        filename: Name of the Excel file
        
    Returns:
        tuple: (excel_data, filename) or (None, None) on error
    """
    try:
        # Create Excel file
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            # Create summary sheet with flattened data
            rows = []
            for key in data:
                if key == "sources":
                    continue
                    
                if isinstance(data[key], dict):
                    for subkey, value in data[key].items():
                        rows.append({
                            "Category": key,
                            "Field": subkey,
                            "Value": value
                        })
                else:
                    rows.append({
                        "Category": "",
                        "Field": key,
                        "Value": data[key]
                    })
            
            summary_df = pd.DataFrame(rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Create sheets for each category
            categories = [k for k in data.keys() if isinstance(data[k], dict) and k != "sources"]
            for category in categories:
                category_rows = []
                for field, value in data[category].items():
                    category_rows.append({
                        "Field": field,
                        "Value": value
                    })
                    
                category_df = pd.DataFrame(category_rows)
                # Excel sheet names limited to 31 characters
                sheet_name = category[:31]
                category_df.to_excel(writer, sheet_name=sheet_name, index=False)

        buffer.seek(0)
        return buffer, filename
    except Exception as e:
        st.error(f"Error generating Excel: {e}")
        return None, None