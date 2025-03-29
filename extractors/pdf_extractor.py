import tempfile
import os
import PyPDF2
import hashlib
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file and calculate its hash for identification
    
    Args:
        pdf_file: Streamlit uploaded PDF file
        
    Returns:
        tuple: (full_text, page_texts, file_hash)
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name

        # Calculate document hash for feedback lookup
        with open(temp_path, "rb") as file:
            file_hash = hashlib.md5(file.read()).hexdigest()

        with open(temp_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            page_texts = []

            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text:  # Only add non-empty pages
                        page_texts.append(page_text)
                        full_text += page_text + "\n\n"
                except Exception as e:
                    st.warning(
                        f"Warning: Could not extract text from page {page_num+1}: {e}"
                    )

        os.unlink(temp_path)
        return full_text, page_texts, file_hash
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None, None, None