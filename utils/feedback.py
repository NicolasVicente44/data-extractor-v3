import os
import datetime
import pickle
import streamlit as st
from config import FEEDBACK_DB_PATH

def load_feedback_db():
    """
    Load the feedback database from disk
    
    Returns:
        dict: Dictionary of feedback data
    """
    if os.path.exists(FEEDBACK_DB_PATH):
        try:
            with open(FEEDBACK_DB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load feedback database: {e}")
    return {}


def save_feedback(document_hash, original_values, corrected_values):
    """
    Save user feedback for model improvement
    
    Args:
        document_hash: Hash of the document
        original_values: Original extracted values
        corrected_values: User-corrected values
        
    Returns:
        bool: True if feedback was saved successfully, False otherwise
    """
    feedback_db = load_feedback_db()

    if document_hash not in feedback_db:
        feedback_db[document_hash] = []

    feedback_db[document_hash].append(
        {
            "timestamp": datetime.datetime.now(),
            "original_values": original_values,
            "corrected_values": corrected_values,
        }
    )

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(FEEDBACK_DB_PATH), exist_ok=True)
        
        with open(FEEDBACK_DB_PATH, "wb") as f:
            pickle.dump(feedback_db, f)
        return True
    except Exception as e:
        st.error(f"Could not save feedback: {e}")
        return False


def apply_feedback_learning(extracted_values, document_hash):
    """
    Apply previous feedback to improve extraction results
    
    Args:
        extracted_values: Dictionary of extracted values
        document_hash: Hash of the current document
        
    Returns:
        dict: Updated extracted values
    """
    feedback_db = load_feedback_db()
    
    if document_hash in feedback_db:
        latest_feedback = feedback_db[document_hash][-1]
        original_values = latest_feedback["original_values"]
        corrected_values = latest_feedback["corrected_values"]

        for category in corrected_values:
            if category not in extracted_values:
                continue

            for field in corrected_values[category]:
                if field not in extracted_values[category]:
                    continue

                if (
                    category in original_values
                    and field in original_values[category]
                    and original_values[category][field]
                    != corrected_values[category][field]
                ):

                    if (
                        extracted_values[category][field]
                        == original_values[category][field]
                    ):
                        extracted_values[category][field] = corrected_values[
                            category
                        ][field]
    
    return extracted_values