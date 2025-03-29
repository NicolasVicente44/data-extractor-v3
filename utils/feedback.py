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


def apply_feedback(document_hash, extracted_values):
    """
    Apply previous feedback to improve extraction results
    
    Args:
        document_hash: Hash of the document
        extracted_values: Current extraction results
        
    Returns:
        dict: Updated extraction results
    """
    feedback_db = load_feedback_db()
    
    if document_hash in feedback_db and feedback_db[document_hash]:
        # Get the most recent feedback
        latest_feedback = feedback_db[document_hash][-1]
        original_values = latest_feedback["original_values"]
        corrected_values = latest_feedback["corrected_values"]
        
        # Apply corrections
        for key in corrected_values:
            if key not in extracted_values or key == "sources":
                continue
                
            if isinstance(corrected_values[key], dict):
                # Handle nested fields
                for subkey, value in corrected_values[key].items():
                    if subkey in extracted_values[key]:
                        # Only apply correction if the current value matches the original value that was corrected
                        if key in original_values and subkey in original_values[key] and extracted_values[key][subkey] == original_values[key][subkey]:
                            extracted_values[key][subkey] = value
            else:
                # Handle top-level fields
                if key in original_values and extracted_values[key] == original_values[key]:
                    extracted_values[key] = corrected_values[key]
    
    return extracted_values