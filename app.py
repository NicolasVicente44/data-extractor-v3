import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import tempfile
import os
import PyPDF2
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import datetime
import hashlib
import io

# Set page title
st.set_page_config(page_title="Insurance Policy Data Extractor", layout="wide")

# Constants and Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(CACHE_DIR, exist_ok=True)
FEEDBACK_DB_PATH = os.path.join(CACHE_DIR, "feedback_db.pkl")

# Define consistent schema for all insurance policies
INSURANCE_SCHEMA = {
    "General Information": ["Company Name", "Policy Number"],
    "Benefit Summary": [
        "Benefit Amount",
        "Eligibility Period",
        "Definition of Salary",
        "Child Coverage",
        "Student Extension",
    ],
    "Life Insurance": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Reduction Schedule",
        "Termination Age",
    ],
    "Optional Life Insurance": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Termination Age",
        "Spouse Amount",
        "Child Amount",
    ],
    "STD": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Waiting Period",
        "Maximum Benefit Period",
        "Termination Age",
    ],
    "LTD": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Waiting Period",
        "Definition of Disability",
        "Maximum Benefit Period",
        "COLA",
        "Termination Age",
    ],
    "Critical Illness": [
        "Benefit Amount",
        "Covered Conditions",
        "Multi-Occurrence",
        "Dependent Coverage",
        "Termination Age",
    ],
    "Health Coverage": [
        "Preferred Provider",
        "Deductibles",
        "Drug Coverage",
        "Preventative Services",
        "Out-of-Province Coverage",
        "Hospital Care",
        "Termination Age",
    ],
    "Dental Benefits": [
        "Dental Coverage",
        "Employee Assistance Plan",
        "Virtual Health Care",
        "Termination Age",
    ],
}

# Value format specifications
VALUE_FORMAT_SPECS = {
    "percentage": {
        "pattern": r"(\d+(?:\.\d+)?)%",
        "format": "{0}% of Salary",
        "validation": lambda x: 0 <= float(x) <= 100,
    },
    "multiplier": {
        "pattern": r"(\d+(?:\.\d+)?)x",
        "format": "{0}x Annual Salary",
        "validation": lambda x: 0 <= float(x) <= 10,
    },
    "currency": {
        "pattern": r"\$?([\d,]+(?:\.\d+)?)",
        "format": "${0:,.2f}",
        "validation": lambda x: float(x.replace(",", "")) >= 0,
    },
    "age": {
        "pattern": r"(?:age|to age)?\s*(\d{2,3})",
        "format": "Age {0}",
        "validation": lambda x: 0 <= int(x) <= 100,
    },
    "days": {
        "pattern": r"(\d+)\s*days?",
        "format": "{0} days",
        "validation": lambda x: 0 <= int(x) <= 365,
    },
    "weeks": {
        "pattern": r"(\d+)\s*weeks?",
        "format": "{0} weeks",
        "validation": lambda x: 0 <= int(x) <= 52,
    },
    "months": {
        "pattern": r"(\d+)\s*months?",
        "format": "{0} months",
        "validation": lambda x: 0 <= int(x) <= 48,
    },
    "years": {
        "pattern": r"(\d+)\s*years?",
        "format": "{0} years",
        "validation": lambda x: 0 <= int(x) <= 20,
    },
    "to_age": {
        "pattern": r"to\s*age\s*(\d{2,3})",
        "format": "To Age {0}",
        "validation": lambda x: 0 <= int(x) <= 100,
    },
}


# Load feedback database
def load_feedback_db():
    if os.path.exists(FEEDBACK_DB_PATH):
        try:
            with open(FEEDBACK_DB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load feedback database: {e}")
    return {}


# Save feedback to database
def save_feedback(document_hash, original_values, corrected_values):
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
        with open(FEEDBACK_DB_PATH, "wb") as f:
            pickle.dump(feedback_db, f)
        return True
    except Exception as e:
        st.error(f"Could not save feedback: {e}")
        return False


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
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


# Document structure analysis (simplified, without carrier detection)
def analyze_document_structure(text, page_texts):
    structure_info = {
        "document_type": "unknown",
        "format_type": "unknown",
        "section_markers": [],
        "table_regions": [],
        "field_locations": {},
    }

    # Try to identify document type
    doc_type_patterns = [
        (
            "booklet",
            [r"\bbooklet\b", r"\bemployee\s+booklet\b", r"\bbenefit\s+booklet\b"],
        ),
        (
            "summary",
            [
                r"\bsummary\b",
                r"\bbenefits?\s+summary\b",
                r"\bsummary\s+of\s+benefits\b",
            ],
        ),
        ("certificate", [r"\bcertificate\b", r"\binsurance\s+certificate\b"]),
        ("policy", [r"\bpolicy\b", r"\binsurance\s+policy\b", r"\bgroup\s+policy\b"]),
    ]

    for doc_type, patterns in doc_type_patterns:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure_info["document_type"] = doc_type
                break
        if structure_info["document_type"] != "unknown":
            break

    # Identify format type
    if len(re.findall(r"[\.]{4,}", text)) > 5:
        structure_info["format_type"] = "dot_leader"
    elif len(re.findall(r"[^\n]+\n[^\n]+", text)) > len(text) / 300:
        structure_info["format_type"] = "line_break"
    elif len(re.findall(r":\s+", text)) > len(text) / 400:
        structure_info["format_type"] = "colon_label"
    elif len(re.findall(r"\|\s+\|", text)) > 5:
        structure_info["format_type"] = "table_based"

    # Detect section markers
    section_patterns = [
        (r"([A-Z][A-Z\s]{5,30}[A-Z])", "uppercase_title"),
        (r"(?:^|\n)((?:[A-Z][a-z]+\s){1,4}[A-Z][a-z]+)(?:\n|:)", "title_case"),
        (r"(?:^|\n)(\d+\.\s+[A-Z][a-zA-Z\s]{5,30})(?:\n|:)", "numbered_section"),
    ]

    for pattern, section_type in section_patterns:
        for match in re.finditer(pattern, text):
            section_name = match.group(1).strip()
            if len(section_name.split()) >= 2 or section_type == "uppercase_title":
                structure_info["section_markers"].append(
                    {
                        "text": section_name,
                        "type": section_type,
                        "position": match.start(),
                    }
                )

    # Sort section markers by position
    structure_info["section_markers"].sort(key=lambda x: x["position"])

    # Find table regions
    table_patterns = [
        r"(?:\|\s+){2,}",
        r"(?:[-]+\+[-]+){2,}",
        r"(?:[^\n]+\n){2,}(?:\s{2,}[^\s][^\n]+\n){2,}",
    ]

    for pattern in table_patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)

            table_start = match.start()
            table_end = match.end()

            structure_info["table_regions"].append(
                {
                    "start": table_start,
                    "end": table_end,
                    "context": text[table_start:table_end],
                }
            )

    # Find common insurance field locations
    common_field_patterns = {
        "Life Insurance,Benefit Amount": [
            r"basic\s+life[^.]*?(?:amount|benefit|coverage)[^.]*?(\d)",
            r"life\s+insurance[^.]*?(?:amount|benefit|coverage)[^.]*?(\d)",
        ],
        "STD,Benefit Amount": [
            r"(?:short\s*term|std)[^.]*?(?:benefit|percentage|amount)[^.]*?(\d{1,3}%)",
            r"(?:short\s*term|std)[^.]*?(?:pays|provides)[^.]*?(\d{1,3}%)",
        ],
        "LTD,Benefit Amount": [
            r"(?:long\s*term|ltd)[^.]*?(?:benefit|percentage|amount)[^.]*?(\d{1,3}%)",
            r"(?:long\s*term|ltd)[^.]*?(?:pays|provides)[^.]*?(\d{1,3}%)",
        ],
        "Life Insurance,Termination Age": [
            r"(?:life|coverage)[^.]*?(?:terminates|ceases)[^.]*?(?:age|at)\s*(\d{2})",
        ],
        "STD,Waiting Period": [
            r"(?:short\s*term|std)[^.]*?(?:waiting|elimination)[^.]*?(?:period|days)[^.]*?(\d{1,3})",
        ],
        "LTD,Waiting Period": [
            r"(?:long\s*term|ltd)[^.]*?(?:waiting|elimination)[^.]*?(?:period|days)[^.]*?(\d{1,3})",
        ],
    }

    for field_key, patterns in common_field_patterns.items():
        category, field = field_key.split(",")
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if field_key not in structure_info["field_locations"]:
                    structure_info["field_locations"][field_key] = []

                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)

                structure_info["field_locations"][field_key].append(
                    {"position": match.start(), "context": text[start:end]}
                )

    return structure_info


# Smart text chunking
def smart_chunk_text(text, page_texts, structure_info, chunk_size=300, overlap=100):
    chunks = []
    chunk_sources = []
    chunk_metadata = []

    # Define natural text boundaries
    boundary_patterns = [r"\n\s*\n", r"\.\s+[A-Z]", r"[.!?]\s+"]

    # Use section markers for chunking if available
    if structure_info["section_markers"]:
        boundaries = [
            (marker["position"], marker["text"])
            for marker in structure_info["section_markers"]
        ]

        for i in range(len(boundaries)):
            start_pos, section_name = boundaries[i]

            end_pos = len(text)
            if i < len(boundaries) - 1:
                end_pos = boundaries[i + 1][0]

            section_text = text[start_pos:end_pos]

            if len(section_text) < 1500:
                chunks.append(section_text)
                chunk_sources.append(f"Section: {section_name}")
                chunk_metadata.append(
                    {"type": "section", "name": section_name, "position": start_pos}
                )
            else:
                current_pos = 0
                while current_pos < len(section_text):
                    target_pos = current_pos + chunk_size
                    if target_pos >= len(section_text):
                        chunk = section_text[current_pos:]
                        if len(chunk) > 50:
                            chunks.append(chunk)
                            chunk_sources.append(f"Section: {section_name}")
                            chunk_metadata.append(
                                {
                                    "type": "section_part",
                                    "name": section_name,
                                    "position": start_pos + current_pos,
                                }
                            )
                        break

                    best_boundary = target_pos
                    boundary_found = False

                    search_start = max(0, target_pos - overlap)
                    search_end = min(len(section_text), target_pos + overlap)
                    search_text = section_text[search_start:search_end]

                    for pattern in boundary_patterns:
                        matches = list(re.finditer(pattern, search_text))
                        if matches:
                            closest_match = min(
                                matches,
                                key=lambda m: abs(search_start + m.end() - target_pos),
                            )
                            best_boundary = search_start + closest_match.end()
                            boundary_found = True
                            break

                    chunk = section_text[current_pos:best_boundary]
                    if len(chunk) > 50:
                        chunks.append(chunk)
                        chunk_sources.append(f"Section: {section_name}")
                        chunk_metadata.append(
                            {
                                "type": "section_part",
                                "name": section_name,
                                "position": start_pos + current_pos,
                            }
                        )

                    if boundary_found:
                        current_pos = best_boundary
                    else:
                        current_pos += chunk_size - overlap
    else:
        # Use page boundaries if no sections found
        for page_idx, page_text in enumerate(page_texts):
            if not page_text or len(page_text.strip()) < 50:
                continue

            if len(page_text) < 2000:
                chunks.append(page_text)
                chunk_sources.append(f"Page {page_idx+1}")
                chunk_metadata.append(
                    {"type": "page", "page": page_idx + 1, "position": -1}
                )
            else:
                current_pos = 0
                while current_pos < len(page_text):
                    target_pos = current_pos + chunk_size
                    if target_pos >= len(page_text):
                        chunk = page_text[current_pos:]
                        if len(chunk) > 50:
                            chunks.append(chunk)
                            chunk_sources.append(f"Page {page_idx+1}")
                            chunk_metadata.append(
                                {
                                    "type": "page_part",
                                    "page": page_idx + 1,
                                    "position": -1,
                                }
                            )
                        break

                    best_boundary = target_pos
                    boundary_found = False

                    search_start = max(0, target_pos - overlap)
                    search_end = min(len(page_text), target_pos + overlap)
                    search_text = page_text[search_start:search_end]

                    for pattern in boundary_patterns:
                        matches = list(re.finditer(pattern, search_text))
                        if matches:
                            closest_match = min(
                                matches,
                                key=lambda m: abs(search_start + m.end() - target_pos),
                            )
                            best_boundary = search_start + closest_match.end()
                            boundary_found = True
                            break

                    chunk = page_text[current_pos:best_boundary]
                    if len(chunk) > 50:
                        chunks.append(chunk)
                        chunk_sources.append(f"Page {page_idx+1}")
                        chunk_metadata.append(
                            {"type": "page_part", "page": page_idx + 1, "position": -1}
                        )

                    if boundary_found:
                        current_pos = best_boundary
                    else:
                        current_pos += chunk_size - overlap

    # Add table regions as chunks
    for table_region in structure_info["table_regions"]:
        table_text = text[table_region["start"] : table_region["end"]]
        if len(table_text) > 50:
            chunks.append(table_text)
            chunk_sources.append("Table")
            chunk_metadata.append({"type": "table", "position": table_region["start"]})

    # Add specific field location chunks
    for field_key, locations in structure_info["field_locations"].items():
        for location in locations:
            field_chunk = location["context"]
            if len(field_chunk) > 50:
                chunks.append(field_chunk)
                chunk_sources.append(f"Field: {field_key}")
                chunk_metadata.append(
                    {
                        "type": "field_specific",
                        "field": field_key,
                        "position": location["position"],
                    }
                )

    return chunks, chunk_sources, chunk_metadata


# Create TF-IDF embeddings
def create_embeddings(chunks, queries):
    try:
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            max_features=15000,
            min_df=2,
            max_df=0.85,
        )

        all_texts = chunks + queries
        tfidf_vectorizer.fit(all_texts)

        chunk_vectors = tfidf_vectorizer.transform(chunks)
        query_vectors = tfidf_vectorizer.transform(queries)

        return chunk_vectors.toarray(), query_vectors.toarray(), tfidf_vectorizer
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None, None, None


# Enhanced semantic search (simplified, no carrier-specific boosts)
def semantic_search(
    chunk_vectors, query_vectors, chunks, chunk_metadata, category, field, top_k=5
):
    results = []

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vectors, chunk_vectors)

    # Setup content-based boost scores
    boost_scores = np.zeros(len(chunks))

    # Define chunk type boosts
    type_boosts = {
        "section": 0.15,
        "section_part": 0.1,
        "table": 0.25,
        "field_specific": 0.4,
        "page": 0.05,
        "page_part": 0.0,
        "fallback": 0.0,
    }

    # Apply metadata-based boosts
    for i, metadata in enumerate(chunk_metadata):
        chunk_type = metadata["type"]
        if chunk_type in type_boosts:
            boost_scores[i] += type_boosts[chunk_type]

        if chunk_type in ["section", "section_part"] and "name" in metadata:
            section_name = metadata["name"].lower()

            if category.lower() in section_name:
                boost_scores[i] += 0.25
            elif any(word in section_name for word in category.lower().split()):
                boost_scores[i] += 0.15

            # Field-specific section boosts
            if field == "Benefit Amount" and any(
                term in section_name for term in ["benefit", "amount", "coverage"]
            ):
                boost_scores[i] += 0.2
            elif field == "Waiting Period" and any(
                term in section_name for term in ["waiting", "elimination", "period"]
            ):
                boost_scores[i] += 0.2
            elif "termination" in field.lower() and any(
                term in section_name for term in ["termination", "ceases"]
            ):
                boost_scores[i] += 0.2

        if chunk_type == "field_specific" and "field" in metadata:
            target_field = f"{category},{field}"
            if metadata["field"] == target_field:
                boost_scores[i] += 0.5

        if chunk_type == "table":
            table_field_boost = {
                "Benefit Amount": 0.3,
                "Waiting Period": 0.25,
                "Maximum Benefit Period": 0.25,
                "Termination Age": 0.3,
                "Reduction Schedule": 0.2,
            }

            if field in table_field_boost:
                boost_scores[i] += table_field_boost[field]

    # Apply the boosts
    for i in range(len(similarities)):
        combined_scores = similarities[i] * 0.7 + boost_scores * 0.3

        top_indices = combined_scores.argsort()[-top_k:][::-1]

        top_chunks = [(chunks[idx], similarities[i][idx], idx) for idx in top_indices]

        results.append(top_chunks)

    return results


# Standardized field type detection
def detect_field_type(value, field):
    lower_value = value.lower()

    if "%" in lower_value:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower_value)
        if match:
            return "percentage", match.group(1)

    if "x" in lower_value or "times" in lower_value:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)", lower_value)
        if match:
            return "multiplier", match.group(1)

    if "$" in lower_value or re.search(r"\d+,\d{3}", lower_value):
        match = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", lower_value)
        if match:
            return "currency", match.group(1).replace(",", "")

    if "age" in lower_value or field == "Termination Age":
        match = re.search(r"(?:age|to age)?\s*(\d{2,3})", lower_value)
        if match:
            return "age", match.group(1)

    if "day" in lower_value:
        match = re.search(r"(\d+)\s*days?", lower_value)
        if match:
            return "days", match.group(1)

    if "week" in lower_value:
        match = re.search(r"(\d+)\s*weeks?", lower_value)
        if match:
            return "weeks", match.group(1)

    if "month" in lower_value:
        match = re.search(r"(\d+)\s*months?", lower_value)
        if match:
            return "months", match.group(1)

    if "year" in lower_value:
        match = re.search(r"(\d+)\s*years?", lower_value)
        if match:
            return "years", match.group(1)

    match = re.search(r"(\d+(?:\.\d+)?)", lower_value)
    if match:
        return "numeric", match.group(1)

    return "text", value.strip()


# Uniform value formatting function
def format_value(value, field, category):
    if value == "Not Found":
        return value

    field_type, extracted_value = detect_field_type(value, field)

    if field_type in VALUE_FORMAT_SPECS:
        format_spec = VALUE_FORMAT_SPECS[field_type]
        try:
            if "validation" in format_spec and format_spec["validation"](
                extracted_value
            ):
                return format_spec["format"].format(extracted_value)
        except:
            pass

    if field == "Benefit Amount":
        if field_type == "percentage":
            return f"{extracted_value}% of Salary"
        elif field_type == "multiplier":
            return f"{extracted_value}x Annual Salary"
        elif field_type == "currency":
            try:
                amount = float(extracted_value)
                return f"${amount:,.2f}"
            except:
                return value

    elif field == "Waiting Period":
        if field_type in ["days", "weeks", "months"]:
            unit = field_type
            return f"{extracted_value} {unit}"

    elif field == "Termination Age":
        if field_type == "age":
            return f"Age {extracted_value}"
        elif field_type == "to_age":
            return f"To Age {extracted_value}"

    elif field == "Maximum Benefit Period":
        if field_type in ["weeks", "months", "years"]:
            unit = field_type
            return f"{extracted_value} {unit}"
        elif field_type == "to_age":
            return f"To Age {extracted_value}"

    return value


# Basic field validation based on field type
def is_valid_value(value, field, category=None):
    try:
        field_type, extracted_value = detect_field_type(value, field)

        if (
            field_type in VALUE_FORMAT_SPECS
            and "validation" in VALUE_FORMAT_SPECS[field_type]
        ):
            try:
                return VALUE_FORMAT_SPECS[field_type]["validation"](extracted_value)
            except:
                pass

        if field == "Benefit Amount":
            if "%" in value:
                percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", value, re.IGNORECASE)
                if percent_match:
                    percentage = float(percent_match.group(1))
                    return percentage <= 100

            elif "x" in value.lower() or "times" in value.lower():
                multiplier_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:x|times)", value.lower()
                )
                if multiplier_match:
                    multiplier = float(multiplier_match.group(1))
                    if category == "Life Insurance":
                        return 0.5 <= multiplier <= 5
                    else:
                        return multiplier <= 10

        elif field == "Termination Age":
            age_match = re.search(r"(?:Age|age)\s*(\d{2,3})", value)
            if age_match:
                age = int(age_match.group(1))
                if 55 <= age <= 100:
                    return True
                return False

        elif field == "Waiting Period":
            if category == "STD":
                day_match = re.search(r"(\d+)\s*days?", value, re.IGNORECASE)
                if day_match:
                    days = int(day_match.group(1))
                    return 0 <= days <= 180

            elif category == "LTD":
                day_match = re.search(r"(\d+)\s*days?", value, re.IGNORECASE)
                if day_match:
                    days = int(day_match.group(1))
                    return 30 <= days <= 365

                month_match = re.search(r"(\d+)\s*months?", value, re.IGNORECASE)
                if month_match:
                    months = int(month_match.group(1))
                    return 1 <= months <= 12

        return True
    except Exception:
        return True


# Extract field values with more general patterns
def extract_values(chunks_with_scores, category, field):

    base_patterns = {
        "dollar_amount": r"\$\s*([\d,]+(?:\.\d+)?)",
        "percentage": r"(\d{1,3}(?:\.\d{1,2})?)%(?:\s*of\s*(?:your)?(?:\s*(?:monthly|annual))?(?:\s*(?:earnings|salary|income)))?",
        "multiplier": r"(\d{1,2}(?:\.\d{1,2})?)(?:\s*|\-)(x|times)(?:\s*(?:your)?(?:\s*annual)?(?:\s*(?:earnings|salary|income)))?",
        "age": r"(?:age|at)\s*(\d{2,3})",
        "days": r"(\d{1,3})\s*(?:calendar|consecutive)?\s*(?:day|calendar day)s?",
        "weeks": r"(\d{1,3})\s*weeks?",
        "months": r"(\d{1,3})\s*months?",
        "years": r"(\d{1,3})\s*years?",
        "text_value": r"(?:is|are|:)\s*([A-Za-z0-9\s,]+)(?:\.|\n|$)",
    }

    # Generic field patterns
    field_patterns = {
        "Benefit Amount": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": base_patterns["multiplier"],
                "priority": 3,
                "format": "multiplier",
            },
            {
                "pattern": base_patterns["dollar_amount"],
                "priority": 2,
                "format": "currency",
            },
            {
                "pattern": r"benefit(?:\s*amount)?[\s:]*(\d{1,3}(?:\.\d{1,2})?)%",
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(\d{1,3})%\s*of\s*(?:your|the)?\s*(?:monthly|annual)?\s*(?:salary|earnings|income)",
                "priority": 4,
                "format": "percentage",
            },
            {
                "pattern": r"coverage(?:\s*amount)?[\s:]*(\d{1,2})\s*(?:x|times)",
                "priority": 3,
                "format": "multiplier",
            },
        ],
        "Termination Age": [
            {"pattern": base_patterns["age"], "priority": 3, "format": "age"},
            {
                "pattern": r"(?:terminates|ceases|ends)[\s:]*(?:at)?\s*age\s*(\d{2,3})",
                "priority": 4,
                "format": "age",
            },
            {
                "pattern": r"(?:to|until)\s*age\s*(\d{2,3})",
                "priority": 3,
                "format": "age",
            },
            {"pattern": r"age\s*(\d{2,3})", "priority": 2, "format": "age"},
        ],
        "Waiting Period": [
            {"pattern": base_patterns["days"], "priority": 4, "format": "days"},
            {"pattern": base_patterns["weeks"], "priority": 3, "format": "weeks"},
            {"pattern": base_patterns["months"], "priority": 3, "format": "months"},
            {
                "pattern": r"(?:waiting|elimination)\s*period[\s:]*(\d{1,3})\s*(?:calendar|consecutive|business)?\s*(?:day|week|month)s?",
                "priority": 4,
                "format": "auto",
            },
        ],
        "Maximum Benefit Period": [
            {"pattern": base_patterns["weeks"], "priority": 3, "format": "weeks"},
            {"pattern": base_patterns["months"], "priority": 3, "format": "months"},
            {"pattern": base_patterns["years"], "priority": 3, "format": "years"},
            {"pattern": r"to\s*age\s*(\d{2,3})", "priority": 4, "format": "to_age"},
            {
                "pattern": r"(?:maximum|benefit)\s*period[\s:]*(\d{1,3})\s*(?:week|month|year)s?",
                "priority": 4,
                "format": "auto",
            },
        ],
        "Definition of Disability": [
            {
                "pattern": r"own\s*occupation[^.]*?(?:for|period|of)?\s*(\d{1,3})\s*(?:month|year)s?",
                "priority": 4,
                "format": "own_occupation",
            },
            {
                "pattern": r"first\s*(\d{1,3})\s*(?:month|year)s?[^.]*?own\s*occupation",
                "priority": 4,
                "format": "own_occupation",
            },
            {
                "pattern": r"(?:definition|disabled|disability)[^.]*?((?:own|regular|any)\s*occupation)",
                "priority": 3,
                "format": "text",
            },
        ],
        "Covered Conditions": [
            {
                "pattern": r"(?:covered\s*conditions|covered\s*illnesses)[^:]*:([^.]+)",
                "priority": 4,
                "format": "conditions",
            },
            {
                "pattern": r"(?:conditions|illnesses)\s*(?:covered|included)(?:\s*are)?:([^.]+)",
                "priority": 3,
                "format": "conditions",
            },
        ],
        "Company Name": [
             {"pattern": r'(?:insurer|insurance\s*company|carrier)[^:]*:\s*([A-Z][A-Za-z\s&.,\']+)', "priority": 4, "format": "text"},
    {"pattern": r'(?:underwritten|issued)\s*by[^:]*:\s*([A-Z][A-Za-z\s&.,\']+)', "priority": 3, "format": "text"},
    # Add these new patterns
    {"pattern": r'([A-Z][A-Za-z\s\.,\']+(?:Inc\.|LLC|Ltd\.|Corporation|Company|Co\.))', "priority": 4, "format": "text"},
    {"pattern": r'([A-Z][A-Za-z\s]+(?:Insurance|Life|Financial)(?:\s[A-Za-z\s]+)?)', "priority": 4, "format": "text"},
    {"pattern": r'(?:your plan sponsor|sponsored by)\s+([A-Z][A-Za-z\s\.,\']+)', "priority": 5, "format": "text"},
    {"pattern": r'^([A-Z][A-Za-z\s\.,\']+(?:Inc\.|LLC|Ltd\.))\s+(?:is|has|Plan)', "priority": 5, "format": "text"},
    {"pattern": r'(?:from|by|with)\s+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})', "priority": 3, "format": "text"},
    {"pattern": r'selected\s+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})', "priority": 5, "format": "text"},
        ],
        "Policy Number": [
       {"pattern": r'policy\s*(?:number|no|#)[\s:]+([A-Z0-9][\w\-\.\/]{3,20})', "priority": 4, "format": "text"},
    {"pattern": r'certificate\s*(?:number|no|#)[\s:]+([A-Z0-9][\w\-\.\/]{3,20})', "priority": 3, "format": "text"},
    # Add these new patterns
    {"pattern": r'Plan\s*Number:?\s*([A-Z0-9][\w\-\.\/]{3,20})', "priority": 5, "format": "text"},
    {"pattern": r'(?:Group|Contract|Plan|Policy)\s*(?:Number|No\.|#)?\s*:?\s*([A-Z0-9][\w\-\.\/]{3,20})', "priority": 4, "format": "text"},
    {"pattern": r'(?:Group|Contract|Plan|Policy)\s*(?:Number|No\.|#)?\s*:?\s*([0-9]{5,7})', "priority": 4, "format": "text"},
    {"pattern": r'([G][0-9]{5,10})', "priority": 3, "format": "text"},
    {"pattern": r'(?:number|no|#)\s*([0-9]{5,7})', "priority": 3, "format": "text"},
        ],
        "Deductibles": [
            {
                "pattern": base_patterns["dollar_amount"],
                "priority": 3,
                "format": "currency",
            },
            {
                "pattern": r"deductible[\s:]*\$?\s*([\d,]+)",
                "priority": 4,
                "format": "currency",
            },
        ],
        "Drug Coverage": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(?:drug|prescription|medication)[^.]*?(?:covered|reimbursed|paid)[^.]*?(\d{1,3})%",
                "priority": 4,
                "format": "percentage",
            },
        ],
        "Dental Coverage": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(?:dental|basic\s*dental)[^.]*?(?:covered|reimbursed|paid)[^.]*?(\d{1,3})%",
                "priority": 4,
                "format": "percentage",
            },
        ],
    }

    # Default patterns for fields without specific patterns
    default_patterns = [
        {
            "pattern": base_patterns["dollar_amount"],
            "priority": 2,
            "format": "currency",
        },
        {"pattern": base_patterns["percentage"], "priority": 2, "format": "percentage"},
        {
            "pattern": r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            "priority": 1,
            "format": "numeric",
        },
        {"pattern": base_patterns["text_value"], "priority": 1, "format": "text"},
    ]

    # Get the appropriate patterns for this field
    field_specific_patterns = field_patterns.get(field, default_patterns)

    # Universal formatting function based on pattern type
    def format_value_from_pattern(value, format_type, match_value):
        try:
            if format_type == "percentage":
                percentage_match = re.search(r"(\d+(?:\.\d+)?)", match_value)
                if percentage_match:
                    percentage = percentage_match.group(1)
                    return f"{percentage}% of Salary"
                return (
                    f"{match_value}% of Salary"
                    if not "%" in match_value
                    else match_value
                )

            elif format_type == "multiplier":
                multiplier_match = re.search(r"(\d+(?:\.\d+)?)", match_value)
                if multiplier_match:
                    multiplier = multiplier_match.group(1)
                    return f"{multiplier}x Annual Salary"
                return (
                    f"{match_value}x Annual Salary"
                    if not "x" in match_value.lower()
                    else match_value
                )

            elif format_type == "currency":
                amount_match = re.search(r"([\d,]+(?:\.\d+)?)", match_value)
                if amount_match:
                    amount_str = amount_match.group(1).replace(",", "")
                    try:
                        amount = float(amount_str)
                        return f"${amount:,.2f}"
                    except:
                        pass
                return f"${match_value}" if not "$" in match_value else match_value

            elif format_type == "age":
                age_match = re.search(r"(\d{2,3})", match_value)
                if age_match:
                    age = age_match.group(1)
                    return f"Age {age}"
                return match_value

            elif format_type == "days":
                return f"{match_value} days"

            elif format_type == "weeks":
                return f"{match_value} weeks"

            elif format_type == "months":
                return f"{match_value} months"

            elif format_type == "years":
                return f"{match_value} years"

            elif format_type == "auto":
                if "day" in value.lower():
                    return f"{match_value} days"
                elif "week" in value.lower():
                    return f"{match_value} weeks"
                elif "month" in value.lower():
                    return f"{match_value} months"
                elif "year" in value.lower():
                    return f"{match_value} years"
                return match_value

            elif format_type == "to_age":
                return f"To Age {match_value}"

            elif format_type == "own_occupation":
                time_unit = "months"
                if "year" in value.lower():
                    time_unit = "years"
                return f"Own Occupation for {match_value} {time_unit}"

            elif format_type == "conditions":
                conditions = match_value.strip()
                conditions = re.sub(r"\s+", " ", conditions)
                return conditions

            return match_value
        except:
            return value

    results = []

    # Context boost factors
    context_boost = {
        "field_in_line": 2.0,
        "category_in_context": 1.5,
        "table_header": 1.8,
    }

    # Process each chunk
    for chunk_data in chunks_with_scores:
        chunk, similarity_score, chunk_idx = chunk_data
        chunk_lower = chunk.lower()

        boost = 1.0
        field_lower = field.lower()
        category_lower = category.lower()

        # Check if field name appears in same line as potential values
        for line in chunk.split("\n"):
            line_lower = line.lower()
            if field_lower in line_lower:
                boost *= context_boost["field_in_line"]
                break

        # Check if category appears in context
        if category_lower in chunk_lower:
            boost *= context_boost["category_in_context"]

        # Check for table structure with this field/category
        table_header_pattern = (
            r"(?:"
            + re.escape(field_lower)
            + r"|"
            + re.escape(category_lower)
            + r")\s*(?:\||\t|:)"
        )
        if re.search(table_header_pattern, chunk_lower):
            boost *= context_boost["table_header"]

        # Try each pattern for this field
        for pattern_info in field_specific_patterns:
            pattern = pattern_info["pattern"]
            priority = pattern_info["priority"]
            format_type = pattern_info["format"]

            try:
                # Find all matches in the chunk
                matches = re.findall(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    # If match is a tuple (from capturing groups), take first non-empty group
                    if isinstance(match, tuple):
                        match = next((m for m in match if m), "")

                    # Clean up the match
                    match_value = match.strip()
                    if match_value:
                        # Skip very short values unless they're numbers
                        if len(match_value) < 2 and not match_value.isdigit():
                            continue

                        # Format the value based on the pattern type
                        original_value = match_value
                        formatted_value = format_value_from_pattern(
                            chunk, format_type, match_value
                        )

                        # Apply standard formatting for consistency
                        standardized_value = format_value(
                            formatted_value, field, category
                        )

                        # Calculate final score: similarity * boost * priority
                        final_score = similarity_score * boost * priority

                        # Validate the value
                        if is_valid_value(standardized_value, field, category):
                            results.append(
                                (standardized_value, final_score, original_value)
                            )
            except Exception:
                continue

    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates, keeping highest score
    unique_values = {}
    for formatted_value, score, original_value in results:
        if (
            formatted_value not in unique_values
            or score > unique_values[formatted_value]["score"]
        ):
            unique_values[formatted_value] = {
                "score": score,
                "original": original_value,
            }

    return [
        (value, data["score"], data["original"])
        for value, data in unique_values.items()
    ]


# Generate comprehensive field queries
def generate_field_queries():
    queries = {}

    # Generate standard queries for each field in schema
    for category, fields in INSURANCE_SCHEMA.items():
        queries[category] = {}
        for field in fields:
            # Generate multiple queries for each field with variants
            queries[category][field] = [
                f"{category} {field}",
                f"{field} {category}",
                f"{field}",
                f"{category} {field} insurance",
            ]

    # Add specific queries for common fields
    field_specific_queries = {
        "General Information": {
            "Company Name": [
                "insurance company name",
                "insurance carrier",
                "insurer name",
                "underwritten by",
                "provided by insurance company",
            ],
            "Policy Number": [
                "policy number",
                "certificate number",
                "group policy number",
                "contract number",
                "policy identification",
            ],
        },
        "Life Insurance": {
            "Benefit Amount": [
                "life insurance benefit amount",
                "basic life coverage amount",
                "life insurance coverage",
                "group life insurance benefit",
                "life benefit amount",
            ],
            "Termination Age": [
                "life insurance termination age",
                "coverage ceases at age",
                "life benefits terminate",
                "life insurance terminates",
                "life coverage ends",
            ],
        },
        "Short-Term Disability": {
            "Benefit Amount": [
                "short-term disability benefit amount",
                "STD benefit percentage",
                "weekly indemnity amount",
                "short term disability pays",
                "STD coverage amount",
            ],
            "Waiting Period": [
                "STD waiting period",
                "short-term disability elimination period",
                "days before STD begins",
                "elimination period STD",
                "when STD benefits begin",
            ],
        },
        "Long-Term Disability": {
            "Benefit Amount": [
                "long-term disability benefit amount",
                "LTD benefit percentage",
                "monthly disability benefit",
                "LTD pays",
                "long term disability amount",
            ],
            "Waiting Period": [
                "LTD waiting period",
                "long-term disability elimination period",
                "days before LTD begins",
                "elimination period LTD",
                "when LTD benefits begin",
            ],
        },
    }

    # Merge field-specific queries into main queries
    for category, fields in field_specific_queries.items():
        if category in queries:
            for field, additional_queries in fields.items():
                if field in queries[category]:
                    queries[category][field].extend(additional_queries)

    return queries


# Basic cross-field validation for consistency
def apply_cross_field_validation(extracted_values):
    field_relations = [
        # If life is X times salary, optional life might be similar
        (
            "Life Insurance,Benefit Amount",
            "Optional Life Insurance,Benefit Amount",
            lambda v1, v2: (
                v1 if "Annual Salary" in v1 and v2 == "Not Found" and "x" in v1 else v2
            ),
        ),
        # STD and LTD termination ages are often the same
        (
            "STD,Termination Age",
            "LTD,Termination Age",
            lambda v1, v2: (
                v1
                if v1 != "Not Found"
                and v2 == "Not Found"
                and re.search(r"Age \d{2}", v1)
                else v2
            ),
        ),
        # LTD waiting period often matches STD maximum benefit period
        (
            "STD,Maximum Benefit Period",
            "LTD,Waiting Period",
            lambda v1, v2: (
                v1
                if v1 != "Not Found"
                and v2 == "Not Found"
                and any(unit in v1.lower() for unit in ["week", "day", "month"])
                else v2
            ),
        ),
        # Various termination ages are often the same
        (
            "Life Insurance,Termination Age",
            "Critical Illness,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        # Health and Dental often have the same termination age
        (
            "Health Coverage,Termination Age",
            "Dental Benefits,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        # Other common termination ages
        (
            "Life Insurance,Termination Age",
            "STD,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        (
            "Life Insurance,Termination Age",
            "LTD,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
    ]

    # Apply field relation rules
    for relation in field_relations:
        source_path, target_path, rule_func = relation

        source_category, source_field = source_path.split(",")

        if (
            source_category in extracted_values
            and source_field in extracted_values[source_category]
        ):
            source_value = extracted_values[source_category][source_field]

            if target_path:
                target_category, target_field = target_path.split(",")

                if (
                    target_category in extracted_values
                    and target_field in extracted_values[target_category]
                ):
                    target_value = extracted_values[target_category][target_field]

                    new_target_value = rule_func(source_value, target_value)
                    extracted_values[target_category][target_field] = new_target_value

    # Apply standard formatting to all fields
    for category in extracted_values:
        for field in extracted_values[category]:
            value = extracted_values[category][field]

            if value == "Not Found":
                continue

            standardized_value = format_value(value, field, category)
            extracted_values[category][field] = standardized_value

    return extracted_values


# Main extraction function
def extract_insurance_data(text, page_texts, document_hash):
    if not text or not page_texts:
        st.error("No text content found in the PDF")
        return None

    try:
        # Analyze document structure (no carrier detection)
        structure_info = analyze_document_structure(text, page_texts)

        # Display document type
        doc_type = (
            structure_info["document_type"].capitalize()
            if structure_info["document_type"] != "unknown"
            else "Insurance Document"
        )
        st.info(f"Processing {doc_type}")

        # Check for previous feedback
        feedback_db = load_feedback_db()
        if document_hash in feedback_db:
            st.info(
                f"This document has been processed before. Learning from {len(feedback_db[document_hash])} previous corrections."
            )

        # Intelligent text chunking
        chunks, chunk_sources, chunk_metadata = smart_chunk_text(
            text, page_texts, structure_info
        )

        # Generate search queries (unified approach, not carrier-specific)
        field_queries = generate_field_queries()

        # Flatten queries for vectorization
        flat_queries = []
        query_metadata = []

        for category, fields in field_queries.items():
            for field, queries in fields.items():
                for query in queries:
                    flat_queries.append(query)
                    query_metadata.append((category, field))

        # Create TF-IDF embeddings
        chunk_vectors, query_vectors, vectorizer = create_embeddings(
            chunks, flat_queries
        )

        if chunk_vectors is None or query_vectors is None:
            st.error("Failed to create embeddings")
            return None

        st.info("Using semantic vector search for field extraction")

        # Process search queries
        organized_results = defaultdict(lambda: defaultdict(list))

        # Process in batches
        batch_size = 20
        for i in range(0, len(flat_queries), batch_size):
            batch_end = min(i + batch_size, len(flat_queries))
            batch_query_vectors = query_vectors[i:batch_end]
            batch_metadata = query_metadata[i:batch_end]

            batch_results = []
            for j, (category, field) in enumerate(batch_metadata):
                if j < len(batch_query_vectors):
                    results = semantic_search(
                        chunk_vectors,
                        batch_query_vectors[j : j + 1],
                        chunks,
                        chunk_metadata,
                        category,
                        field,
                        top_k=5,
                    )
                    if results:
                        batch_results.extend(results)

            for j, results in enumerate(batch_results):
                if j < len(batch_metadata):
                    category, field = batch_metadata[j]
                    for chunk, score, idx in results:
                        if (chunk, score, idx) not in organized_results[category][
                            field
                        ]:
                            organized_results[category][field].append(
                                (chunk, score, idx)
                            )

        # Extract values
        extracted_values = {}

        # Ensure all categories and fields exist
        for category in INSURANCE_SCHEMA:
            extracted_values[category] = {}
            for field in INSURANCE_SCHEMA[category]:
                extracted_values[category][field] = "Not Found"

        # Extract where we have results
        for category in organized_results:
            for field in organized_results[category]:
                chunks_with_scores = organized_results[category][field]

                try:
                    # Use generalized extraction without carrier-specific patterns
                    values = extract_values(chunks_with_scores, category, field)

                    if values:
                        # Take the most likely value (highest similarity score)
                        best_value = values[0][0]
                        extracted_values[category][field] = best_value
                except Exception as e:
                    st.warning(f"Error extracting {field} in {category}: {e}")

        # Apply cross-field validation (without carrier-specific rules)
        extracted_values = apply_cross_field_validation(extracted_values)

        # Apply feedback learning
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
                            st.info(
                                f"Applied correction to {category} - {field} based on previous feedback"
                            )

        # Store document hash for feedback
        st.session_state.document_hash = document_hash
        st.session_state.structure_info = structure_info

        return extracted_values
    except Exception as e:
        st.error(f"Error in data extraction: {e}")
        import traceback

        st.error(traceback.format_exc())
        return None


# Download functions
def get_csv_download(data, filename="insurance_data.csv"):
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


# Main application
def main():
    st.title("Insurance Policy Data Extractor")
    st.write(
        "Upload any insurance policy PDF to extract standardized data using semantic search."
    )

    # Sidebar
    with st.sidebar:
        st.subheader("About")
        st.write("This app extracts standardized data from insurance policy PDFs.")
        st.write("Features:")
        st.write("- Semantic vector search")
        st.write("- Consistent output structure")
        st.write("- Standardized field formatting")
        st.write("- Unified extraction approach")

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

                st.info(
                    "All data will be extracted into a unified, standardized format using semantic search."
                )

                if st.button("Extract Policy Data", type="primary"):
                    with st.spinner("Extracting data with semantic search..."):
                        extracted_data = extract_insurance_data(
                            pdf_text, page_texts, document_hash
                        )

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

                    st.write(
                        f"Extraction: {found_fields}/{total_fields} fields ({confidence:.1f}%)"
                    )
                    st.write("Format: Standardized structure with consistent fields")

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

                        # Submit and feedback options
                        col1, col2 = st.columns(2)
                        with col1:
                            submitted = st.form_submit_button("Save Changes")
                        with col2:
                            collect_feedback = st.checkbox(
                                "Submit corrections as feedback for model to learn from", value=True
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
                                        st.session_state.document_hash,
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

                    # Data summary
                    st.subheader("Data Structure")

                    # Display in columns
                    col1, col2, col3 = st.columns(3)

                    categories = list(INSURANCE_SCHEMA.keys())
                    cats_per_col = len(categories) // 3 + (
                        1 if len(categories) % 3 > 0 else 0
                    )

                    for i, col in enumerate([col1, col2, col3]):
                        start_idx = i * cats_per_col
                        end_idx = min(start_idx + cats_per_col, len(categories))

                        with col:
                            for category in categories[start_idx:end_idx]:
                                st.write(f"**{category}**")
                                for field in INSURANCE_SCHEMA[category]:
                                    value = export_data[category][field]
                                    mark = "✓" if value != "Not Found" else ""
                                    st.write(f"- {field} {mark}")
            else:
                st.error("Failed to process PDF")

        st.info("Every uploaded PDF produces the same standardized data structure")

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

        st.expander("FAQ").write(
            """
        Q: How does this tool work without carrier detection?
        A: It uses semantic vector search to understand the meaning of policy text, rather than relying on carrier-specific patterns.
        
        Q: Why standardize the output format?
        A: Standardized output makes it easier to compare policies, analyze data, and integrate with other systems.
        
        Q: What if a field isn't found?
        A: The field will still appear in the output as "Not Found" to maintain the consistent structure.
        
        Q: How accurate is the extraction?
        A: Accuracy depends on the document quality and formatting, but the semantic approach works well across different policy types.
        """
        )

        st.expander("Export Formats").write(
            """
        CSV: Flattened view with Category, Field, Value columns
        JSON: Hierarchical structure with nested categories and fields
        Excel: Multiple sheets with summary and category-specific pages
        """
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        import traceback

        st.error(traceback.format_exc())

        # Create empty data structure for error case
        empty_data = {}
        for category in INSURANCE_SCHEMA:
            empty_data[category] = {}
            for field in INSURANCE_SCHEMA[category]:
                empty_data[category][field] = "Not Found"

        st.session_state.extracted_data = empty_data
        st.session_state.edited_data = empty_data
