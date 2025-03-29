# Insurance Policy Data Extractor

A Streamlit application that extracts standardized data from insurance policy PDFs using semantic vector search.

## Features

- Extract structured data from insurance policy documents
- Semantic search with TF-IDF vectorization
- Standardized field formatting
- Cross-field validation
- User feedback and continuous learning
- Export to CSV, JSON, and Excel formats

## Project Structure

```
insurance_extractor/
│
├── app.py                       # Main Streamlit application 
├── config.py                    # Configuration constants and schema
│
├── extractors/
│   ├── __init__.py
│   ├── pdf_extractor.py         # PDF text extraction logic
│   ├── semantic_extractor.py    # Semantic search and field extraction
│   └── value_extractor.py       # Field value extraction and formatting
│
├── models/
│   ├── __init__.py
│   ├── document.py              # Document structure models 
│   ├── embeddings.py            # TF-IDF and vector operations
│   └── schema.py                # Insurance data schema definition (optional)
│
├── utils/
│   ├── __init__.py
│   ├── chunking.py              # Text chunking utilities
│   ├── feedback.py              # Feedback collection utilities
│   ├── formatting.py            # Value formatting utilities 
│   ├── validation.py            # Field validation utilities
│   └── export.py                # Export functions (CSV, JSON, Excel)
│
└── data/                        # Directory for feedback database
    └── feedback_db.pkl          # Feedback database
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## How to Use

1. Upload an insurance policy PDF
2. Wait for the semantic extraction process to complete
3. Review and edit the extracted data if needed
4. Download the data in your preferred format (CSV, JSON, or Excel)

## Background

This tool uses semantic vector search to understand insurance policy text and extract key information in a standardized format. Instead of relying on exact pattern matching, it uses TF-IDF vectorization to understand the semantic meaning of text chunks and match them to field queries.

Key components:
- Document structure analysis
- Smart text chunking
- TF-IDF embeddings and similarity search
- Field value extraction with regex patterns
- Cross-field validation
- Feedback-based continuous learning

The result is a standardized data structure that works across different insurance carriers and policy formats.