import streamlit as st
from collections import defaultdict
import numpy as np

from models.document import DocumentStructure
from models.embeddings import EmbeddingsManager
from utils.chunking import smart_chunk_text
from utils.validation import apply_cross_field_validation
from utils.feedback import apply_feedback_learning
from extractors.value_extractor import extract_values, generate_field_queries
from config import INSURANCE_SCHEMA


class SemanticExtractor:
    """
    Main extractor class that orchestrates the document analysis and data extraction
    """
    
    def __init__(self, text, page_texts, document_hash):
        """
        Initialize the semantic extractor
        
        Args:
            text: Full document text
            page_texts: List of text content per page
            document_hash: Document hash for feedback identification
        """
        self.text = text
        self.page_texts = page_texts
        self.document_hash = document_hash
        self.doc_structure = DocumentStructure(text, page_texts)
        self.structure_info = self.doc_structure.structure_info

    
        
    def extract_data(self):
        """
        Extract insurance data from the document using semantic search
        
        Returns:
            dict: Extracted insurance data by category and field
        """
        # Intelligent text chunking
        chunks, chunk_sources, chunk_metadata = smart_chunk_text(
            self.text, self.page_texts, self.structure_info
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
        chunk_vectors, query_vectors, vectorizer = EmbeddingsManager.create_embeddings(
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
                    results = EmbeddingsManager.semantic_search(
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
        extracted_values = self._initialize_extracted_values()

        # Extract where we have results
        for category in organized_results:
            for field in organized_results[category]:
                chunks_with_scores = organized_results[category][field]

                try:
                    # Use generalized extraction without carrier-specific patterns
                    values = extract_values(chunks_with_scores, category, field)

                    if values:
                        # Take the most likely value (highest similarity score)
                        best_value, best_score, original_value, source_chunk = values[0]
                        extracted_values[category][field] = best_value
                        # Store the source chunk for this field
                        extracted_values["sources"][category][field] = source_chunk
                except Exception as e:
                    st.warning(f"Error extracting {field} in {category}: {e}")

        # Apply cross-field validation (without carrier-specific rules)
        extracted_values = apply_cross_field_validation(extracted_values)

        # Apply feedback learning
        extracted_values = apply_feedback_learning(extracted_values, self.document_hash)

        return extracted_values
        
    def _initialize_extracted_values(self):
        """
        Initialize the extracted values dictionary with default values
        
        Returns:
            dict: Empty extracted values structure
        """
        extracted_values = {}

        # Ensure all categories and fields exist
        for category in INSURANCE_SCHEMA:
            extracted_values[category] = {}
            for field in INSURANCE_SCHEMA[category]:
                extracted_values[category][field] = "Not Found"

        # Add sources dictionary to store source information
        extracted_values["sources"] = {}
        for category in INSURANCE_SCHEMA:
            extracted_values["sources"][category] = {}
            for field in INSURANCE_SCHEMA[category]:
                extracted_values["sources"][category][field] = ""
                
        return extracted_values