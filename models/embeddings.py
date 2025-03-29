import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class EmbeddingsManager:
    """
    Creates and manages TF-IDF embeddings for document chunks and search queries
    """
    
    @staticmethod
    def create_embeddings(chunks, queries):
        """
        Create TF-IDF embeddings for chunks and queries
        
        Args:
            chunks: List of text chunks
            queries: List of search queries
            
        Returns:
            tuple: (chunk_vectors, query_vectors, vectorizer)
        """
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
    
    @staticmethod
    def semantic_search(
        chunk_vectors, query_vectors, chunks, chunk_metadata, category, field, top_k=5
    ):
        """
        Perform semantic search using vector embeddings with content-based boosts
        
        Args:
            chunk_vectors: Vector embeddings of text chunks
            query_vectors: Vector embeddings of search queries
            chunks: List of text chunks
            chunk_metadata: Metadata for each chunk
            category: Insurance category being searched
            field: Field being searched
            top_k: Number of top results to return
            
        Returns:
            list: Search results with similarity scores
        """
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