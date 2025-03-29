import re
import streamlit as st

def smart_chunk_text(text, page_texts, structure_info, chunk_size=300, overlap=100):
    """
    Split text into meaningful chunks based on document structure
    
    Args:
        text: Full document text
        page_texts: List of text content per page
        structure_info: Document structure information
        chunk_size: Target size for chunks
        overlap: Overlap between adjacent chunks
        
    Returns:
        tuple: (chunks, chunk_sources, chunk_metadata)
    """
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
                _process_large_section(
                    section_text, 
                    section_name, 
                    start_pos, 
                    chunk_size,
                    overlap,
                    boundary_patterns,
                    chunks,
                    chunk_sources,
                    chunk_metadata
                )
    else:
        # Use page boundaries if no sections found
        _chunk_by_pages(
            page_texts, 
            chunk_size, 
            overlap, 
            boundary_patterns, 
            chunks, 
            chunk_sources, 
            chunk_metadata
        )

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


def _process_large_section(
    section_text, 
    section_name, 
    start_pos, 
    chunk_size, 
    overlap, 
    boundary_patterns, 
    chunks, 
    chunk_sources, 
    chunk_metadata
):
    """Helper method to process large text sections"""
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


def _chunk_by_pages(
    page_texts, 
    chunk_size, 
    overlap, 
    boundary_patterns, 
    chunks, 
    chunk_sources, 
    chunk_metadata
):
    """Helper method to chunk text by pages"""
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