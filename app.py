import os
import re
import json
import copy
import logging
import fitz
import pandas as pd
import streamlit as st
import tempfile
import concurrent.futures
import functools
import time
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from pathlib import Path

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants - moved outside of class for better organization
INSURANCE_TERMS = {
    "life_insurance": [
        "life insurance", "basic life", "group life", "employee life", "life benefit",
        "life coverage", "basic group life", "optional life", "optional employee life",
        "basic term life", "supplemental life", "accidental death", "ad&d", "add",
        "employee life insurance",
    ],
    "disability": [
        "disability", "ltd", "std", "long term disability", "short term disability",
        "weekly indemnity", "salary continuance", "income replacement", "disability income",
        "extended disability", "wage loss replacement", "income protection",
        "disability benefits", "sick leave", "disability insurance",
    ],
    "dental": [
        "dental", "dental care", "dental coverage", "dental plan", "dental benefit",
        "basic dental", "preventive dental", "major dental", "restorative dental",
        "orthodontic", "endodontic", "periodontic", "dentures", "dental services",
        "oral care", "dental procedures",
    ],
    "health": [
        "health", "extended health", "health care", "medical", "drug", "prescription",
        "paramedical", "vision", "hospital", "health benefits", "healthcare",
        "supplementary health", "eye care", "medical services", "health insurance",
        "prescription drugs", "medicine", "eyewear",
    ],
}

# Define the standard schema for extracted data with source tracking
STANDARD_SCHEMA = {
    "general_information": {
        "company_name": {"value": None, "source": None, "page": None, "confidence": None},
        "policy_number": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "benefit_summary": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "eligibility_period": {"value": None, "source": None, "page": None, "confidence": None},
        "definition_of_salary": {"value": None, "source": None, "page": None, "confidence": None},
        "child_coverage": {"value": None, "source": None, "page": None, "confidence": None},
        "student_extension": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "life_insurance": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "non_evidence_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "overall_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "reduction_schedule": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "optional_life_insurance": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "non_evidence_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "dependent_life_insurance": {
        "spouse_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "child_amount": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "short_term_disability": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "non_evidence_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "overall_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "waiting_period": {
            "accident": {"value": None, "source": None, "page": None, "confidence": None},
            "sickness": {"value": None, "source": None, "page": None, "confidence": None},
            "hospitalization": {"value": None, "source": None, "page": None, "confidence": None},
        },
        "maximum_benefit_period": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "long_term_disability": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "non_evidence_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "overall_maximum": {"value": None, "source": None, "page": None, "confidence": None},
        "waiting_period": {"value": None, "source": None, "page": None, "confidence": None},
        "definition_of_disability": {"value": None, "source": None, "page": None, "confidence": None},
        "maximum_benefit_period": {"value": None, "source": None, "page": None, "confidence": None},
        "cola": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
        "taxability": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "critical_illness": {
        "benefit_amount": {"value": None, "source": None, "page": None, "confidence": None},
        "covered_conditions": {"value": None, "source": None, "page": None, "confidence": None},
        "multi_occurrence": {"value": None, "source": None, "page": None, "confidence": None},
        "dependent_coverage": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "health_care": {
        "preferred_provider_arrangement": {"value": None, "source": None, "page": None, "confidence": None},
        "deductibles": {
            "drugs": {"value": None, "source": None, "page": None, "confidence": None},
            "other_health_care": {"value": None, "source": None, "page": None, "confidence": None},
        },
        "co_insurance": {
            "drugs": {"value": None, "source": None, "page": None, "confidence": None},
            "other_health_care": {"value": None, "source": None, "page": None, "confidence": None},
        },
        "drug_coverage": {"value": None, "source": None, "page": None, "confidence": None},
        "preventative_services": {"value": None, "source": None, "page": None, "confidence": None},
        "out_of_province_country_coverage": {"value": None, "source": None, "page": None, "confidence": None},
        "hospital_care": {"value": None, "source": None, "page": None, "confidence": None},
        "vision_care": {
            "eye_exams": {"value": None, "source": None, "page": None, "confidence": None},
            "glasses_contacts": {"value": None, "source": None, "page": None, "confidence": None},
        },
    },
    "dental_care": {
        "deductible": {"value": None, "source": None, "page": None, "confidence": None},
        "co_insurance": {
            "basic_services": {"value": None, "source": None, "page": None, "confidence": None},
            "major_services": {"value": None, "source": None, "page": None, "confidence": None},
            "orthodontics": {"value": None, "source": None, "page": None, "confidence": None},
        },
        "annual_maximum": {
            "basic": {"value": None, "source": None, "page": None, "confidence": None},
            "major": {"value": None, "source": None, "page": None, "confidence": None},
            "ortho": {"value": None, "source": None, "page": None, "confidence": None},
        },
        "recall_exam_frequency": {"value": None, "source": None, "page": None, "confidence": None},
        "fee_guide": {"value": None, "source": None, "page": None, "confidence": None},
        "termination_age": {"value": None, "source": None, "page": None, "confidence": None},
    },
    "additional_benefits": {
        "employee_assistance_plan": {"value": None, "source": None, "page": None, "confidence": None},
        "healthcare_spending_account": {"value": None, "source": None, "page": None, "confidence": None},
        "virtual_health_care": {"value": None, "source": None, "page": None, "confidence": None},
    },
}

# Enhanced field patterns for improved accuracy
FIELD_PATTERNS = {
    "general_information": {
        "company_name": [
            r"(?i)(?:provided by|underwritten by|insurer)[:\s]+([A-Za-z0-9\s&.,'-]+?(?:Insurance|Life|Benefits|Financial|Inc\.|Ltd\.))(?:\s|$|,|\.|;)",
            r"(?i)((?:[A-Z][a-z]*\s)+(?:Insurance|Life|Benefits|Financial|Inc\.|Ltd\.))(?:\s|$|,|\.|;)",
            r"(?i)([A-Za-z0-9\s&.,'-]+?(?:Insurance|Life|Benefits|Financial|Inc\.|Ltd\.))(?:\sPolicy)",
            r"(?i)Group Insurance Program for ([A-Za-z0-9\s&.,'-]+?)(?:\s|$|,|\.|;)",
        ],
        "policy_number": [
            r"(?i)policy\s*(?:#|number|no|num)[:\s]*([A-Z0-9-]+)",
            r"(?i)(?:group|plan|certificate)\s*(?:#|number|no|num)[:\s]*([A-Z0-9-]+)",
            r"(?i)policy\s*(?:id)[:\s]*([A-Z0-9-]+)",
            r"(?i)certificate\s*(?:id)[:\s]*([A-Z0-9-]+)",
            r"(?i)group\s*policy\s*(?:number)?[:\s]*([A-Z0-9-]+)",
        ],
    },
    "life_insurance": {
        "benefit_amount": [
            r"(?i)(?:life insurance|basic life).*?benefit\s*amount.*?(\d+(?:\.\d+)?\s*(?:times|x).*?(?:salary|earnings|pay|income))",
            r"(?i)(?:life insurance|basic life).*?(\d+(?:\.\d+)?\s*(?:times|x).*?(?:salary|earnings|pay|income))",
            r"(?i)(?:benefit amount|coverage amount|coverage).*?(\d+(?:\.\d+)?\s*(?:times|x).*?(?:salary|earnings|pay|income))",
            r"(?i)(?:benefit amount|coverage amount|coverage).*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?i)basic\s*life.*?(\d+(?:\.\d+)?\s*(?:times|x).*?(?:salary|earnings|pay|income))",
        ],
        "termination_age": [
            r"(?i)(?:life insurance|basic life).*?termination\s*age.*?(\d+)",
            r"(?i)termination\s*age.*?(\d+)(?:\s*years)?",
            r"(?i)coverage\s*(?:terminates|ends|ceases).*?age\s*(\d+)",
            r"(?i)(?:terminates|ends|ceases)\s*(?:at|on).*?age\s*(\d+)",
            r"(?i)basic\s*life.*?age\s*(\d+)",
        ],
        "non_evidence_maximum": [
            r"(?i)non(?:-|\s)evidence\s*(?:maximum|limit).*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?i)without\s*evidence.*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?i)evidence\s*free\s*(?:maximum|limit).*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?i)no\s*medical\s*evidence.*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
        ],
    },
    "long_term_disability": {
        "benefit_amount": [
            r"(?i)(?:long\s*term\s*disability|ltd).*?benefit\s*amount.*?(\d+(?:\.\d+)?%\s*(?:of|earnings|salary|income))",
            r"(?i)(?:benefit amount|coverage).*?(\d+(?:\.\d+)?%\s*(?:of|earnings|salary|income))",
            r"(?i)(\d+(?:\.\d+)?%\s*(?:of|earnings|salary|income)(?:\s*up\s*to\s*\$\d+(?:,\d+)*(?:\.\d+)?)?)",
            r"(?i)ltd.*?(\d+(?:\.\d+)?%\s*(?:of|earnings|salary|income))",
        ],
        "waiting_period": [
            r"(?i)(?:long\s*term\s*disability|ltd).*?waiting\s*period.*?(\d+\s*(?:days|weeks|months))",
            r"(?i)waiting\s*period.*?(\d+\s*(?:days|weeks|months))",
            r"(?i)elimination\s*period.*?(\d+\s*(?:days|weeks|months))",
            r"(?i)benefits\s*begin\s*after.*?(\d+\s*(?:days|weeks|months))",
            r"(?i)qualifying\s*period.*?(\d+\s*(?:days|weeks|months))",
        ],
        "maximum_benefit_period": [
            r"(?i)maximum\s*benefit\s*period.*?(age\s*\d+|\d+\s*(?:years|months))",
            r"(?i)benefits\s*(?:are\s*)?payable\s*(?:until|to).*?(age\s*\d+|\d+\s*(?:years|months))",
            r"(?i)ltd.*?(?:until|to)\s*(age\s*\d+)",
        ],
    },
    "dental_care": {
        "co_insurance": {
            "basic_services": [
                r"(?i)basic\s*(?:services|procedures|dental).*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\d+(?:\.\d+)?%)",
                r"(?i)(?:preventive|prevention|diagnostic|maintenance).*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\d+(?:\.\d+)?%)",
                r"(?i)(?:preventive|prevention|diagnostic|maintenance|basic).*?(\d+(?:\.\d+)?%)",
                r"(?i)dental\s*care.*?basic.*?(\d+(?:\.\d+)?%)",
            ],
            "major_services": [
                r"(?i)(?:major|extensive).*?(?:services|procedures|restorative).*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\d+(?:\.\d+)?%)",
                r"(?i)major\s*(?:services|procedures|restorative).*?(\d+(?:\.\d+)?%)",
                r"(?i)(?:crowns|bridges|dentures).*?(\d+(?:\.\d+)?%)",
                r"(?i)dental\s*care.*?major.*?(\d+(?:\.\d+)?%)",
            ],
            "orthodontics": [
                r"(?i)(?:orthodontic|braces).*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\d+(?:\.\d+)?%)",
                r"(?i)orthodontic\s*(?:services|procedures|treatment).*?(\d+(?:\.\d+)?%)",
                r"(?i)dental\s*care.*?ortho.*?(\d+(?:\.\d+)?%)",
            ],
        },
        "annual_maximum": {
            "basic": [
                r"(?i)(?:basic|preventive).*?annual\s*maximum.*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
                r"(?i)annual\s*maximum.*?basic.*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
                r"(?i)maximum\s*(?:per|each)\s*(?:year|12\s*months).*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            ],
        },
    },
    "health_care": {
        "drug_coverage": [
            r"(?i)(?:prescription\s*)?drugs.*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\d+(?:\.\d+)?%)",
            r"(?i)(?:prescription\s*)?drugs.*?(?:coverage|co(?:\s|-)insurance|pays|reimburse).*?(\$\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?i)drug\s*plan.*?(\d+(?:\.\d+)?%)",
        ],
        "vision_care": {
            "glasses_contacts": [
                r"(?i)(?:glasses|lenses|frames|contacts|eyewear).*?(?:coverage|pays|reimburse|up\s*to).*?(\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:per|every|each)\s*\d+\s*(?:year|month|months))?)",
                r"(?i)vision\s*care.*?(\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:per|every|each)\s*\d+\s*(?:year|month|months))?)",
            ],
        },
    },
}

# Section headers to help with context-aware extraction
SECTION_MARKERS = {
    "life_insurance": [
        r"(?i)LIFE\s+INSURANCE",
        r"(?i)BASIC\s+LIFE",
        r"(?i)EMPLOYEE\s+LIFE\s+INSURANCE",
        r"(?i)GROUP\s+LIFE\s+INSURANCE",
    ],
    "optional_life": [
        r"(?i)OPTIONAL\s+LIFE",
        r"(?i)SUPPLEMENTAL\s+LIFE",
        r"(?i)VOLUNTARY\s+LIFE",
        r"(?i)ADDITIONAL\s+LIFE",
    ],
    "long_term_disability": [
        r"(?i)LONG\s+TERM\s+DISABILITY",
        r"(?i)LTD\s+BENEFITS?",
        r"(?i)LONG\s+TERM\s+DISABILITY\s+INCOME",
    ],
    "short_term_disability": [
        r"(?i)SHORT\s+TERM\s+DISABILITY",
        r"(?i)STD\s+BENEFITS?",
        r"(?i)WEEKLY\s+INDEMNITY",
        r"(?i)SALARY\s+CONTINUANCE",
    ],
    "dental_care": [
        r"(?i)DENTAL\s+(?:CARE|BENEFITS|COVERAGE|PLAN)",
        r"(?i)DENTAL\s+INSURANCE",
    ],
    "health_care": [
        r"(?i)(?:EXTENDED|SUPPLEMENTARY)\s+HEALTH(?:\s+CARE)?",
        r"(?i)HEALTH\s+(?:CARE|BENEFITS|COVERAGE|PLAN)",
        r"(?i)MEDICAL\s+(?:CARE|BENEFITS|COVERAGE|PLAN)",
    ],
}

class InsurancePolicyExtractor:
    """
    Improved class for extracting structured data from insurance policy PDFs
    with better performance and accuracy, prioritizing LLM-based extraction
    """

    def __init__(self, use_ollama: bool = True, ollama_model: str = "mistral"):
        """
        Initialize the extractor with Ollama

        Args:
            use_ollama: Whether to use Ollama for enhanced extraction
            ollama_model: Model to use with Ollama (default: mistral)
        """
        self.use_ollama = use_ollama
        self.ollama_available = False
        self.ollama_model = ollama_model
        self.cache_dir = Path(tempfile.gettempdir()) / "insurance_extractor_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize Ollama if enabled
        if self.use_ollama and OLLAMA_AVAILABLE:
            try:
                # Test Ollama connection
                models = ollama.list()
                self.ollama_available = True
                logger.info(f"Connected to Ollama API - available models: {[m['name'] for m in models.get('models', [])]}")
                
                # Check if requested model is available, fallback to mistral if not
                available_models = [m['name'] for m in models.get('models', [])]
                if self.ollama_model not in available_models:
                    logger.warning(f"Requested model '{self.ollama_model}' not available. Available models: {available_models}")
                    if 'mistral' in available_models:
                        self.ollama_model = 'mistral'
                        logger.info(f"Falling back to 'mistral' model")
                    elif len(available_models) > 0:
                        self.ollama_model = available_models[0]
                        logger.info(f"Falling back to '{self.ollama_model}' model")
                    else:
                        logger.warning("No models available in Ollama")
                        self.ollama_available = False
                
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {e}")
                self.ollama_available = False
        elif self.use_ollama:
            logger.warning("Ollama module not found, install with 'pip install ollama'")

    def extract_from_pdf(self, pdf_path: str, progress_callback=None) -> Dict[str, Any]:
        """
        Extract all structured data from a PDF file with progress tracking
        and enhanced Ollama-first extraction strategy

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with extracted fields
        """
        start_time = time.time()
        logger.info(f"Extracting data from PDF: {pdf_path}")

        # Initialize the output dictionary with our standard schema
        extracted_data = copy.deepcopy(STANDARD_SCHEMA)

        # Update progress
        if progress_callback:
            progress_callback(0.05, "Initializing extraction...")

        try:
            # Extract text from PDF with page tracking
            pdf_text, page_texts = self._extract_text_from_pdf(pdf_path)
            
            if progress_callback:
                progress_callback(0.15, "Text extraction complete...")

            if not pdf_text.strip():
                logger.error("Failed to extract text from PDF - document appears to be empty or image-based")
                return extracted_data

            # Extract document sections for context-aware processing
            document_sections = self._identify_document_sections(pdf_text, page_texts)
            
            if progress_callback:
                progress_callback(0.25, "Document sections identified...")

            # Extract tables from PDF
            tables = self._extract_tables_from_pdf(pdf_path)
            
            if progress_callback:
                progress_callback(0.35, "Tables extracted...")

            # Log table info for debugging
            logger.info(f"Extracted {len(tables)} tables from PDF")
            for i, table in enumerate(tables[:3]):  # Log first 3 tables only
                logger.info(f"Table {i+1} shape: {table.shape}")
                logger.info(f"Table {i+1} columns: {list(table.columns)}")

            # CHANGED STRATEGY: Use Ollama as primary extraction method when available
            if self.use_ollama and self.ollama_available:
                if progress_callback:
                    progress_callback(0.40, "Starting comprehensive LLM-based extraction...")
                    
                # Primary extraction using LLM
                self._enhance_with_ollama(pdf_text, page_texts, extracted_data, document_sections)
                
                if progress_callback:
                    progress_callback(0.80, "LLM-based extraction complete...")
                    
                # Use rules-based approach as fallback to fill in any missing fields
                self._extract_using_rules(pdf_text, page_texts, tables, extracted_data, document_sections)
                
                if progress_callback:
                    progress_callback(0.90, "Fallback rule-based extraction complete...")
            else:
                # Fallback to rules-based approach only if Ollama not available
                if progress_callback:
                    progress_callback(0.40, "Ollama not available, using rule-based extraction only...")
                    
                self._extract_using_rules(pdf_text, page_texts, tables, extracted_data, document_sections)
                
                if progress_callback:
                    progress_callback(0.90, "Rule-based extraction complete...")

            # Post-process to improve consistency and formatting
            if progress_callback:
                progress_callback(0.95, "Post-processing and validating extracted data...")
                
            self._post_process_extracted_data(extracted_data)

            # Count successfully extracted fields
            extracted_count = 0
            for section in extracted_data.values():
                for field in section.values():
                    if isinstance(field, dict) and field.get("value"):
                        extracted_count += 1
                    elif isinstance(field, dict):
                        for subfield in field.values():
                            if isinstance(subfield, dict) and subfield.get("value"):
                                extracted_count += 1

            logger.info(f"Successfully extracted {extracted_count} fields from PDF")
            logger.info(f"Total extraction time: {time.time() - start_time:.2f} seconds")
            
            if progress_callback:
                progress_callback(1.0, "Extraction complete!")
                
            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting data from PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return extracted_data

    def _extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract all text from a PDF file with page tracking using multiple methods
        to ensure maximum text recovery and performance optimizations

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple containing:
                - String with all text from the PDF
                - Dictionary mapping page numbers to page text
        """
        text = ""
        page_texts = {}

        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"PDF has {pdf_document.page_count} pages")

            # Use ThreadPoolExecutor for parallel processing of pages
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a function to process a single page
                def process_page(page_num):
                    page = pdf_document[page_num]
                    
                    # Try different text extraction methods
                    page_text = page.get_text("text")  # Basic text extraction
                    
                    # If basic extraction yields very little text, try other methods
                    if len(page_text.strip()) < 50:
                        logger.info(f"Page {page_num+1} has little text, trying HTML extraction")
                        try:
                            page_text_html = page.get_text("html")
                            
                            # Extract text from HTML if it yields more content
                            if len(page_text_html) > len(page_text):
                                import re
                                # Remove HTML tags to get plain text
                                page_text = re.sub(r"<.*?>", " ", page_text_html)
                                page_text = re.sub(r"\s+", " ", page_text).strip()
                        except Exception as e:
                            logger.warning(f"HTML extraction failed for page {page_num+1}: {e}")
                    
                    # If still little text, the page might be an image/scan
                    if len(page_text.strip()) < 50:
                        logger.info(f"Page {page_num+1} may be an image/scan")
                    
                    return page_num + 1, page_text

                # Submit all page extraction tasks
                futures = [executor.submit(process_page, page_num) for page_num in range(pdf_document.page_count)]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    page_num, page_text = future.result()
                    page_texts[page_num] = page_text
                    
            # Combine all page texts in order
            for page_num in sorted(page_texts.keys()):
                text += page_texts[page_num] + "\n\n"  # Add double newline between pages

            pdf_document.close()
            return text, page_texts

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return text, page_texts

    def _identify_document_sections(self, pdf_text: str, page_texts: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
        """
        Identify key sections in the document for context-aware extraction

        Args:
            pdf_text: Full text of the PDF
            page_texts: Dictionary mapping page numbers to page text

        Returns:
            Dictionary mapping section types to section info (text, page, etc.)
        """
        sections = {}
        
        # Find sections based on section markers
        for section_type, markers in SECTION_MARKERS.items():
            # Try to find each marker
            for marker in markers:
                # Search all page texts to find the section
                for page_num, page_text in page_texts.items():
                    match = re.search(marker, page_text)
                    if match:
                        # Extract the section text - try to find the end by looking for next section header
                        start_pos = match.start()
                        section_text = page_text[start_pos:]
                        
                        # Try to find the end using other section markers
                        section_end_matches = []
                        for other_type, other_markers in SECTION_MARKERS.items():
                            if other_type != section_type:
                                for other_marker in other_markers:
                                    end_match = re.search(other_marker, section_text)
                                    if end_match and end_match.start() > 10:  # Ensure it's not part of current header
                                        section_end_matches.append(end_match.start())
                        
                        # If found other section markers, cut the text at the first one
                        if section_end_matches:
                            end_pos = min(section_end_matches)
                            section_text = section_text[:end_pos].strip()
                        
                        # Store the section
                        sections[section_type] = {
                            "text": section_text,
                            "page": page_num,
                            "marker": match.group(0)
                        }
                        break
                
                # If section found, stop looking for this section type
                if section_type in sections:
                    break
        
        # For any missing sections, try to find them based on key terms
        for section_type, terms in INSURANCE_TERMS.items():
            if section_type not in sections:
                # Look for sections with high concentration of relevant terms
                section_candidates = []
                
                for page_num, page_text in page_texts.items():
                    # Count matches of terms in this page
                    term_count = sum(1 for term in terms if re.search(r'\b' + re.escape(term) + r'\b', page_text, re.IGNORECASE))
                    if term_count > 3:  # At least 3 relevant terms
                        section_candidates.append((page_num, term_count))
                
                # If found candidates, use the page with highest term count
                if section_candidates:
                    best_page = max(section_candidates, key=lambda x: x[1])[0]
                    sections[section_type] = {
                        "text": page_texts[best_page],
                        "page": best_page,
                        "marker": f"Inferred {section_type} section"
                    }
        
        logger.info(f"Identified {len(sections)} document sections: {list(sections.keys())}")
        return sections
        
    def _extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF file with enhanced reliability

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of pandas DataFrames containing tables
        """
        tables = []
        
        try:
            pdf_document = fitz.open(pdf_path)

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]

                # Try the built-in table detection
                tab = page.find_tables()
                if tab.tables:
                    for i, table in enumerate(tab.tables):
                        try:
                            df = table.to_pandas()
                            # Clean up table data
                            df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

                            # Check if table is usable (not just a single cell or mostly empty)
                            if df.shape[0] > 1 and df.shape[1] > 1:
                                # Clean column names
                                df.columns = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(df.columns)]
                                
                                # Add metadata to table
                                df.attrs["page_num"] = page_num + 1
                                df.attrs["table_idx"] = i
                                
                                tables.append(df)
                                logger.info(f"Extracted table from page {page_num+1}: {df.shape}")
                        except Exception as e:
                            logger.warning(f"Failed to convert table to DataFrame: {e}")

            pdf_document.close()
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return tables

    def _extract_using_rules(
        self,
        pdf_text: str,
        page_texts: Dict[int, str],
        tables: List[pd.DataFrame],
        extracted_data: Dict[str, Any],
        document_sections: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Extract data using improved rule-based approach with source tracking and enhanced patterns

        Args:
            pdf_text: Text extracted from PDF
            page_texts: Dictionary mapping page numbers to page text
            tables: Tables extracted from PDF
            extracted_data: Dictionary to populate with extracted data
            document_sections: Dictionary with document sections for contextual extraction
        """
        # Find which page a match is on
        def find_page_for_match(match_text, default_page=None):
            if not match_text:
                return default_page, None

            match_text = str(match_text).strip()
            for page_num, page_text in page_texts.items():
                if match_text in page_text:
                    # Get some context (100 chars before and after)
                    match_pos = page_text.find(match_text)
                    start_pos = max(0, match_pos - 100)
                    end_pos = min(len(page_text), match_pos + len(match_text) + 100)
                    context = page_text[start_pos:end_pos].replace("\n", " ").strip()
                    return page_num, context
            return default_page, None

        # Helper function to update extracted data with source info
        def update_field(field_path, value, source_page=None, source_context=None, confidence=0.8, validation_func=None):
            if not value:
                return False
                
            # Validate the value if a validation function is provided
            if validation_func and not validation_func(value):
                logger.info(f"Validation failed for {field_path} = {value}")
                return False

            # Navigate to the field and update it
            path_parts = field_path.split(".")
            current = extracted_data

            for part in path_parts[:-1]:
                current = current[part]

            # Get page and context if not provided
            if source_page is None or source_context is None:
                page_num, context = find_page_for_match(value, source_page)
                source_page = page_num or source_page
                source_context = context or source_context

            # Only update if field is not already populated or we have higher confidence
            if not current[path_parts[-1]]["value"] or current[path_parts[-1]]["confidence"] < confidence:
                current[path_parts[-1]]["value"] = value
                current[path_parts[-1]]["source"] = source_context
                current[path_parts[-1]]["page"] = source_page
                current[path_parts[-1]]["confidence"] = confidence
                logger.info(f"Rule-based extraction: {field_path} = {value} (confidence: {confidence})")
                return True
            return False

        # Process field extraction based on patterns
        for section_name, fields in FIELD_PATTERNS.items():
            # Check if we have a dedicated section for this type
            section_info = document_sections.get(section_name, None)
            section_text = section_info["text"] if section_info else pdf_text
            section_page = section_info["page"] if section_info else None
            
            # Process all fields in this section
            for field_name, patterns in fields.items():
                if isinstance(patterns, dict):
                    # This is a nested field (like co_insurance)
                    for subfield_name, subpatterns in patterns.items():
                        field_path = f"{section_name}.{field_name}.{subfield_name}"
                        
                        # Try each pattern in priority order
                        for pattern in subpatterns:
                            # First try in the specific section
                            if section_info:
                                match = re.search(pattern, section_text)
                                if match:
                                    value = match.group(1).strip()
                                    if update_field(field_path, value, section_page, confidence=0.85):
                                        break
                            
                            # If not found in specific section or no section available, try in whole document
                            match = re.search(pattern, pdf_text)
                            if match:
                                value = match.group(1).strip()
                                if update_field(field_path, value, confidence=0.75):
                                    break
                else:
                    # This is a direct field
                    field_path = f"{section_name}.{field_name}"
                    
                    # Try each pattern in priority order
                    for pattern in patterns:
                        # First try in the specific section
                        if section_info:
                            match = re.search(pattern, section_text)
                            if match:
                                value = match.group(1).strip()
                                if update_field(field_path, value, section_page, confidence=0.85):
                                    break
                        
                        # If not found in specific section or no section available, try in whole document
                        match = re.search(pattern, pdf_text)
                        if match:
                            value = match.group(1).strip()
                            if update_field(field_path, value, confidence=0.75):
                                break

        # ========== TABLE EXTRACTION ==========
        # Process tables for more structured data with improved accuracy
        for table_num, table in enumerate(tables):
            try:
                # Extract page information from table metadata
                table_page = table.attrs.get("page_num", None)
                table_context = f"Table {table_num+1}, Page {table_page}" if table_page else f"Table {table_num+1}"
                
                # Convert column names to string and lowercase for easier matching
                table.columns = [str(col).lower() if not pd.isna(col) else f"col_{i}" 
                                for i, col in enumerate(table.columns)]

                # Process each row of the table
                for idx, row in table.iterrows():
                    # Convert row to string for pattern matching
                    row_str = " ".join(str(val).lower() for val in row.values if pd.notna(val))
                    
                    # Determine what type of information this row might contain
                    section_type = None
                    for key, terms in INSURANCE_TERMS.items():
                        if any(re.search(r'\b' + re.escape(term) + r'\b', row_str, re.IGNORECASE) for term in terms):
                            section_type = key
                            break
                    
                    if not section_type:
                        continue  # Skip rows we can't categorize
                        
                    # Process based on the section type
                    row_context = f"{table_context}, Row {idx+1}"
                    
                    # Extract data based on section type
                    if section_type == "life_insurance":
                        # Look for "X times salary/earnings" pattern
                        times_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)", row_str)
                        if times_match:
                            times_value = times_match.group(1)
                            
                            # Try to get the full phrase
                            full_times_match = re.search(
                                r"(\d+(?:\.\d+)?)\s*(?:x|times)(?:\s*(?:annual|basic|regular|annual))?\s*(?:salary|earnings|income)",
                                row_str,
                            )
                            
                            benefit_amount = (
                                full_times_match.group(0) if full_times_match else f"{times_value} times salary"
                            )
                            
                            update_field(
                                "life_insurance.benefit_amount", 
                                benefit_amount, 
                                table_page, 
                                row_context, 
                                confidence=0.85
                            )
                            
                    elif section_type == "disability":
                        # Determine if it's LTD or STD
                        is_ltd = any(term in row_str for term in ["long term", "ltd"])
                        is_std = any(term in row_str for term in ["short term", "std", "weekly indemnity"])
                        
                        # Look for percentage values which might indicate benefit amount
                        percentage_match = re.search(r"(\d+(?:\.\d+)?%(?:\s*of\s*(?:salary|earnings|income))?)", row_str)
                        if percentage_match:
                            percentage = percentage_match.group(1)
                            
                            if is_ltd:
                                update_field(
                                    "long_term_disability.benefit_amount", 
                                    percentage, 
                                    table_page, 
                                    row_context, 
                                    confidence=0.85
                                )
                            elif is_std:
                                update_field(
                                    "short_term_disability.benefit_amount", 
                                    percentage, 
                                    table_page, 
                                    row_context, 
                                    confidence=0.85
                                )
                                
                    elif section_type == "dental":
                        # Look for categories and percentages
                        is_basic = any(term in row_str for term in ["basic", "preventive", "routine", "diagnostic"])
                        is_major = any(term in row_str for term in ["major", "extensive", "restorative", "crown", "bridge"])
                        is_ortho = any(term in row_str for term in ["ortho", "orthodontic", "braces"])
                        
                        percentage_match = re.search(r"(\d+(?:\.\d+)?%)", row_str)
                        if percentage_match:
                            percentage = percentage_match.group(1)
                            
                            if is_basic:
                                update_field(
                                    "dental_care.co_insurance.basic_services", 
                                    percentage, 
                                    table_page, 
                                    row_context, 
                                    confidence=0.85
                                )
                            elif is_major:
                                update_field(
                                    "dental_care.co_insurance.major_services", 
                                    percentage, 
                                    table_page, 
                                    row_context, 
                                    confidence=0.85
                                )
                            elif is_ortho:
                                update_field(
                                    "dental_care.co_insurance.orthodontics", 
                                    percentage, 
                                    table_page, 
                                    row_context, 
                                    confidence=0.85
                                )
                                
                    elif section_type == "health":
                        # Look for vision, drugs, etc.
                        is_vision = any(term in row_str for term in ["vision", "glasses", "contacts", "eyewear", "eye exam"])
                        is_drugs = any(term in row_str for term in ["drug", "prescription", "medication", "pharmacy"])
                        
                        # Look for dollar amounts
                        dollar_match = re.search(r"(\$\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:per|every|each)\s*\d+\s*(?:year|month|months))?)", row_str)
                        if dollar_match and is_vision:
                            vision_amount = dollar_match.group(1)
                            update_field(
                                "health_care.vision_care.glasses_contacts", 
                                vision_amount, 
                                table_page, 
                                row_context, 
                                confidence=0.85
                            )
                            
                        # Look for percentages
                        percentage_match = re.search(r"(\d+(?:\.\d+)?%)", row_str)
                        if percentage_match and is_drugs:
                            drug_coverage = percentage_match.group(1)
                            update_field(
                                "health_care.co_insurance.drugs", 
                                drug_coverage, 
                                table_page, 
                                row_context, 
                                confidence=0.85
                            )

            except Exception as e:
                logger.warning(f"Error processing table {table_num}: {e}")
                continue

    def _post_process_extracted_data(self, extracted_data: Dict[str, Any]) -> None:
        """
        Post-process extracted data to ensure consistency, proper formatting,
        and additional validation.

        Args:
            extracted_data: Dictionary with extracted data to be post-processed
        """
        logger.info("Post-processing extracted data for improved quality")
        
        # Helper function to check if a field has been populated
        def is_populated(field_path: str) -> bool:
            path_parts = field_path.split(".")
            current = extracted_data
            
            for part in path_parts[:-1]:
                if part not in current:
                    return False
                current = current[part]
                
            last_part = path_parts[-1]
            return last_part in current and current[last_part].get("value") is not None
            
        # Helper function to get a field value
        def get_field_value(field_path: str) -> Any:
            path_parts = field_path.split(".")
            current = extracted_data
            
            for part in path_parts[:-1]:
                if part not in current:
                    return None
                current = current[part]
                
            last_part = path_parts[-1]
            if last_part in current and current[last_part].get("value") is not None:
                return current[last_part]["value"]
            return None
            
        # Helper function to set a field value
        def set_field_value(field_path: str, value: Any, confidence: float = 0.7) -> None:
            path_parts = field_path.split(".")
            current = extracted_data
            
            for part in path_parts[:-1]:
                current = current[part]
                
            last_part = path_parts[-1]
            if not current[last_part].get("value") or current[last_part].get("confidence", 0) < confidence:
                current[last_part]["value"] = value
                current[last_part]["confidence"] = confidence
                current[last_part]["source"] = "Post-processing inference"
        
        # Format percentage values consistently
        percentage_fields = [
            "long_term_disability.benefit_amount",
            "short_term_disability.benefit_amount",
            "dental_care.co_insurance.basic_services",
            "dental_care.co_insurance.major_services",
            "dental_care.co_insurance.orthodontics",
            "health_care.co_insurance.drugs",
        ]
        
        for field_path in percentage_fields:
            if is_populated(field_path):
                value = get_field_value(field_path)
                # Ensure percentage values have % sign
                if value and re.search(r'\d+', value) and "%" not in value:
                    set_field_value(field_path, f"{value}%")
                # Clean up extra text around percentages
                match = re.search(r'(\d+(?:\.\d+)?%)', value) if value else None
                if match:
                    set_field_value(field_path, match.group(1))
        
        # Format dollar amounts consistently
        dollar_fields = [
            "life_insurance.non_evidence_maximum",
            "life_insurance.overall_maximum",
            "dependent_life_insurance.spouse_amount",
            "dependent_life_insurance.child_amount",
            "dental_care.annual_maximum.basic",
            "health_care.vision_care.glasses_contacts",
        ]
        
        for field_path in dollar_fields:
            if is_populated(field_path):
                value = get_field_value(field_path)
                # Ensure dollar values have $ sign
                if value and re.search(r'\d+', value) and "$" not in value:
                    set_field_value(field_path, f"${value}")
                # Clean up extra text around dollar amounts
                match = re.search(r'(\$\d+(?:,\d+)*(?:\.\d+)?)', value) if value else None
                if match:
                    set_field_value(field_path, match.group(1))
        
        # Format ages consistently
        age_fields = [
            "life_insurance.termination_age",
            "optional_life_insurance.termination_age",
            "dental_care.termination_age",
        ]
        
        for field_path in age_fields:
            if is_populated(field_path):
                value = get_field_value(field_path)
                # Extract just the age number
                match = re.search(r'(\d+)', value) if value else None
                if match:
                    set_field_value(field_path, f"Age {match.group(1)}")
        
        # Validate and format waiting periods
        waiting_period_fields = [
            "short_term_disability.waiting_period.accident",
            "short_term_disability.waiting_period.sickness",
            "long_term_disability.waiting_period",
        ]
        
        for field_path in waiting_period_fields:
            if is_populated(field_path):
                value = get_field_value(field_path)
                # Extract the number and unit for waiting periods
                match = re.search(r'(\d+)\s*(day|days|week|weeks|month|months)', value, re.IGNORECASE) if value else None
                if match:
                    number = match.group(1)
                    unit = match.group(2).lower()
                    # Standardize plural
                    if unit == "day" and number != "1":
                        unit = "days"
                    elif unit == "week" and number != "1":
                        unit = "weeks"
                    elif unit == "month" and number != "1":
                        unit = "months"
                    
                    set_field_value(field_path, f"{number} {unit}")
        
        # Cross-validate benefits where possible
        if is_populated("life_insurance.benefit_amount") and "times" in get_field_value("life_insurance.benefit_amount").lower():
            # Life benefit described as X times salary is common
            pass  # This is already good
        
        logger.info("Post-processing complete")
    
    def _enhance_with_ollama(
        self, 
        pdf_text: str, 
        page_texts: Dict[int, str], 
        extracted_data: Dict[str, Any],
        document_sections: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Enhanced extraction using Ollama LLM as the primary method for all fields with
        comprehensive processing and validation.

        Args:
            pdf_text: Text extracted from PDF
            page_texts: Dictionary mapping page numbers to page text
            extracted_data: Dictionary to populate with extracted data
            document_sections: Dictionary with document sections for contextual extraction
        """
        if not self.use_ollama or not self.ollama_available:
            return

        logger.info("Starting comprehensive Ollama extraction for all fields")
        
        # Find which page a text appears on
        def find_page_for_text(text, default_page=None):
            if not text:
                return default_page, "Ollama extraction"

            text = str(text).strip()
            for page_num, page_text in page_texts.items():
                if text in page_text:
                    # Get context (100 chars before and after)
                    match_pos = page_text.find(text)
                    start_pos = max(0, match_pos - 100)
                    end_pos = min(len(page_text), match_pos + len(text) + 100)
                    context = page_text[start_pos:end_pos].replace("\n", " ").strip()
                    return page_num, context
            return default_page, "Ollama extraction"
        
        # Chunk the document text for processing
        chunks = self._chunk_text(pdf_text, max_length=6000, overlap=500)
        logger.info(f"Document split into {len(chunks)} chunks for comprehensive processing")
        
        # Define all fields to extract using Ollama with specific prompts
        all_extraction_fields = {
            # General information
            "general_information.company_name": {
                "prompt": "What is the name of the insurance company or insurer that provides this policy? Look for terms like 'provided by', 'underwritten by', 'insurer', etc.",
                "validation": lambda x: len(x) < 100,
                "confidence": 0.8
            },
            "general_information.policy_number": {
                "prompt": "What is the policy number, group policy number, or certificate number for this insurance policy?",
                "validation": lambda x: bool(re.search(r'[A-Z0-9-]+', x)),
                "confidence": 0.8
            },
            
            # Benefit summary
            "benefit_summary.eligibility_period": {
                "prompt": "What is the eligibility waiting period for benefits? This is usually a period like '3 months of employment' before coverage begins.",
                "validation": lambda x: len(x) < 100,
                "confidence": 0.75
            },
            "benefit_summary.definition_of_salary": {
                "prompt": "How is salary defined for benefit calculations in this policy? Look for terms like 'base salary', 'annual earnings', etc.",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.75
            },
            
            # Life insurance
            "life_insurance.benefit_amount": {
                "prompt": "What is the life insurance benefit amount? This is typically expressed as a multiple of salary (e.g., '2 times annual earnings') or a fixed amount.",
                "validation": lambda x: bool(re.search(r'(times|x|\$)', x, re.IGNORECASE)),
                "confidence": 0.85
            },
            "life_insurance.non_evidence_maximum": {
                "prompt": "What is the non-evidence maximum or guaranteed issue amount for life insurance? This is the maximum coverage without requiring medical evidence.",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "life_insurance.overall_maximum": {
                "prompt": "What is the overall maximum life insurance coverage amount available?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "life_insurance.reduction_schedule": {
                "prompt": "Is there a reduction schedule for life insurance? This would describe how coverage is reduced at certain ages (e.g., reduces to 50% at age 65).",
                "validation": lambda x: len(x) < 200,
                "confidence": 0.75
            },
            "life_insurance.termination_age": {
                "prompt": "At what age does life insurance coverage terminate? Look for phrases like 'terminates at age X' or 'coverage ends at age X'.",
                "validation": lambda x: bool(re.search(r'(\d+)', x)),
                "confidence": 0.85
            },
            
            # Optional life insurance
            "optional_life_insurance.benefit_amount": {
                "prompt": "What is the optional or voluntary life insurance benefit amount or coverage available? This would be additional coverage beyond the basic life insurance.",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.75
            },
            "optional_life_insurance.non_evidence_maximum": {
                "prompt": "What is the non-evidence maximum or guaranteed issue amount for optional/voluntary life insurance?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.75
            },
            "optional_life_insurance.termination_age": {
                "prompt": "At what age does optional life insurance coverage terminate?",
                "validation": lambda x: bool(re.search(r'(\d+)', x)),
                "confidence": 0.75
            },
            
            # Dependent life insurance
            "dependent_life_insurance.spouse_amount": {
                "prompt": "What is the life insurance coverage amount for a spouse or partner under the dependent coverage?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.75
            },
            "dependent_life_insurance.child_amount": {
                "prompt": "What is the life insurance coverage amount for dependent children under the dependent coverage?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.75
            },
            
            # Short Term Disability
            "short_term_disability.benefit_amount": {
                "prompt": "What is the short term disability (STD) benefit amount? This is typically expressed as a percentage of salary.",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.85
            },
            "short_term_disability.waiting_period.accident": {
                "prompt": "What is the waiting period (elimination period) for short term disability benefits for accidents?",
                "validation": lambda x: bool(re.search(r'(\d+\s*(?:day|week|month))', x, re.IGNORECASE)),
                "confidence": 0.8
            },
            "short_term_disability.waiting_period.sickness": {
                "prompt": "What is the waiting period (elimination period) for short term disability benefits for illness or sickness?",
                "validation": lambda x: bool(re.search(r'(\d+\s*(?:day|week|month))', x, re.IGNORECASE)),
                "confidence": 0.8
            },
            "short_term_disability.maximum_benefit_period": {
                "prompt": "What is the maximum benefit period for short term disability? How long can benefits be paid?",
                "validation": lambda x: bool(re.search(r'(\d+\s*(?:day|week|month))', x, re.IGNORECASE)),
                "confidence": 0.8
            },
            
            # Long Term Disability
            "long_term_disability.benefit_amount": {
                "prompt": "What is the long term disability (LTD) benefit amount? This is typically expressed as a percentage of salary.",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.85
            },
            "long_term_disability.waiting_period": {
                "prompt": "What is the waiting period (elimination period) for long term disability benefits?",
                "validation": lambda x: bool(re.search(r'(\d+\s*(?:day|week|month))', x, re.IGNORECASE)),
                "confidence": 0.8
            },
            "long_term_disability.definition_of_disability": {
                "prompt": "What is the definition of disability used for long term disability? Look for 'own occupation', 'any occupation', or similar terms.",
                "validation": lambda x: len(x) < 200,
                "confidence": 0.75
            },
            "long_term_disability.maximum_benefit_period": {
                "prompt": "What is the maximum benefit period for long term disability? How long can benefits be paid? Often this is until a certain age.",
                "validation": lambda x: len(x) < 100,
                "confidence": 0.8
            },
            "long_term_disability.taxability": {
                "prompt": "Are the long term disability benefits taxable? Is there any mention of tax treatment of benefits?",
                "validation": lambda x: len(x) < 100,
                "confidence": 0.75
            },
            
            # Dental care
            "dental_care.deductible": {
                "prompt": "What is the deductible amount for dental care coverage?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "dental_care.co_insurance.basic_services": {
                "prompt": "What percentage of basic dental services is covered? Basic services typically include preventive care, cleanings, check-ups, x-rays.",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.85
            },
            "dental_care.co_insurance.major_services": {
                "prompt": "What percentage of major dental services is covered? Major services typically include crowns, bridges, dentures, etc.",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.85
            },
            "dental_care.co_insurance.orthodontics": {
                "prompt": "What percentage of orthodontic services (braces) is covered under dental care?",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.8
            },
            "dental_care.annual_maximum.basic": {
                "prompt": "What is the annual maximum coverage amount for basic dental services?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "dental_care.recall_exam_frequency": {
                "prompt": "What is the frequency allowed for dental recall exams or check-ups? (e.g., once every 6 months)",
                "validation": lambda x: len(x) < 100,
                "confidence": 0.75
            },
            
            # Health care
            "health_care.deductibles.drugs": {
                "prompt": "What is the deductible amount specifically for prescription drugs under the health care plan?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "health_care.co_insurance.drugs": {
                "prompt": "What percentage of prescription drug costs is covered under the health care plan?",
                "validation": lambda x: bool(re.search(r'(\d+%)', x)),
                "confidence": 0.85
            },
            "health_care.drug_coverage": {
                "prompt": "Describe the prescription drug coverage under the health care plan. What type of formulary is used?",
                "validation": lambda x: len(x) < 200,
                "confidence": 0.75
            },
            "health_care.vision_care.eye_exams": {
                "prompt": "What coverage is provided for eye exams under the vision care benefits?",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.75
            },
            "health_care.vision_care.glasses_contacts": {
                "prompt": "What coverage is provided for glasses, frames, or contact lenses under the vision care benefits?",
                "validation": lambda x: bool(re.search(r'(\$[\d,]+)', x)),
                "confidence": 0.8
            },
            "health_care.hospital_care": {
                "prompt": "What type of hospital room coverage is provided? (e.g., private room, semi-private room)",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.75
            },
            
            # Additional benefits
            "additional_benefits.employee_assistance_plan": {
                "prompt": "Is there an Employee Assistance Program (EAP) mentioned in the policy? What services does it provide?",
                "validation": lambda x: len(x) < 200,
                "confidence": 0.7
            },
            "additional_benefits.healthcare_spending_account": {
                "prompt": "Is there a Healthcare Spending Account (HSA) or Health Care Spending Account (HCSA) mentioned? What is the annual amount?",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.7
            },
            "additional_benefits.virtual_health_care": {
                "prompt": "Is there any virtual healthcare or telemedicine service mentioned in the policy?",
                "validation": lambda x: len(x) < 150,
                "confidence": 0.7
            },
        }
        
        # Group fields by category for more efficient batch processing
        categories = {
            "general": [
                "general_information.company_name", 
                "general_information.policy_number",
                "benefit_summary.eligibility_period",
                "benefit_summary.definition_of_salary"
            ],
            "life": [
                "life_insurance.benefit_amount",
                "life_insurance.non_evidence_maximum",
                "life_insurance.overall_maximum",
                "life_insurance.reduction_schedule",
                "life_insurance.termination_age",
                "optional_life_insurance.benefit_amount",
                "optional_life_insurance.non_evidence_maximum",
                "optional_life_insurance.termination_age",
                "dependent_life_insurance.spouse_amount",
                "dependent_life_insurance.child_amount"
            ],
            "disability": [
                "short_term_disability.benefit_amount",
                "short_term_disability.waiting_period.accident",
                "short_term_disability.waiting_period.sickness",
                "short_term_disability.maximum_benefit_period",
                "long_term_disability.benefit_amount",
                "long_term_disability.waiting_period",
                "long_term_disability.definition_of_disability",
                "long_term_disability.maximum_benefit_period",
                "long_term_disability.taxability"
            ],
            "health_dental": [
                "dental_care.deductible",
                "dental_care.co_insurance.basic_services",
                "dental_care.co_insurance.major_services",
                "dental_care.co_insurance.orthodontics",
                "dental_care.annual_maximum.basic",
                "dental_care.recall_exam_frequency",
                "health_care.deductibles.drugs",
                "health_care.co_insurance.drugs",
                "health_care.drug_coverage",
                "health_care.vision_care.eye_exams",
                "health_care.vision_care.glasses_contacts",
                "health_care.hospital_care"
            ],
            "additional": [
                "additional_benefits.employee_assistance_plan",
                "additional_benefits.healthcare_spending_account",
                "additional_benefits.virtual_health_care"
            ]
        }
        
        # Process all chunks with all field categories for comprehensive extraction
        results = {}
        
        for category, field_paths in categories.items():
            logger.info(f"Processing {category} category with {len(field_paths)} fields")
            
            # For each category, process all chunks to find the best answers
            category_results = {}
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Prepare batch prompt for this category
                    batch_prompt = f"You are an expert insurance policy analyzer extracting information from a policy document. For each numbered question below, provide ONLY the specific information requested from the provided text. Be precise and concise. If you cannot find the exact information, respond with the question number followed by 'NOT FOUND'.\n\n"
                    
                    for idx, field_path in enumerate(field_paths):
                        field_info = all_extraction_fields[field_path]
                        batch_prompt += f"QUESTION {idx+1}: {field_info['prompt']}\n\n"
                    
                    batch_prompt += f"POLICY TEXT EXTRACT:\n{chunk}\n\n"
                    batch_prompt += "IMPORTANT: For each question, respond with ONLY the specific information requested. Format your response as 'ANSWER 1: [your answer]', 'ANSWER 2: [your answer]', etc. Be concise and precise."
                    
                    # Call Ollama with batch prompt
                    response = ollama.generate(
                        model=self.ollama_model,
                        prompt=batch_prompt,
                        temperature=0.1,  # Lower temperature for more deterministic responses
                    )
                    
                    response_text = response["response"].strip()
                    
                    # Parse responses for each field
                    for idx, field_path in enumerate(field_paths):
                        answer_marker = f"ANSWER {idx+1}:"
                        next_marker = f"ANSWER {idx+2}:" if idx < len(field_paths) - 1 else None
                        
                        # Extract answer
                        start_pos = response_text.find(answer_marker)
                        if start_pos == -1:
                            continue
                            
                        start_pos += len(answer_marker)
                        end_pos = response_text.find(next_marker, start_pos) if next_marker else len(response_text)
                        
                        value = response_text[start_pos:end_pos].strip()
                        
                        # Skip not found responses
                        if value.lower() in ["not found", "n/a", "none", "not applicable", "not specified", "unknown", "i couldn't find"]:
                            continue
                        
                        # Skip if value is too long (likely incorrect)
                        if len(value) > 300:
                            continue
                            
                        # Apply field-specific validation if available
                        field_info = all_extraction_fields[field_path]
                        validation_func = field_info.get("validation")
                        if validation_func and not validation_func(value):
                            continue
                            
                        # Store result with confidence
                        confidence = field_info.get("confidence", 0.7)
                        
                        # Increase confidence for responses with specific patterns
                        if re.search(r'\d+%', value) and "percentage" in field_info["prompt"].lower():
                            confidence += 0.1
                        if re.search(r'\$\d+', value) and "amount" in field_info["prompt"].lower():
                            confidence += 0.1
                        
                        # Track all results for this field from different chunks
                        if field_path not in category_results:
                            category_results[field_path] = []
                            
                        category_results[field_path].append({
                            "value": value,
                            "chunk_idx": chunk_idx,
                            "confidence": confidence
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing {category} category chunk {chunk_idx}: {e}")
                    continue
            
            # For each field, select the best result from all chunks
            for field_path, field_results in category_results.items():
                if not field_results:
                    continue
                    
                # Sort by confidence and choose the highest confidence result
                best_result = sorted(field_results, key=lambda x: x["confidence"], reverse=True)[0]
                
                # Add to final results
                results[field_path] = {
                    "value": best_result["value"],
                    "confidence": best_result["confidence"],
                    "chunk_idx": best_result["chunk_idx"]
                }
        
        # Update extracted_data with the results
        for field_path, result in results.items():
            try:
                # Navigate to the field
                path_parts = field_path.split(".")
                current = extracted_data
                
                for part in path_parts[:-1]:
                    current = current[part]
                    
                last_part = path_parts[-1]
                
                # Skip if already populated with higher confidence
                if current[last_part]["value"] and current[last_part]["confidence"] > result["confidence"]:
                    continue
                
                # Get page and context
                chunk_idx = result["chunk_idx"]
                chunk_text = chunks[chunk_idx]
                value = result["value"]
                
                # Try to find the text in the chunk to get page number
                page_num, context = find_page_for_text(value)
                if not context:
                    # Use chunk as context if exact match not found
                    context = f"Extracted from policy document (chunk {chunk_idx+1})"
                
                # Update the field
                current[last_part]["value"] = value
                current[last_part]["source"] = context
                current[last_part]["page"] = page_num
                current[last_part]["confidence"] = result["confidence"]
                
                logger.info(f"Ollama extraction: {field_path} = {value} (confidence: {result['confidence']})")
                
            except Exception as e:
                logger.error(f"Error updating field {field_path}: {e}")
                continue
                
        logger.info(f"Completed comprehensive Ollama extraction with {len(results)} fields extracted")

    def _chunk_text(self, text: str, max_length: int = 4000, overlap: int = 300) -> List[str]:
        """
        Split text into chunks of maximum length with overlap, trying to break at
        meaningful boundaries

        Args:
            text: Text to split
            max_length: Maximum length of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        # If text is short enough, return it as a single chunk
        if text_length <= max_length:
            return [text]

        while start < text_length:
            end = min(start + max_length, text_length)

            # Try to end at a meaningful boundary if possible
            if end < text_length:
                # Look for section headers (ALL CAPS followed by newline)
                section_end = text.rfind("\n", start, end)
                while section_end > 0:
                    # Check if line before this is ALL CAPS
                    prev_line_start = text.rfind("\n", start, section_end - 1)
                    if prev_line_start == -1:
                        prev_line_start = start
                    prev_line = text[prev_line_start:section_end].strip()

                    if prev_line.isupper() and len(prev_line) > 5:
                        # Found a section header, use this as boundary
                        end = section_end + 1
                        break

                    section_end = text.rfind("\n", start, section_end - 1)

                # If no section header found, try paragraph break
                if section_end <= start:
                    para_end = text.rfind("\n\n", start, end)
                    if para_end > start + max_length // 2:
                        end = para_end + 2
                    else:
                        # Try sentence break
                        sentence_end = text.rfind(". ", start, end)
                        if sentence_end > start + max_length // 2:
                            end = sentence_end + 2

            chunks.append(text[start:end])
            start = end - overlap

        return chunks


class StreamlitApp:
    """
    Streamlit web application for insurance policy data extraction with improved UI
    """

    def __init__(self):
        """Initialize the Streamlit application"""
        self.extractor = InsurancePolicyExtractor(use_ollama=True)

    def run(self):
        """Run the Streamlit application with improved performance and UX"""
        st.title("Insurance Policy Data Extractor")
        st.write("Extract structured data from insurance policy documents with AI-powered processing")

        # Sidebar for settings and instructions
        with st.sidebar:
            st.header("Settings")
            
            # Ollama settings
            st.subheader("Extraction Engine")
            use_ollama = st.checkbox("Use AI-powered extraction (recommended)", value=True)
            
            # LLM model selection when Ollama is enabled
            ollama_model = "mistral"
            if use_ollama:
                model_options = ["mistral", "llama2", "mixtral", "orca-mini"]
                ollama_model = st.selectbox(
                    "Select AI model (if available)",
                    options=model_options,
                    index=0,
                    help="The AI model used for extraction. Mistral is recommended for best results."
                )

            if use_ollama != self.extractor.use_ollama or ollama_model != self.extractor.ollama_model:
                self.extractor = InsurancePolicyExtractor(use_ollama=use_ollama, ollama_model=ollama_model)
                
            # Extraction detail settings
            st.subheader("Extraction Settings")
            extraction_detail = st.select_slider(
                "Extraction detail",
                options=["Basic", "Standard", "Comprehensive"],
                value="Comprehensive",
                help="Basic is fastest but extracts fewer fields. Comprehensive is thorough but takes longer."
            )
            
            if extraction_detail == "Basic":
                st.caption("Basic extraction focuses on essential policy information only")
            elif extraction_detail == "Standard":
                st.caption("Standard extraction balances speed and thoroughness")
            else:
                st.caption("Comprehensive extraction attempts to extract all fields")

            st.header("About")
            st.write(
                """
                This tool extracts structured data from insurance policy PDFs using 
                AI-powered analysis. It accurately identifies and extracts key policy 
                details like coverage amounts, deductibles, and eligibility criteria.
                """
            )

            st.header("Workflow")
            st.write(
                """
                1. Upload a PDF insurance policy
                2. Wait for AI extraction to complete
                3. Review and edit the extracted data
                4. Export the data as CSV or JSON
                """
            )

            # Performance tips
            st.header("Performance Tips")
            st.write(
                """
                - Text-based PDFs work best (scanned documents may have reduced accuracy)
                - Comprehensive extraction typically takes 30-60 seconds
                - Toggle off AI extraction for faster but less accurate results
                """
            )

            # Package installation
            st.header("Setup")
            st.info("AI-powered extraction requires Ollama to be installed on your system")
            
            with st.expander("Installation Instructions"):
                st.markdown("""
                1. Install Ollama from [ollama.ai](https://ollama.ai)
                2. Install Python client with `pip install ollama`
                3. Pull the models: `ollama pull mistral`
                """)
                
                if st.button("Install Python Client"):
                    with st.spinner("Installing Python client for Ollama..."):
                        os.system("pip install -q ollama")
                    st.success("Ollama client installed! You may need to restart the application.")

        # Initialize session state
        if "current_step" not in st.session_state:
            st.session_state.current_step = "upload"  # Possible values: 'upload', 'extracting', 'review', 'finalize'

        if "extracted_data" not in st.session_state:
            st.session_state.extracted_data = None

        if "current_file" not in st.session_state:
            st.session_state.current_file = None

        # UPLOAD STEP
        if st.session_state.current_step == "upload":
            st.write("### Upload Insurance Policy PDF")
            uploaded_file = st.file_uploader("Upload an insurance policy PDF", type=["pdf"])

            if uploaded_file:
                st.write(f"File: **{uploaded_file.name}**")
                if st.button("Extract Data", key="extract_btn", type="primary"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name
                        
                    # Set state for extracting step
                    st.session_state.current_step = "extracting"
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.temp_path = temp_path
                    st.rerun()

        # EXTRACTING STEP
        elif st.session_state.current_step == "extracting":
            st.write(f"### Extracting Data from {st.session_state.current_file}")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback
            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.write(status)
            
            # Extract data with progress updates
            try:
                extracted_data = self.extractor.extract_from_pdf(
                    st.session_state.temp_path, 
                    progress_callback=update_progress
                )
                
                # Set state variables for review step
                st.session_state.extracted_data = extracted_data
                st.session_state.current_step = "review"
                
                # Clean up the temporary file in background
                progress_bar.progress(1.0)
                status_text.write("✅ Extraction complete! Redirecting to review page...")
                
                # Wait briefly to show completion
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during extraction: {e}")
                import traceback
                st.code(traceback.format_exc())
                
                # Provide option to return to upload
                if st.button("Return to Upload"):
                    st.session_state.current_step = "upload"
                    st.rerun()

        # REVIEW STEP
        elif st.session_state.current_step == "review":
            st.write(f"### Review & Edit Extracted Data from {st.session_state.current_file}")
            st.write("Please review the automatically extracted data below. You can edit any field to correct or add information.")

            # Create a copy of data for editing if not already created
            if "edited_data" not in st.session_state:
                st.session_state.edited_data = copy.deepcopy(st.session_state.extracted_data)

            # Add buttons for navigation
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Start Over"):
                    # Clean up and reset state
                    if hasattr(st.session_state, 'temp_path') and os.path.exists(st.session_state.temp_path):
                        try:
                            os.remove(st.session_state.temp_path)
                        except:
                            pass
                    st.session_state.current_step = "upload"
                    st.session_state.extracted_data = None
                    st.session_state.current_file = None
                    if "edited_data" in st.session_state:
                        del st.session_state.edited_data
                    st.rerun()

            with col3:
                if st.button("Finalize ➡️", type="primary"):
                    st.session_state.current_step = "finalize"
                    st.rerun()

            # Display editable data in tabs
            self._display_editable_data(st.session_state.edited_data)

        # FINALIZE STEP
        elif st.session_state.current_step == "finalize":
            st.write("### Finalize and Export Data")
            st.write(f"Review final data for {st.session_state.current_file} before exporting.")

            # Preview the final data
            with st.expander("Preview Final Data", expanded=True):
                self._display_extracted_data(st.session_state.edited_data, st.session_state.current_file)

            # Export options
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("⬅️ Back to Edit"):
                    st.session_state.current_step = "review"
                    st.rerun()

            with col2:
                st.download_button(
                    label="📊 Download CSV",
                    data=self._flatten_dict_for_csv(st.session_state.edited_data).to_csv(index=False),
                    file_name=f"{st.session_state.current_file.split('.')[0]}_extracted.csv",
                    mime="text/csv",
                )

            with col3:
                # Clean values for JSON export
                clean_data = self._prepare_data_for_export(st.session_state.edited_data)
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(clean_data, indent=2),
                    file_name=f"{st.session_state.current_file.split('.')[0]}_extracted.json",
                    mime="application/json",
                )

            # Option to start over
            if st.button("Start New Extraction", type="primary"):
                # Clean up
                if hasattr(st.session_state, 'temp_path') and os.path.exists(st.session_state.temp_path):
                    try:
                        os.remove(st.session_state.temp_path)
                    except:
                        pass

                # Reset state
                st.session_state.current_step = "upload"
                st.session_state.extracted_data = None
                st.session_state.current_file = None
                if "edited_data" in st.session_state:
                    del st.session_state.edited_data
                st.rerun()

    # Continuation of the StreamlitApp class methods...

    def _display_extracted_data(self, data: Dict[str, Any], filename: str) -> None:
        """
        Display the extracted data in the Streamlit interface (read-only view)
        with improved formatting and clarity

        Args:
            data: Extracted data dictionary
            filename: Original filename
        """
        # Use tabs for different sections
        tabs = st.tabs([
            "General Information",
            "Life Insurance",
            "Disability",
            "Health & Dental",
            "All Data",
        ])

        # General Information tab
        with tabs[0]:
            st.subheader("General Information")

            col1, col2 = st.columns(2)
            with col1:
                company_name = data["general_information"]["company_name"]["value"] if isinstance(
                    data["general_information"]["company_name"], dict) else data["general_information"]["company_name"]
                st.metric("Company Name", company_name or "Not found")
            with col2:
                policy_number = data["general_information"]["policy_number"]["value"] if isinstance(
                    data["general_information"]["policy_number"], dict) else data["general_information"]["policy_number"]
                st.metric("Policy Number", policy_number or "Not found")

            st.subheader("Benefit Summary")
            benefit_data = data["benefit_summary"]
            if any(v.get("value") if isinstance(v, dict) else v for v in benefit_data.values()):
                for key, field in benefit_data.items():
                    value = field.get("value") if isinstance(field, dict) else field
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No benefit summary data found")

        # Life Insurance tab
        with tabs[1]:
            st.subheader("Life Insurance")

            life_data = data["life_insurance"]
            if any(v.get("value") if isinstance(v, dict) else v for v in life_data.values()):
                for key, field in life_data.items():
                    value = field.get("value") if isinstance(field, dict) else field
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No life insurance data found")

            st.subheader("Optional Life Insurance")
            optional_life_data = data["optional_life_insurance"]
            if any(v.get("value") if isinstance(v, dict) else v for v in optional_life_data.values()):
                for key, field in optional_life_data.items():
                    value = field.get("value") if isinstance(field, dict) else field
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No optional life insurance data found")

            st.subheader("Dependent Life Insurance")
            dependent_life_data = data["dependent_life_insurance"]
            if any(v.get("value") if isinstance(v, dict) else v for v in dependent_life_data.values()):
                for key, field in dependent_life_data.items():
                    value = field.get("value") if isinstance(field, dict) else field
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No dependent life insurance data found")

        # Disability tab
        with tabs[2]:
            st.subheader("Short Term Disability")

            std_data = data["short_term_disability"]
            has_data = any(
                v.get("value") if isinstance(v, dict) else v
                for k, v in std_data.items()
                if k != "waiting_period"
            )
            has_waiting_period = False

            if isinstance(std_data["waiting_period"], dict) and not isinstance(
                std_data["waiting_period"].get("accident"), dict
            ):
                has_waiting_period = any(std_data["waiting_period"].values())
            else:
                has_waiting_period = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in std_data["waiting_period"].values()
                )

            if has_data or has_waiting_period:
                for key, field in std_data.items():
                    if key != "waiting_period":
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

                if has_waiting_period:
                    st.write("**Waiting Period:**")
                    waiting_data = std_data["waiting_period"]
                    for key, field in waiting_data.items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.title()}:** {value}")
            else:
                st.info("No short term disability data found")

            st.subheader("Long Term Disability")

            ltd_data = data["long_term_disability"]
            if any(v.get("value") if isinstance(v, dict) else v for v in ltd_data.values()):
                for key, field in ltd_data.items():
                    value = field.get("value") if isinstance(field, dict) else field
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No long term disability data found")

        # Health & Dental tab
        with tabs[3]:
            st.subheader("Health Care")

            health_data = data["health_care"]
            has_main_data = any(
                v.get("value") if isinstance(v, dict) else v
                for k, v in health_data.items()
                if k not in ["deductibles", "co_insurance", "vision_care"]
            )

            has_deductibles = False
            if "deductibles" in health_data and isinstance(health_data["deductibles"], dict):
                has_deductibles = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in health_data["deductibles"].values()
                )

            has_coinsurance = False
            if "co_insurance" in health_data and isinstance(health_data["co_insurance"], dict):
                has_coinsurance = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in health_data["co_insurance"].values()
                )

            has_vision = False
            if "vision_care" in health_data and isinstance(health_data["vision_care"], dict):
                has_vision = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in health_data["vision_care"].values()
                )

            if has_main_data or has_deductibles or has_coinsurance or has_vision:
                for key, field in health_data.items():
                    if key not in ["deductibles", "co_insurance", "vision_care"]:
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

                if has_deductibles:
                    st.write("**Deductibles:**")
                    for key, field in health_data["deductibles"].items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.replace('_', ' ').title()}:** {value}")

                if has_coinsurance:
                    st.write("**Co-insurance:**")
                    for key, field in health_data["co_insurance"].items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.replace('_', ' ').title()}:** {value}")

                if has_vision:
                    st.write("**Vision Care:**")
                    for key, field in health_data["vision_care"].items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No health care data found")

            st.subheader("Dental Care")

            dental_data = data["dental_care"]
            has_main_data = any(
                v.get("value") if isinstance(v, dict) else v
                for k, v in dental_data.items()
                if k not in ["co_insurance", "annual_maximum"]
            )

            has_coinsurance = False
            if "co_insurance" in dental_data and isinstance(dental_data["co_insurance"], dict):
                has_coinsurance = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in dental_data["co_insurance"].values()
                )

            has_annual_max = False
            if "annual_maximum" in dental_data and isinstance(dental_data["annual_maximum"], dict):
                has_annual_max = any(
                    v.get("value") if isinstance(v, dict) else v
                    for v in dental_data["annual_maximum"].values()
                )

            if has_main_data or has_coinsurance or has_annual_max:
                for key, field in dental_data.items():
                    if key not in ["co_insurance", "annual_maximum"]:
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

                if has_coinsurance:
                    st.write("**Co-insurance:**")
                    for key, field in dental_data["co_insurance"].items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.replace('_', ' ').title()}:** {value}")

                if has_annual_max:
                    st.write("**Annual Maximum:**")
                    for key, field in dental_data["annual_maximum"].items():
                        value = field.get("value") if isinstance(field, dict) else field
                        if value:
                            st.write(f"  - **{key.title()}:** {value}")
            else:
                st.info("No dental care data found")

        # All Data tab (JSON view)
        with tabs[4]:
            st.subheader("All Extracted Data (JSON)")
            st.json(self._prepare_data_for_export(data))

    def _edit_section(self, section_data: Dict, section_path: str) -> None:
        """
        Create an editable section with source information and improved UI

        Args:
            section_data: Dictionary with section data
            section_path: Path to this section in the overall data structure
        """
        # Create a two-column layout for each field
        for field_name, field_data in section_data.items():
            if isinstance(field_data, dict) and "value" in field_data:
                # Get values
                current_value = field_data.get("value", "")
                confidence = field_data.get("confidence", None)
                source = field_data.get("source", None)
                page = field_data.get("page", None)

                # Create columns with better proportions for UI
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Display the field
                    field_label = field_name.replace("_", " ").title()
                    field_id = f"{section_path}.{field_name}"

                    # Show confidence if available with icons
                    if confidence:
                        confidence_label = ""
                        if confidence >= 0.8:
                            confidence_label = "🟢 High"
                        elif confidence >= 0.6:
                            confidence_label = "🟡 Medium"
                        else:
                            confidence_label = "🔴 Low"
                        field_label = f"{field_label} ({confidence_label})"

                    # Editable field
                    new_value = st.text_input(field_label, value=current_value, key=field_id)
                    if new_value != current_value:
                        # Update the value in session state
                        field_data["value"] = new_value

                with col2:
                    # Confidence indicator with better visualization
                    if confidence:
                        confidence_pct = int(confidence * 100)
                        st.progress(confidence)
                        st.caption(f"Confidence: {confidence_pct}%")
                        
                        # Source information with cleaner display
                        if source:
                            with st.expander("Source", expanded=False):
                                if page:
                                    st.caption(f"Page {page}")
                                st.info(source)
                    else:
                        # Placeholder for alignment
                        st.write("")
                        
                        # Still show source if available
                        if source:
                            with st.expander("Source", expanded=False):
                                if page:
                                    st.caption(f"Page {page}")
                                st.info(source)

                # Add a subtle separator between fields
                st.markdown("---")

    def _display_editable_data(self, data: Dict[str, Any]) -> None:
        """
        Display the extracted data in an editable format with improved UI organization

        Args:
            data: Dictionary with extracted data
        """
        # Use tabs for different categories
        tabs = st.tabs([
            "General Information",
            "Life Insurance",
            "Disability",
            "Health & Dental",
            "Additional Benefits",
        ])

        # General Information tab
        with tabs[0]:
            st.subheader("General Information")
            self._edit_section(data["general_information"], "general_information")

            st.subheader("Benefit Summary")
            self._edit_section(data["benefit_summary"], "benefit_summary")

        # Life Insurance tab
        with tabs[1]:
            st.subheader("Life Insurance")
            self._edit_section(data["life_insurance"], "life_insurance")

            st.subheader("Optional Life Insurance")
            self._edit_section(data["optional_life_insurance"], "optional_life_insurance")

            st.subheader("Dependent Life Insurance")
            self._edit_section(data["dependent_life_insurance"], "dependent_life_insurance")

        # Disability tab
        with tabs[2]:
            st.subheader("Short Term Disability")

            # Handle main STD fields
            std_fields = {k: v for k, v in data["short_term_disability"].items() if k != "waiting_period"}
            self._edit_section(std_fields, "short_term_disability")

            # Handle waiting period separately
            st.subheader("STD Waiting Period")
            if isinstance(data["short_term_disability"]["waiting_period"], dict):
                if "accident" in data["short_term_disability"]["waiting_period"]:
                    self._edit_section(data["short_term_disability"]["waiting_period"], "short_term_disability.waiting_period")

            st.subheader("Long Term Disability")
            self._edit_section(data["long_term_disability"], "long_term_disability")

            st.subheader("Critical Illness")
            self._edit_section(data["critical_illness"], "critical_illness")

        # Health & Dental tab
        with tabs[3]:
            st.subheader("Health Care")

            # Main health fields
            health_fields = {k: v for k, v in data["health_care"].items() if k not in ["deductibles", "co_insurance", "vision_care"]}
            self._edit_section(health_fields, "health_care")

            # Health deductibles
            st.subheader("Health Deductibles")
            if isinstance(data["health_care"]["deductibles"], dict):
                self._edit_section(data["health_care"]["deductibles"], "health_care.deductibles")

            # Health co-insurance
            st.subheader("Health Co-Insurance")
            if isinstance(data["health_care"]["co_insurance"], dict):
                self._edit_section(data["health_care"]["co_insurance"], "health_care.co_insurance")

            # Vision care
            st.subheader("Vision Care")
            if isinstance(data["health_care"]["vision_care"], dict):
                self._edit_section(data["health_care"]["vision_care"], "health_care.vision_care")

            st.subheader("Dental Care")

            # Main dental fields
            dental_fields = {k: v for k, v in data["dental_care"].items() if k not in ["co_insurance", "annual_maximum"]}
            self._edit_section(dental_fields, "dental_care")

            # Dental co-insurance
            st.subheader("Dental Co-Insurance")
            if isinstance(data["dental_care"]["co_insurance"], dict):
                self._edit_section(data["dental_care"]["co_insurance"], "dental_care.co_insurance")

            # Dental annual maximum
            st.subheader("Dental Annual Maximum")
            if isinstance(data["dental_care"]["annual_maximum"], dict):
                self._edit_section(data["dental_care"]["annual_maximum"], "dental_care.annual_maximum")

        # Additional Benefits tab
        with tabs[4]:
            st.subheader("Additional Benefits")
            self._edit_section(data["additional_benefits"], "additional_benefits")

    def _prepare_data_for_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for export by simplifying the structure

        Args:
            data: Dictionary with extracted data

        Returns:
            Dictionary with simplified data for export
        """
        export_data = {}

        def process_dict(d: Dict, prefix=""):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    if "value" in value:  # Field with metadata
                        result[key] = value["value"]
                    else:  # Nested section
                        nested = process_dict(value)
                        if nested:  # Only add non-empty sections
                            result[key] = nested
                else:
                    result[key] = value

            # Filter out None/empty values for a cleaner export
            return {k: v for k, v in result.items() if v is not None and v != ""}

        for section, section_data in data.items():
            processed = process_dict(section_data)
            if processed:  # Only add non-empty sections
                export_data[section] = processed

        return export_data

    def _flatten_dict_for_csv(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Flatten the nested dictionary for CSV export

        Args:
            data: Nested dictionary with extracted data

        Returns:
            Pandas DataFrame with flattened data
        """
        flattened = {}

        def flatten_recursive(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    if "value" in value:  # This is a field with metadata
                        flattened[prefix + key] = value["value"]
                    else:  # This is a nested section
                        flatten_recursive(value, prefix + key + ".")
                else:
                    flattened[prefix + key] = value

        flatten_recursive(data)

        # Replace None with empty strings for better CSV output
        for key, value in flattened.items():
            if value is None:
                flattened[key] = ""

        return pd.DataFrame([flattened])


def main():
    """Main function to run the Streamlit application with optimized settings"""
    st.set_page_config(
        page_title="Insurance Policy Data Extractor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add custom CSS for better UI
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton button {
        width: 100%;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize and run the app
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()