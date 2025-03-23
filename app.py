import os
import json
import time
import fitz
import pandas as pd
import streamlit as st
import tempfile
import re
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


STANDARD_SCHEMA = {
    "policy_metadata": {
        "company_name": None,
        "policy_number": None,
        "policy_effective_date": None,
    },
    "general_provisions": {
        "eligibility_period": None,
        "definition_of_salary": None,
        "child_coverage_terminates_at_age": None,
        "student_extension_to_age": None,
    },
    "life_insurance_and_add": {
        "benefit_amount": None,
        "non_evidence_maximum": None,
        "overall_maximum": None,
        "reduction_schedule": None,
        "termination_age": None,
    },
    "optional_life_insurance": {
        "available": None,
    },
    "dependent_life_insurance": {
        "spouse_amount": None,
        "child_amount": None,
        "children_covered_from_age": None,
        "termination_age": None,
    },
    "short_term_disability": {
        "available": None,
    },
    "long_term_disability": {
        "available": None,
    },
    "critical_illness": {
        "policy_number": None,
        "benefit_amount": None,
        "covered_conditions": None,
        "multi_occurrence": None,
        "dependent_coverage": None,
        "termination_age": None,
    },
    "health_care": {
        "preferred_provider_arrangement": None,
        "deductible_drugs": None,
        "dispensing_fee_cap": None,
        "deductible_other_health_care": None,
        "co_insurance_drugs": None,
        "co_insurance_professional_services": None,
        "co_insurance_other_health_care": None,
        "drug_card": None,
        "hospital": None,
        "in_home_nursing_care": None,
        "psychologist": None,
        "chiropractor": None,
        "acupuncture": None,
        "naturopath_homeopath": None,
        "physiotherapist": None,
        "podiatrist": None,
        "osteopath": None,
        "speech_therapist": None,
        "massage_therapist": None,
        "orthotics_orthopedic_shoes": None,
        "hearing_aids": None,
        "gender_affirmation": None,
        "eye_exams": None,
        "vision_care": None,
        "out_of_province_country_coverage": None,
        "trip_limitation": None,
        "termination_age": None,
    },
    "dental_care": {
        "available": None,
    },
    "additional_benefits": {
        "employee_assistance_plan": None,
        "medical_second_opinion_service": None,
        "healthcare_spending_account": None,
        "cost_plus_plan": None,
        "virtual_healthcare": None,
    },
    "other": {
        "details": None,
    }
}


@dataclass
class ExtractionResult:
    """Class to store extraction results with confidence scores"""
    value: str = None
    confidence: float = 0.0
    source: str = None  # Method that found this result
    context: str = None  # Text surrounding the extracted value


class ContextWindowExtractor:
    def __init__(self):
        self.max_workers = 4
        self.context_window_size = 150  # Characters before and after key terms
        self.field_key_terms = self._initialize_key_terms()
        self.value_extractors = self._initialize_value_extractors()
        
    def _initialize_key_terms(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize key terms for each field in the schema"""
        terms = {}
        
        # Policy metadata key terms
        terms["policy_metadata"] = {
            "company_name": ["insurance company", "insurer", "carrier", "underwriter", "provider"],
            "policy_number": ["policy number", "policy #", "policy no", "group number", "certificate"],
            "policy_effective_date": ["effective date", "policy date", "commencement date", "start date"]
        }
        
        # General provisions key terms
        terms["general_provisions"] = {
            "eligibility_period": ["eligibility period", "waiting period", "probationary period", "becomes eligible"],
            "definition_of_salary": ["definition of salary", "salary definition", "earnings definition", "salary means"],
            "child_coverage_terminates_at_age": ["child coverage terminates", "dependent coverage ends", "child eligibility ends", "dependent termination age"],
            "student_extension_to_age": ["student extension", "full-time student", "student coverage", "extended coverage", "student eligibility"]
        }
        
        # Life insurance key terms
        terms["life_insurance_and_add"] = {
            "benefit_amount": ["life insurance benefit", "life benefit", "basic life", "life coverage", "death benefit"],
            "non_evidence_maximum": ["non evidence maximum", "non-evidence limit", "guaranteed issue", "without evidence", "evidence-free maximum"],
            "overall_maximum": ["overall maximum", "maximum benefit", "coverage maximum", "maximum coverage", "maximum amount"],
            "reduction_schedule": ["reduction schedule", "benefit reduces", "age reduction", "reduced benefit", "decreases at age"],
            "termination_age": ["termination age", "coverage terminates", "benefit terminates", "terminates at age", "expires at age"]
        }
        
        # Optional life insurance key terms
        terms["optional_life_insurance"] = {
            "available": ["optional life", "voluntary life", "supplemental life", "additional life", "extra life"]
        }
        
        # Dependent life insurance key terms
        terms["dependent_life_insurance"] = {
            "spouse_amount": ["spouse amount", "spouse benefit", "spousal coverage", "husband/wife benefit", "partner benefit"],
            "child_amount": ["child amount", "child benefit", "dependent child", "children's benefit", "minor coverage"],
            "children_covered_from_age": ["children covered from", "child eligibility", "child coverage begins", "dependent from age", "eligible child age"],
            "termination_age": ["dependent termination", "dependent coverage ends", "spouse termination", "spouse coverage ends"]
        }
        
        # STD key terms
        terms["short_term_disability"] = {
            "available": ["short term disability", "STD", "weekly indemnity", "salary continuance", "sick leave"]
        }
        
        # LTD key terms
        terms["long_term_disability"] = {
            "available": ["long term disability", "LTD", "disability income", "income protection", "disability benefit"]
        }
        
        # Critical illness key terms
        terms["critical_illness"] = {
            "policy_number": ["critical illness policy", "CI policy number", "critical illness certificate", "CI certificate"],
            "benefit_amount": ["critical illness benefit", "CI benefit", "critical illness coverage", "CI coverage amount"],
            "covered_conditions": ["covered conditions", "critical conditions", "covered illnesses", "CI conditions", "covered diagnosis"],
            "multi_occurrence": ["multi occurrence", "multiple occurrences", "subsequent diagnosis", "recurrence", "multiple claims"],
            "dependent_coverage": ["dependent critical illness", "dependent CI", "spouse critical illness", "child critical illness"],
            "termination_age": ["CI terminates", "critical illness terminates", "CI termination age", "CI coverage ends"]
        }
        
        # Health care key terms (partial list - would be expanded for all fields)
        terms["health_care"] = {
            "preferred_provider_arrangement": ["preferred provider", "provider network", "pharmacy network", "dispensing network"],
            "deductible_drugs": ["drug deductible", "pharmacy deductible", "prescription deductible", "medication deductible"],
            "dispensing_fee_cap": ["dispensing fee cap", "dispensing fee maximum", "fee cap", "pharmacist fee"],
            # More health care terms would be added here
            "termination_age": ["health care terminates", "health coverage ends", "medical terminates", "health benefits end"]
        }
        
        # Dental care key terms
        terms["dental_care"] = {
            "available": ["dental care", "dental coverage", "dental benefit", "dental plan", "dental insurance"]
        }
        
        # Additional benefits key terms
        terms["additional_benefits"] = {
            "employee_assistance_plan": ["employee assistance program", "EAP", "counseling benefit", "employee support"],
            "medical_second_opinion_service": ["second opinion", "medical second opinion", "expert medical opinion", "specialist consult"],
            "healthcare_spending_account": ["healthcare spending account", "HSA", "health spending", "flexible spending", "health care account"],
            "cost_plus_plan": ["cost plus", "cost plus plan", "cost plus arrangement", "cost plus account"],
            "virtual_healthcare": ["virtual healthcare", "telehealth", "telemedicine", "virtual care", "online doctor"]
        }
        
        # Other section key terms
        terms["other"] = {
            "details": ["additional coverage", "additional benefits", "other benefits", "miscellaneous benefits"]
        }
        
        return terms
    
    def _initialize_value_extractors(self) -> Dict[str, re.Pattern]:
        """Initialize specialized extractors for different value formats"""
        extractors = {
            "money": re.compile(r'(\$[\d,\.]+|\d+(?:\.\d+)?[\s]*dollars|\d+(?:\.\d+)?[\s]*cents|\d+(?:\.\d+)?[\s]*\$)'),
            "percentage": re.compile(r'(\d+(?:\.\d+)?[\s]*(?:percent|pct|%))'),
            "age": re.compile(r'(\d+)[\s]*(?:years|year|yrs|yr)?[\s]*(?:of age|old)?'),
            "time_period": re.compile(r'(\d+)[\s]*(?:days?|weeks?|months?|years?)'),
            "date": re.compile(r'(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.\s]*\d{1,2}[,.\s]*\d{2,4}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})'),
            "yes_no": re.compile(r'(yes|no|included|not included|available|not available|excluded|covered|not covered)', re.IGNORECASE),
        }
        return extractors
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            pdf_document = fitz.open(pdf_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text("text") + "\n\n"
            pdf_document.close()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""

    def chunk_text(self, text: str, max_chars: int = 10000) -> List[str]:
        """Split text into manageable chunks"""
        chunks = []
        current_chunk = ""
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chars:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def extract_with_context(self, text: str, section: str, field: str) -> List[ExtractionResult]:
        """Extract values using context windows around key terms"""
        results = []
        
        # Skip if no key terms for this field
        if section not in self.field_key_terms or field not in self.field_key_terms[section]:
            return results
        
        key_terms = self.field_key_terms[section][field]
        
        for term in key_terms:
            # Find all occurrences of the term
            matches = re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE)
            
            for match in matches:
                # Extract context window
                start = max(0, match.start() - self.context_window_size)
                end = min(len(text), match.end() + self.context_window_size)
                context = text[start:end]
                
                # Extract value from context
                value, confidence = self._extract_value_from_context(context, section, field)
                
                if value:
                    results.append(ExtractionResult(
                        value=value,
                        confidence=confidence,
                        source="context_window",
                        context=context
                    ))
        
        return results

    def _extract_value_from_context(self, context: str, section: str, field: str) -> Tuple[Optional[str], float]:
        """Extract value from context using field-specific logic"""
        # Determine likely format based on field name
        format_type = self._determine_format_type(section, field)
        
        # Apply format-specific extraction
        if format_type == "money":
            return self._extract_monetary_value(context)
        elif format_type == "percentage":
            return self._extract_percentage(context)
        elif format_type == "age":
            return self._extract_age(context)
        elif format_type == "time_period":
            return self._extract_time_period(context)
        elif format_type == "date":
            return self._extract_date(context)
        elif format_type == "yes_no":
            return self._extract_yes_no(context)
        else:
            # General text extraction
            return self._extract_general_text(context)

    def _determine_format_type(self, section: str, field: str) -> str:
        """Determine the expected format type based on field name"""
        field_lower = field.lower()
        
        if "amount" in field_lower or "maximum" in field_lower or "deductible" in field_lower:
            return "money"
        elif "co_insurance" in field_lower or "percentage" in field_lower:
            return "percentage"
        elif "age" in field_lower or "terminates" in field_lower:
            return "age"
        elif "period" in field_lower:
            return "time_period"
        elif "date" in field_lower:
            return "date"
        elif "available" in field_lower:
            return "yes_no"
        else:
            return "text"

    def _extract_monetary_value(self, context: str) -> Tuple[Optional[str], float]:
        """Extract monetary values from context"""
        # Look for currency amounts with $ sign or words like "dollars"
        matches = self.value_extractors["money"].findall(context)
        if matches:
            # Return the first match for now - could be improved with proximity analysis
            return matches[0].strip(), 0.8
        return None, 0.0

    def _extract_percentage(self, context: str) -> Tuple[Optional[str], float]:
        """Extract percentage values from context"""
        matches = self.value_extractors["percentage"].findall(context)
        if matches:
            return matches[0].strip(), 0.8
        return None, 0.0

    def _extract_age(self, context: str) -> Tuple[Optional[str], float]:
        """Extract age values from context"""
        matches = self.value_extractors["age"].findall(context)
        if matches:
            # Get the number only
            age = matches[0].strip()
            if age.isdigit():  # Ensure it's a valid number
                return f"{age} years", 0.8
        return None, 0.0

    def _extract_time_period(self, context: str) -> Tuple[Optional[str], float]:
        """Extract time period values from context"""
        matches = self.value_extractors["time_period"].findall(context)
        if matches:
            return matches[0].strip(), 0.8
        return None, 0.0

    def _extract_date(self, context: str) -> Tuple[Optional[str], float]:
        """Extract date values from context"""
        matches = self.value_extractors["date"].findall(context)
        if matches:
            return matches[0].strip(), 0.8
        return None, 0.0

    def _extract_yes_no(self, context: str) -> Tuple[Optional[str], float]:
        """Extract yes/no values from context"""
        matches = self.value_extractors["yes_no"].findall(context)
        if matches:
            value = matches[0].lower().strip()
            # Normalize responses
            if value in ["yes", "included", "available", "covered"]:
                return "Yes", 0.9
            elif value in ["no", "not included", "not available", "excluded", "not covered"]:
                return "No", 0.9
        return None, 0.0

    def _extract_general_text(self, context: str) -> Tuple[Optional[str], float]:
        """Extract general text value from context"""
        # Look for text after colon or similar delimiters
        matches = re.search(r'[:=]\s*([^\.;,\n]{1,50})', context)
        if matches:
            return matches.group(1).strip(), 0.7
        
        # If no clear delimiter, try to extract a reasonable snippet after the key term
        matches = re.search(r'(?:^|\w+)\s+([^\.;,\n]{5,50})', context)
        if matches:
            return matches.group(1).strip(), 0.5
            
        return None, 0.0

    def process_chunk(self, chunk: str, sections: List[str]) -> Dict[str, Dict[str, ExtractionResult]]:
        """Process a chunk of text to extract information for specified sections"""
        results = {}
        
        for section in sections:
            if section not in results:
                results[section] = {}
                
            if section in STANDARD_SCHEMA:
                for field in STANDARD_SCHEMA[section]:
                    # Extract using context windows
                    field_results = self.extract_with_context(chunk, section, field)
                    
                    # Store the best result (highest confidence)
                    if field_results:
                        best_result = max(field_results, key=lambda x: x.confidence)
                        results[section][field] = best_result
        
        return results

    def merge_results(self, results_list: List[Dict[str, Dict[str, ExtractionResult]]]) -> Dict[str, Dict[str, str]]:
        """Merge results from multiple chunks, selecting the highest confidence results"""
        merged = {}
        
        for section in STANDARD_SCHEMA:
            merged[section] = {}
            for field in STANDARD_SCHEMA[section]:
                merged[section][field] = None
                
                best_confidence = 0.0
                best_value = None
                
                for results in results_list:
                    if section in results and field in results[section]:
                        result = results[section][field]
                        if result and result.confidence > best_confidence:
                            best_confidence = result.confidence
                            best_value = result.value
                
                if best_value:
                    merged[section][field] = best_value
                else:
                    format_type = self._determine_format_type(section, field)
                    if format_type == "money":
                        merged[section][field] = "$0"
                    elif format_type == "percentage":
                        merged[section][field] = "0%"
                    elif format_type == "yes_no":
                        merged[section][field] = "No"
                    else:
                        merged[section][field] = "N/A"
        
        return merged

    def extract_from_pdf(self, pdf_path: str, progress_callback=None) -> Dict[str, Dict[str, str]]:
        """Extract information from PDF using context windows approach"""
        if progress_callback:
            progress_callback(0.1, "Extracting text from PDF...")
            
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return STANDARD_SCHEMA
            
        if progress_callback:
            progress_callback(0.2, "Chunking text for processing...")
            
        chunks = self.chunk_text(pdf_text)
        
        if progress_callback:
            progress_callback(0.3, "Processing with context windows...")
        
        all_results = []
        sections = list(STANDARD_SCHEMA.keys())
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_chunk, chunk, sections): i 
                      for i, chunk in enumerate(chunks)}
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                all_results.append(chunk_result)
                
                completed += 1
                if progress_callback:
                    progress_pct = 0.3 + 0.6 * (completed / len(chunks))
                    progress_callback(progress_pct, f"Processing chunk {completed}/{len(chunks)}...")
        
        if progress_callback:
            progress_callback(0.9, "Merging results...")
            
        final_results = self.merge_results(all_results)
        
        if progress_callback:
            progress_callback(1.0, "Extraction complete!")
            
        return final_results


class StreamlitApp:
    def __init__(self):
        self.extractor = ContextWindowExtractor()

    def run(self):
        st.title("Context Window Insurance Policy Data Extractor")
        st.write("Extract insurance policy data using context windows for more accurate results")

        with st.sidebar:
            st.header("Instructions")
            st.write("""
            1. Upload an insurance policy PDF
            2. Wait for extraction to complete
            3. Review and edit the extracted data
            4. Export to CSV or JSON
            """)
            
            st.header("About This Tool")
            st.write("""
            This tool uses a context window approach to extract insurance policy details:
            
            - Searches for key terms in the policy document
            - Analyzes surrounding text to find relevant values
            - Applies specialized extraction based on field type
            - Assigns confidence scores to extraction results
            """)

        if "step" not in st.session_state:
            st.session_state.step = "upload"
            
        if st.session_state.step == "upload":
            self._upload_step()
        elif st.session_state.step == "extracting":
            self._extraction_step()
        elif st.session_state.step == "review":
            self._review_step()
        elif st.session_state.step == "export":
            self._export_step()

    def _upload_step(self):
        st.header("Upload Insurance Policy")
        uploaded_file = st.file_uploader("Select PDF file", type=["pdf"])
        
        if uploaded_file:
            st.session_state.filename = uploaded_file.name
            
            if st.button("Extract Data", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(uploaded_file.getbuffer())
                    st.session_state.temp_path = temp.name
                
                st.session_state.step = "extracting"
                st.rerun()

    def _extraction_step(self):
        st.header(f"Extracting from {st.session_state.filename}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.write(status)
        
        try:
            extracted_data = self.extractor.extract_from_pdf(
                st.session_state.temp_path,
                progress_callback=update_progress
            )
            
            st.session_state.extracted_data = extracted_data
            st.session_state.edited_data = extracted_data.copy()
            st.session_state.step = "review"
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during extraction: {e}")
            if st.button("Return to Upload"):
                st.session_state.step = "upload"
                st.rerun()

    def _review_step(self):
        st.header(f"Review Extracted Data: {st.session_state.filename}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Start Over"):
                st.session_state.step = "upload"
                st.rerun()
        with col2:
            if st.button("Export Data ➡️", type="primary"):
                st.session_state.step = "export"
                st.rerun()
        
        edited_data = {}
        
        for section, fields in st.session_state.extracted_data.items():
            section_title = section.replace("_", " ").title()
            with st.expander(section_title, expanded=True):
                if section not in edited_data:
                    edited_data[section] = {}
                    
                field_keys = list(fields.keys())
                num_fields = len(field_keys)
                
                if num_fields > 3:
                    col1, col2 = st.columns(2)
                    for i, field in enumerate(field_keys):
                        field_label = field.replace("_", " ").title()
                        container = col1 if i < (num_fields // 2) else col2
                        with container:
                            new_value = st.text_input(
                                field_label,
                                value=fields[field] if fields[field] is not None else "",
                                key=f"{section}.{field}"
                            )
                            edited_data[section][field] = new_value if new_value else None
                else:
                    for field in field_keys:
                        field_label = field.replace("_", " ").title()
                        new_value = st.text_input(
                            field_label,
                            value=fields[field] if fields[field] is not None else "",
                            key=f"{section}.{field}"
                        )
                        edited_data[section][field] = new_value if new_value else None
        
        st.session_state.edited_data = edited_data

    def _export_step(self):
        st.header("Export Data")
        
        with st.expander("Review Extracted Data", expanded=True):
            for section, fields in st.session_state.edited_data.items():
                st.subheader(section.replace("_", " ").title())
                
                col1, col2 = st.columns(2)
                field_items = list(fields.items())
                half = len(field_items) // 2
                
                for i, (field, value) in enumerate(field_items):
                    if i < half:
                        with col1:
                            if value and value not in ["None", "N/A"]:
                                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
                    else:
                        with col2:
                            if value and value not in ["None", "N/A"]:
                                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Back to Edit"):
                st.session_state.step = "review"
                st.rerun()
        
        flat_data = {}
        for section, fields in st.session_state.edited_data.items():
            for field, value in fields.items():
                flat_data[f"{section}.{field}"] = value
                
        df = pd.DataFrame([flat_data])
        
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                data=csv,
                file_name=f"insurance_extraction_{st.session_state.filename}.csv",
                mime="text/csv"
            )
            
        with col3: 
            json_data = json.dumps(st.session_state.edited_data, indent=2)
            st.download_button(
                "📄 Download JSON",
                data=json_data,
                file_name=f"insurance_extraction_{st.session_state.filename}.json",
                mime="application/json"
            )
            
        if st.button("Start New Extraction", type="primary"):
            if hasattr(st.session_state, "temp_path"):
                try:
                    os.remove(st.session_state.temp_path)
                except:
                    pass
                    
            st.session_state.step = "upload"
            st.rerun()


def main():
    st.set_page_config(
        page_title="Context Window Insurance Extractor",
        page_icon="📋",
        layout="wide"
    )
    
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()