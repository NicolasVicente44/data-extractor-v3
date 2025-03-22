import os
import json
import time
import fitz
import pandas as pd
import streamlit as st
import tempfile
import requests
import concurrent.futures
from typing import Dict, List, Any, Optional
from functools import lru_cache


STANDARD_SCHEMA = {
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
        "details": None,
    },
    "dependent_life_insurance": {
        "spouse_amount": None,
        "child_amount": None,
        "children_covered_from_age": None,
        "termination_age": None,
    },
    "short_term_disability": {
        "benefit_amount": None,
        "non_evidence_maximum": None,
        "overall_maximum": None,
        "waiting_period": None,
        "definition_of_disability": None,
        "maximum_benefit_period": None,
        "cola": None,
        "termination_age": None,
        "taxability_of_benefits": None,
    },
    "long_term_disability": {
        "benefit_amount": None,
        "non_evidence_maximum": None,
        "overall_maximum": None,
        "waiting_period": None,
        "definition_of_disability": None,
        "maximum_benefit_period": None,
        "cola": None,
        "termination_age": None,
        "taxability_of_benefits": None,
    },
    "critical_illness": {
        "available": None,
        "details": None,
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
        "practitioners": None,
        "vision_care": None,
        "out_of_province_country_coverage": None,
    },
    "dental_care": {
        "deductible": None,
        "co_insurance_basic": None,
        "co_insurance_major": None,
        "co_insurance_orthodontics": None,
        "annual_maximum": None,
        "lifetime_maximum": None,
        "recall_exam_frequency": None,
        "fee_guide": None,
    },
    "additional_benefits": {
        "employee_assistance_plan": None,
        "medical_second_opinion_service": None,
        "healthcare_spending_account": None,
        "cost_plus_plan": None,
    },
    "other": {
        "details": None,
    },
    "manulife_vitality": {
        "available": None,
        "details": None,
    }
}


class AIInsuranceExtractor:
    def __init__(self):
        self.model_name = "llama3:latest"
        self.ollama_url = "http://localhost:11434/api/generate"
        self.cache = {}
        self.max_workers = 4
        
    def _check_ollama_available(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                return False
                
            models = response.json()
            
            if "models" in models:
                model_list = models.get("models", [])
                return any(model.get("name") == "llama3" or model.get("name") == "llama3:latest" for model in model_list)
            else:
                model_names = [model.get("name", "") for model in models] if isinstance(models, list) else []
                return "llama3" in model_names or "llama3:latest" in model_names
        except:
            return False
    
    def extract_text_from_pdf(self, pdf_path):
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

    def chunk_text(self, text, max_chars=6000):
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
    
    def ai_chunk_text(self, text):
        if len(text) < 6000:
            return [text]
            
        chunks_count = (len(text) // 5000) + 1  
        max_chunks = min(chunks_count, 10)  # Limit to 10 chunks max
        
        prompt = f"""
        I need to split a large insurance policy document into {max_chunks} meaningful chunks for further processing.
        Please identify the {max_chunks} most logical break points in the document.
        Return ONLY the character position numbers where I should split the text, as a JSON array of numbers.
        For example: [4500, 9000, 13500, 18000]
        
        Do not include any explanation, just the JSON array.
        """
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get("response", "").strip()
                
                try:
                    import json
                    positions = json.loads(extracted_text)
                    if isinstance(positions, list) and all(isinstance(pos, int) for pos in positions):
                        chunks = []
                        start_pos = 0
                        
                        for pos in sorted(positions):
                            if pos > start_pos and pos < len(text):
                                chunks.append(text[start_pos:pos])
                                start_pos = pos
                                
                        chunks.append(text[start_pos:])
                        return chunks
                except:
                    pass
        except:
            pass
            
        # Fallback to simple chunking if AI chunking fails
        return self.chunk_text(text)
    
    @lru_cache(maxsize=32)
    def extract_section_with_ollama(self, chunk_id, section_name):
        chunk = self.chunk_cache.get(chunk_id, "")
        if not chunk:
            return {}
            
        if not self._check_ollama_available():
            st.error("Ollama service not available. Please install and run Ollama first.")
            return {}
        
        display_section = section_name.replace("_", " ").title()
        fields = STANDARD_SCHEMA.get(section_name, {}).keys()
        field_list = "\n".join([f"- {field.replace('_', ' ').title()}" for field in fields])
        
        prompt = f"""
        Extract ALL the following fields from the {display_section} section of this insurance policy text.
        
        Fields to extract:
        {field_list}
        
        For each field:
        - If you find an exact value (like '$500,000', '80%', '65 years', etc.), include it
        - If not found, use null
        
        Return ONLY a valid JSON object with the fields as keys.
        Do not include any explanation or other text outside the JSON object.
        
        Insurance policy text:
        {chunk}
        """
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get("response", "").strip()
                
                try:
                    # First try to parse the entire text as JSON
                    extracted_data = json.loads(extracted_text)
                    return extracted_data
                except json.JSONDecodeError:
                    # If that fails, try to find a JSON object in the text
                    start_idx = extracted_text.find('{')
                    end_idx = extracted_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_text = extracted_text[start_idx:end_idx+1]
                        try:
                            extracted_data = json.loads(json_text)
                            return extracted_data
                        except:
                            pass
                    
                    return {}
            else:
                return {}
                
        except Exception as e:
            st.warning(f"Error calling Ollama: {e}")
            return {}
    
    def process_chunk(self, chunk_id, chunk, sections):
        results = {}
        
        self.chunk_cache[chunk_id] = chunk
        
        for section_name in sections:
            section_data = self.extract_section_with_ollama(chunk_id, section_name)
            
            if section_name not in results:
                results[section_name] = {}
                
            for field in STANDARD_SCHEMA.get(section_name, {}):
                field_display = field.replace("_", " ").title()
                value = section_data.get(field) or section_data.get(field_display)
                
                if value and str(value).lower() not in ["none", "null", "n/a", "not found"]:
                    results[section_name][field] = value
        
        return results
    
    def merge_results(self, results_list):
        merged = {}
        
        for section in STANDARD_SCHEMA:
            merged[section] = {}
            for field in STANDARD_SCHEMA[section]:
                merged[section][field] = None
                
                for results in results_list:
                    if section in results and field in results[section]:
                        if results[section][field] is not None:
                            merged[section][field] = results[section][field]
                            break
                
                if merged[section][field] is None:
                    if "amount" in field or "maximum" in field:
                        merged[section][field] = "$0"
                    elif "percentage" in field or "co_insurance" in field:
                        merged[section][field] = "0%"
                    else:
                        merged[section][field] = "N/A"
        
        return merged

    def extract_from_pdf(self, pdf_path, progress_callback=None):
        if progress_callback:
            progress_callback(0.1, "Extracting text from PDF...")
            
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return STANDARD_SCHEMA
            
        if progress_callback:
            progress_callback(0.15, "Using AI to analyze document structure...")
            
        chunks = self.ai_chunk_text(pdf_text)
        
        if progress_callback:
            progress_callback(0.2, "Processing with Llama3 (using parallel processing)...")
        
        self.chunk_cache = {}
        
        all_results = []
        sections = list(STANDARD_SCHEMA.keys())
        
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            tasks.append((chunk_id, chunk, sections))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_chunk, chunk_id, chunk, sections): i 
                      for i, (chunk_id, chunk, sections) in enumerate(tasks)}
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                all_results.append(chunk_result)
                
                completed += 1
                if progress_callback:
                    progress_pct = 0.2 + 0.7 * (completed / len(tasks))
                    progress_callback(progress_pct, f"Processing chunk {completed}/{len(tasks)}...")
        
        if progress_callback:
            progress_callback(0.9, "Finalizing results with AI...")
            
        final_results = self.merge_results(all_results)
        
        # Additional pass to fill in missing values using AI
        self.fill_missing_with_ai(final_results, pdf_text, progress_callback)
        
        if progress_callback:
            progress_callback(1.0, "Extraction complete!")
            
        return final_results
    
    def fill_missing_with_ai(self, results, full_text, progress_callback=None):
        missing_fields = []
        
        for section in results:
            for field in results[section]:
                if results[section][field] in ["N/A", "$0", "0%"]:
                    missing_fields.append((section, field))
        
        if not missing_fields:
            return
            
        if progress_callback:
            progress_callback(0.92, f"Searching for {len(missing_fields)} missing values...")
            
        # Take only first 1/3 of missing fields to avoid timeout
        critical_missing = missing_fields[:len(missing_fields)//3]
        
        prompt = """
        I'm trying to extract specific insurance policy information that wasn't found in the initial pass.
        For each field below, provide the exact value from the policy if you can find it.
        If you can't find a value, respond with "NOT_FOUND".
        
        Return your answers as a JSON object with section and field names as keys.
        
        Fields to find:
        """
        
        for section, field in critical_missing:
            display_section = section.replace("_", " ").title()
            display_field = field.replace("_", " ").title()
            prompt += f"\n- {display_section}: {display_field}"
            
        prompt += f"\n\nPolicy text (excerpt):\n{full_text[:10000]}"
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get("response", "").strip()
                
                try:
                    # First try to parse the entire text as JSON
                    extracted_data = json.loads(extracted_text)
                    
                    for section in extracted_data:
                        if section in results:
                            for field in extracted_data[section]:
                                if field in results[section]:
                                    value = extracted_data[section][field]
                                    if value and value != "NOT_FOUND":
                                        results[section][field] = value
                except:
                    # JSON parsing failed, continue with current results
                    pass
                    
        except Exception as e:
            # Continue with current results if API call fails
            pass


class StreamlitApp:
    def __init__(self):
        self.extractor = AIInsuranceExtractor()

    def run(self):
        st.title("AI-Powered Insurance Policy Data Extractor")
        st.write("Extract insurance policy data using Llama3 - No regex, all AI")

        with st.sidebar:
            st.header("Settings")
            st.write("Ollama Model: llama3")
            
            st.header("Instructions")
            st.write("""
            1. Ensure Ollama is installed and running locally
            2. Upload an insurance policy PDF
            3. Wait for extraction to complete
            4. Review and edit the extracted data
            5. Export to CSV or JSON
            """)
            
            if st.button("Check Ollama Status"):
                try:
                    service_response = requests.get("http://localhost:11434/api/tags")
                    if service_response.status_code != 200:
                        st.error("❌ Ollama service is not running. Please start Ollama.")
                        st.markdown("Installation: [Ollama website](https://ollama.ai/)")
                        return
                        
                    models = service_response.json()
                    
                    if "models" in models:
                        model_list = models.get("models", [])
                        model_available = any(model.get("name") == "llama3" or model.get("name") == "llama3:latest" for model in model_list)
                    else:
                        model_names = [model.get("name", "") for model in models] if isinstance(models, list) else []
                        model_available = "llama3" in model_names or "llama3:latest" in model_names
                    
                    if model_available:
                        st.success("✅ Ollama is running and llama3 model is loaded!")
                    else:
                        st.warning("⚠️ Ollama is running but llama3 model is not loaded.")
                        st.code("Run this command: ollama pull llama3", language="bash")
                except Exception as e:
                    st.error(f"❌ Ollama is not available: {str(e)}")
                    st.markdown("Installation: [Ollama website](https://ollama.ai/)")

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
        page_title="AI-Powered Insurance Extractor",
        page_icon="📋",
        layout="wide"
    )
    
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()