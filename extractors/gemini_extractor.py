import os
import streamlit as st
from google.generativeai import configure, GenerativeModel
import base64
from config import INSURANCE_SCHEMA, SCHEMA_DESCRIPTION

class GeminiFlashExtractor:
    """
    Insurance policy data extractor using Google's Gemini 2.0 Flash model with direct PDF input only
    """
    
    def __init__(self):
        """Initialize the Gemini extractor with API key from environment variable"""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        configure(api_key=api_key)
        
        # Model Selection:
        self.model_name = 'gemini-2.0-flash'  
        self.model = GenerativeModel(self.model_name)
        
    def extract_data(self, pdf_bytes, page_texts, document_hash):
        """Extract structured data directly from PDF only"""
        st.write("Scanning entire PDF with insurance policy AI...")
        
        # Build the prompt with the schema and example
        import json
        schema_json = json.dumps(INSURANCE_SCHEMA, indent=2)
        schema_description = json.dumps(SCHEMA_DESCRIPTION, indent=2)
        
        prompt_text = f"""
        You are an expert insurance policy data extraction AI. Your job is to extract structured data from this PDF insurance policy document and return it in JSON format.
        
        Extract all insurance policy data according to this json schema:
        
        {schema_json}
        
        Here is a description of each of the fields of the schema for more context:
        
        {schema_description}
        
        Here is an example of properly filled out data in the exact structure required:
        
        
        Follow this example and schema format closely. Return ONLY THE JSON with the extracted data. 
        FOR MISSING FIELDS: use "none" or "$0" for monetary values. Do not use "N/A" or "not applicable" or nill or null in the extraction.
        
        Make sure to keep the exact nested structure shown in the schema and example and only output that.
        
        Do not include any other text or explanations. Just return the JSON data.
        
        CRITICAL INSTRUCTION: You must NEVER use vague phrases like "some waiting period applies" or "varies by benefit". ALWAYS extract the EXACT numerical values, time periods, or specific conditions from the document.

        For example:
        - Instead of "waiting period applies", provide "90 days waiting period"
        - Instead of "some benefits available", list the actual benefits like "dental, vision, prescription coverage"
        - Instead of "varies by condition", provide the actual conditions and their specific values
        
        If the policy states specific values such as a waiting period or a type of benefit or a maximum benefit period, USE THOSE ACTUAL VALUES instead of generic text, determine what the actual values are and use them, even if there are multiple considerations. For example, if it mentions a waiting period, provide the ACTUAL number of days that the waiting period is and what its for; if it lists benefits, include the SPECIFIC benefits listed and in what situation they would be relevant based on the json field. This is important!
        
        BAD: "waiting_period": "Some waiting period will apply to some benefits"
        GOOD: "waiting_period": "90 days for disability benefits, 30 days for prescription coverage"

        BAD: "maximum_benefit": "Benefits may vary"
        GOOD: "maximum_benefit": "$2,000,000 lifetime maximum"

        BAD: "coverage_options": "Various coverage options available"
        GOOD: "coverage_options": ["$50,000 basic", "$100,000 enhanced", "$150,000 premium"]
        
        IMPORTANT INSTRUCTION: Say the month count for the Eligibility Period. 
        
        Make sure the text in the json is formatted correctly do not add line breaks or other stuff like that.      
                
        Examine the attached PDF document carefully including all images, tables, and all pages, and extract all relevant insurance policy information. Here is the policy to perform the analysis and extraction on:
        """
        
        try:
            # Create PDF blob for direct API use
            pdf_part = {
                "mime_type": "application/pdf",
                "data": base64.b64encode(pdf_bytes).decode('utf-8')
            }
            
            # Send both the prompt and PDF to the model - ONLY use the direct PDF approach
            response = self.model.generate_content(
                contents=[prompt_text, pdf_part]
            )
            st.success("✅ Successfully processed document")
                

            
            # Try to parse it as JSON for the app
            try:
                import json
                import re
                
                # Try to find JSON in code blocks first
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.text)
                if json_match:
                    json_str = json_match.group(1).strip()
                    json_data = json.loads(json_str)
                else:
                    # Try the whole response
                    json_data = json.loads(response.text.strip())
                
                return json_data
            except Exception as e:
                # If parsing fails, return empty dict
                st.error(f"Couldn't parse response as JSON: {str(e)}")
                return {}
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return {}