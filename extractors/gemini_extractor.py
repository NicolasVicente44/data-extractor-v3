import os
import streamlit as st
import google.generativeai as genai
from config import INSURANCE_SCHEMA, SCHEMA_DESCRIPTION, EXAMPLE_SCHEMA



class GeminiFlashExtractor:
 
    def __init__(self):
        """Initialize the Gemini extractor with API key from environment variable"""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        
        # Use the correct model name for Gemini 2.0 Flash
        self.model_name = 'gemini-2.0-flash'
        self.model = genai.GenerativeModel(self.model_name)
        
    def extract_data(self, text, page_texts, document_hash):
        """Extract structured data from insurance policy text - simplest possible version"""
        
        # Build the simple prompt with the schema and example
        import json
        schema_json = json.dumps(INSURANCE_SCHEMA, indent=2)
        schema_description = json.dumps(SCHEMA_DESCRIPTION, indent=2)
        example_json = json.dumps(EXAMPLE_SCHEMA, indent=2)
        
        prompt = f"""
        
        You are an expert data extraction AI. Your job is to extract structured data from insurance policy documents and return the structured data in a JSON format, this must be formatted like this as it will be used by an api/other application.
        
        Extract all insurance policy data from this document according to this json schema:
        
        {schema_json}
        
        Here is a description of each of the fields of the schema for more context:
        
        {schema_description}
        
        Here is an example of properly filled out data in the exact structure required:
        
        {example_json}
        
        Follow this example and schema format closely but with the data you receive in this document. Return ONLY THE JSON with the extracted data. 
        FOR MISSING FIELDS: use "none" or "$0" for monetary values. Do not say "N/A" or "not applicable" or nill or null.
        Make sure to keep the exact nested structure shown in the schema and example and only output that. 
        
        If the policy states values, use those actual values instead of the text. Like if it says waiting period for some benefits, say the actual number of days instead of "waiting period may apply".
        
        Here is the most important part of this promptm, the insurnace policy document to perform the extraction on:
        {text}
        """
        
        # Make the API call
        try:
            response = self.model.generate_content(prompt)
            
            # Display the raw response
            st.subheader("Raw Model Response")
            st.text_area("JSON Output", response.text, height=400)
            
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
            st.error(f"Error calling Gemini API: {str(e)}")
            return {}