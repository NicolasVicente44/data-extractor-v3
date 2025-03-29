import os
import streamlit as st
import google.generativeai as genai
from config import INSURANCE_SCHEMA, SCHEMA_DESCRIPTION

# Example schema to show the structure and format
EXAMPLE_SCHEMA = {
  "Company Name": "Tru-Max Security Inc.",
  "Policy #": "G0137421",
  "General Provisions": {
    "Eligibility Period": "waiting period may apply for some benefits",
    "Definition of Salary": "not defined in document",
    "Child Coverage - Under Age": "21 (for Dental coverage), 21 (for Extended Health Care coverage)",
    "Student Extension - Under Age": "later of the age specified in the booklet or age 26"
  },
  "Life Insurance and A.D. & D.": {
    "Benefit Amount - Life insurance": "1 times your annual earnings, to a maximum of $1,000,000",
    "Non-evidence Maximum": "$150,000",
    "Overall Maximum": "$1,000,000",
    "Reduction Schedule": "Your benefit amount reduces by 50% at age 65",
    "Termination Age": "71"
  },
  "Optional Life Insurance": {
    "Benefit Amount - Life insurance": "Contact plan administrator",
    "Non-evidence Maximum": "Contact plan administrator",
    "Overall Maximum": "Contact plan administrator",
    "Termination Age": "Contact plan administrator"
  },
  "Dependent Life Insurance": {
    "Spouse Amount/Child Amount": "$10,000 for your spouse and $5,000 for each dependent child",
    "Children Covered from Age": "newborn child shall become eligible from the moment of birth",
    "Termination Age": "71"
  },
  "Short Term Disability": {
    "Benefit Amount": "Not covered under this plan",
    "Non-evidence Maximum": "Not covered under this plan",
    "Overall Maximum": "Not covered under this plan",
    "Waiting Period - Accident/Sickness/Hospitalization": "Not covered under this plan",
    "Maximum Benefit Period": "Not covered under this plan",
    "Termination Age": "Not covered under this plan",
    "Taxability of Benefits": "Not covered under this plan"
  },
  "Long Term Disability": {
    "Benefit Amount": "66.7% of your first $2,500 of monthly earnings, plus 50% of the next $3,500 of monthly earnings, plus 40% of any excess amount, to a maximum of $10,000",
    "Non-evidence Maximum": "$2,400",
    "Overall Maximum": "$10,000",
    "Waiting Period": "3 months",
    "Definition of Disability": "Totally Disabled means a restriction or lack of ability due to an illness or injury which prevents you from performing the essential duties of: • your own occupation, during the Qualifying Period and the 2 years immediately following the Qualifying Period • any occupation for which you are qualified, or may reasonably become qualified, by training, education or experience, after the 2 years specified above",
    "Maximum Benefit Period": "to age 65 for Total Disability Benefits",
    "COLA": "not applicable",
    "Termination Age": "Age 65 less the Qualifying Period, or your retirement, whichever is earlier",
    "Taxability of Benefits": "The tax position of any payments you receive under this benefit depends on whether you or your employer pays the cost of the benefit."
  },
  "Critical Illness": {
    "Benefit Amount": "Not covered under this plan",
    "Covered Conditions": "Not covered under this plan",
    "Multi Occurrence": "Not covered under this plan",
    "Dependent Coverage": "Not covered under this plan",
    "Termination Age": "Not covered under this plan"
  },
  "Health Care": {
    "Preferred Provider Arrangement": "none",
    "Deductible - Drugs/Dispensing Fee Cap": "$0",
    "Deductible - Other Health Care": "Nil",
    "Co-insurance - Drugs/Prof. Services/Other Health Care": "90% for Hospital Care, Vision, Drugs, Medical Services & Supplies, Professional Services",
    "Drug Coverage": "$8.00 per prescription dispensing fee maximum. For maintenance drugs, no more than 6 dispensing fees will be paid per 12 consecutive months. If you are a Quebec resident, your plan's coverage will coordinate with RAMQ."
  }
}

class GeminiFlashExtractor:
    """
    Dead simple insurance policy data extractor using Google's Gemini 2.0 Flash model
    """
    
    def __init__(self):
        """Initialize the Gemini extractor with API key from environment variable"""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        
        # Use the correct model name for Gemini 2.0 Flash
        self.model_name = 'gemini-2.0-flash'
        self.model = genai.GenerativeModel(self.model_name)
        st.info(f"Using {self.model_name} for extraction")
        
    def extract_data(self, text, page_texts, document_hash):
        """Extract structured data from insurance policy text - simplest possible version"""
        st.write("Sending document to Gemini 2.0 Flash...")
        
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
        For missing fields use "none" or "$0" for monetary values.
        Make sure to keep the exact nested structure shown in the schema and example and only output that.
        
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