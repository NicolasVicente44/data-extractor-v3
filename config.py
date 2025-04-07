import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_DB_PATH = os.path.join(CACHE_DIR, "feedback_db.pkl")

# Modified schema to include source references for each value
INSURANCE_SCHEMA = {
    "Company Name": {"value": "", "source": ""},
    "Policy #": {"value": "", "source": ""},
    "General Provisions": {
        "Eligibility Period": {"value": "", "source": ""},
        "Definition of Salary": {"value": "", "source": ""},
        "Child Coverage - Under Age": {"value": "", "source": ""},
        "Student Extension - Under Age": {"value": "", "source": ""}
    },
    "Life Insurance and A.D. & D.": {
        "Benefit Amount - Life insurance": {"value": "$0", "source": ""},
        "Non-evidence Maximum": {"value": "$0", "source": ""},
        "Overall Maximum": {"value": "$0", "source": ""},
        "Reduction Schedule": {"value": "", "source": ""},
        "Termination Age": {"value": "", "source": ""}
    },
    "Optional Life Insurance": {
        "Benefit Amount - Life insurance": {"value": "$0", "source": ""},
        "Non-evidence Maximum": {"value": "$0", "source": ""},
        "Overall Maximum": {"value": "$0", "source": ""},
        "Termination Age": {"value": "", "source": ""}
    },
    "Dependent Life Insurance": {
        "Spouse Amount/Child Amount": {"value": "$0", "source": ""},
        "Children Covered from Age": {"value": "", "source": ""},
        "Termination Age": {"value": "", "source": ""}
    },
    "Short Term Disability": {
        "Benefit Amount": {"value": "$0", "source": ""},
        "Non-evidence Maximum": {"value": "$0", "source": ""},
        "Overall Maximum": {"value": "$0", "source": ""},
        "Waiting Period - Accident/Sickness/Hospitalization": {"value": "", "source": ""},
        "Maximum Benefit Period": {"value": "", "source": ""},
        "Termination Age": {"value": "", "source": ""},
        "Taxability of Benefits": {"value": "", "source": ""}
    },
    "Long Term Disability": {
        "Benefit Amount": {"value": "$0", "source": ""},
        "Non-evidence Maximum": {"value": "$0", "source": ""},
        "Overall Maximum": {"value": "$0", "source": ""},
        "Waiting Period": {"value": "", "source": ""},
        "Definition of Disability": {"value": "", "source": ""},
        "Maximum Benefit Period": {"value": "", "source": ""},
        "COLA": {"value": "", "source": ""},
        "Termination Age": {"value": "", "source": ""},
        "Taxability of Benefits": {"value": "", "source": ""}
    },
    "Critical Illness": {
        "Benefit Amount": {"value": "$0", "source": ""},
        "Covered Conditions": {"value": "", "source": ""},
        "Multi Occurrence": {"value": "", "source": ""},
        "Dependent Coverage": {"value": "", "source": ""},
        "Termination Age": {"value": "", "source": ""}
    },
    "Health Care": {
        "Preferred Provider Arrangement": {"value": "", "source": ""},
        "Deductible - Drugs/Dispensing Fee Cap": {"value": "$0", "source": ""},
        "Deductible - Other Health Care": {"value": "$0", "source": ""},
        "Co-insurance - Drugs/Prof. Services/Other Health Care": {"value": "", "source": ""},
        "Drug Coverage": {"value": "", "source": ""}
    }
}

# Update the schema description to explain the source field
SCHEMA_DESCRIPTION = {
    "Company Name": {
        "value": "Name of the company or employer offering the policy. It is NOT the insurance provider name, but instead the name of the company usually in the first few pages of the policy. Format: String. If unavailable, use 'none'.",
        "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
    },
    "Policy #": {
        "value": "Unique identifier for the insurance policy. Format: Alphanumeric string. If unavailable, use 'none'.",
        "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
    },
    "General Provisions": {
        "Eligibility Period": {
            "value": "Timeframe before an individual becomes eligible for coverage. Format: Numeric value with units (e.g., '30 days', '3 months'). If unavailable, use 'none'. May be multiple for different coverage types, so if they are all the same, list just one number, if they are different, list them all with the coverage type they apply to. For example: '30 days for life insurance, 60 days for dental care'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Definition of Salary": {
            "value": "Explanation of how salary is calculated for insurance purposes. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Child Coverage - Under Age": {
            "value": "Maximum age under which children are covered. Ensure you specify the maximum age of coverage. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Student Extension - Under Age": {
            "value": "Maximum age extension for students covered under the plan. Ensure you specify the maximum age of coverage. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Life Insurance and A.D. & D.": {
        "Benefit Amount - Life insurance": {
            "value": "Amount paid out for life insurance. Format: Currency (e.g., '$50,000'). If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Non-evidence Maximum": {
            "value": "Maximum coverage amount without requiring medical proof. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Overall Maximum": {
            "value": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Reduction Schedule": {
            "value": "Details of how coverage reduces over time. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which coverage ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Optional Life Insurance": {
        "Benefit Amount - Life insurance": {
            "value": "Amount of optional life insurance available. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Non-evidence Maximum": {
            "value": "Maximum optional coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Overall Maximum": {
            "value": "Total optional coverage limit. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which optional life insurance ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Dependent Life Insurance": {
        "Spouse Amount/Child Amount": {
            "value": "Life insurance coverage for spouse/children. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Children Covered from Age": {
            "value": "Minimum age for child coverage. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which dependent coverage ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Short Term Disability": {
        "Benefit Amount": {
            "value": "Amount paid during short-term disability. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Non-evidence Maximum": {
            "value": "Maximum coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Overall Maximum": {
            "value": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Waiting Period - Accident/Sickness/Hospitalization": {
            "value": "Time before benefits start. Format: Numeric value with units. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Maximum Benefit Period": {
            "value": "Duration of benefits. Format: Numeric value with units. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which short-term disability coverage ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Taxability of Benefits": {
            "value": "Whether benefits are taxable. Format: 'Taxable' or 'Non-taxable'. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Long Term Disability": {
        "Benefit Amount": {
            "value": "Amount paid during long-term disability. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Non-evidence Maximum": {
            "value": "Maximum coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Overall Maximum": {
            "value": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Waiting Period": {
            "value": "Time before long-term disability benefits start. Format: Numeric value with units. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Definition of Disability": {
            "value": "Definition of what qualifies as a disability. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Maximum Benefit Period": {
            "value": "Duration of long-term disability benefits. Format: Numeric value with units. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "COLA": {
            "value": "Cost of Living Adjustment details. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which long-term disability coverage ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Taxability of Benefits": {
            "value": "Whether benefits are taxable. Format: 'Taxable' or 'Non-taxable'. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Critical Illness": {
        "Benefit Amount": {
            "value": "Payout amount for critical illness. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Covered Conditions": {
            "value": "List of illnesses covered. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Multi Occurrence": {
            "value": "Whether multiple claims are allowed. Format: 'Yes' or 'No'. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Dependent Coverage": {
            "value": "Whether dependents are covered. Format: 'Yes' or 'No'. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Termination Age": {
            "value": "Age at which critical illness coverage ends. Format: Numeric value. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    },
    "Health Care": {
        "Preferred Provider Arrangement": {
            "value": "Network of preferred providers. Could be a telehealth type company. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Deductible - Drugs/Dispensing Fee Cap": {
            "value": "Annual deductible for drug costs. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Deductible - Other Health Care": {
            "value": "Annual deductible for other health services. Format: Currency. If unavailable, use '$0'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Co-insurance - Drugs/Prof. Services/Other Health Care": {
            "value": "Percentage covered by insurance. Format: Numeric percentage (e.g., '80%'). If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        },
        "Drug Coverage": {
            "value": "Description of drug coverage. Format: String. If unavailable, use 'none'.",
            "source": "Extract approximately 100 words of text surrounding where this information was found in the document. Include page number if available."
        }
    }
}

# Instructions for the AI model regarding sources
SOURCE_INSTRUCTIONS = """
For each extracted data point, provide:
1. The actual value in the "value" field
2. A contextual excerpt of approximately 100 words from the source document in the "source" field that contains this information
3. If no information is found, set value to default (empty string or $0 as specified) and source to "Information not found in document"
4. Always include page numbers when available in the source reference (format: "Page X: [excerpt...]")
5. If the information spans multiple pages, indicate this in the source (format: "Pages X-Y: [excerpt...]")
6. For tables or structured information, describe the table location along with the excerpt
"""