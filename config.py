import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_DB_PATH = os.path.join(CACHE_DIR, "feedback_db.pkl")

INSURANCE_SCHEMA = {
    "Company Name": "",
    "Policy #": "",
    "General Provisions": {
        "Eligibility Period": "",
        "Definition of Salary": "",
        "Child Coverage - Under Age": "",
        "Student Extension - Under Age": ""
    },
    "Life Insurance and A.D. & D.": {
        "Benefit Amount - Life insurance": "$0",
        "Non-evidence Maximum": "$0",
        "Overall Maximum": "$0",
        "Reduction Schedule": "",
        "Termination Age": ""
    },
    "Optional Life Insurance": {
        "Benefit Amount - Life insurance": "$0",
        "Non-evidence Maximum": "$0",
        "Overall Maximum": "$0",
        "Termination Age": ""
    },
    "Dependent Life Insurance": {
        "Spouse Amount/Child Amount": "$0",
        "Children Covered from Age": "",
        "Termination Age": ""
    },
    "Short Term Disability": {
        "Benefit Amount": "$0",
        "Non-evidence Maximum": "$0",
        "Overall Maximum": "$0",
        "Waiting Period - Accident/Sickness/Hospitalization": "",
        "Maximum Benefit Period": "",
        "Termination Age": "",
        "Taxability of Benefits": ""
    },
    "Long Term Disability": {
        "Benefit Amount": "$0",
        "Non-evidence Maximum": "$0",
        "Overall Maximum": "$0",
        "Waiting Period": "",
        "Definition of Disability": "",
        "Maximum Benefit Period": "",
        "COLA": "",
        "Termination Age": "",
        "Taxability of Benefits": ""
    },
    "Critical Illness": {
        "Benefit Amount": "$0",
        "Covered Conditions": "",
        "Multi Occurrence": "",
        "Dependent Coverage": "",
        "Termination Age": ""
    },
    "Health Care": {
        "Preferred Provider Arrangement": "",
        "Deductible - Drugs/Dispensing Fee Cap": "$0",
        "Deductible - Other Health Care": "$0",
        "Co-insurance - Drugs/Prof. Services/Other Health Care": "",
        "Drug Coverage": ""
    }
}

SCHEMA_DESCRIPTION = {
    "Company Name": "Name of the company or employer offering the policy. It is NOT the insurance provider name, but instead the name of the company usually in the first few pages of the policy. Format: String. If unavailable, use 'none'.",
    "Policy #": "Unique identifier for the insurance policy. Format: Alphanumeric string. If unavailable, use 'none'.",
    "General Provisions": {
      "Eligibility Period": "Timeframe before an individual becomes eligible for coverage. Format: Numeric value with units (e.g., '30 days', '3 months'). If unavailable, use 'none'. May be multiple for different coverage types, so if they are all the same, list just one number, if they are different, list them all with the coverage type they apply to. For example: '30 days for life insurance, 60 days for dental care'.",
      "Definition of Salary": "Explanation of how salary is calculated for insurance purposes. Format: String. If unavailable, use 'none'.",
      "Child Coverage - Under Age": "Maximum age under which children are covered. Format: Numeric value. If unavailable, use 'none'.",
      "Student Extension - Under Age": "Maximum age extension for students covered under the plan. Format: Numeric value. If unavailable, use 'none'."
    },
    "Life Insurance and A.D. & D.": {
      "Benefit Amount - Life insurance": "Amount paid out for life insurance. Format: Currency (e.g., '$50,000'). If unavailable, use '$0'.",
      "Non-evidence Maximum": "Maximum coverage amount without requiring medical proof. Format: Currency. If unavailable, use '$0'.",
      "Overall Maximum": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
      "Reduction Schedule": "Details of how coverage reduces over time. Format: String. If unavailable, use 'none'.",
      "Termination Age": "Age at which coverage ends. Format: Numeric value. If unavailable, use 'none'."
    },
    "Optional Life Insurance": {
      "Benefit Amount - Life insurance": "Amount of optional life insurance available. Format: Currency. If unavailable, use '$0'.",
      "Non-evidence Maximum": "Maximum optional coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
      "Overall Maximum": "Total optional coverage limit. Format: Currency. If unavailable, use '$0'.",
      "Termination Age": "Age at which optional life insurance ends. Format: Numeric value. If unavailable, use 'none'."
    },
    "Dependent Life Insurance": {
      "Spouse Amount/Child Amount": "Life insurance coverage for spouse/children. Format: Currency. If unavailable, use '$0'.",
      "Children Covered from Age": "Minimum age for child coverage. Format: Numeric value. If unavailable, use 'none'.",
      "Termination Age": "Age at which dependent coverage ends. Format: Numeric value. If unavailable, use 'none'."
    },
    "Short Term Disability": {
      "Benefit Amount": "Amount paid during short-term disability. Format: Currency. If unavailable, use '$0'.",
      "Non-evidence Maximum": "Maximum coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
      "Overall Maximum": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
      "Waiting Period - Accident/Sickness/Hospitalization": "Time before benefits start. Format: Numeric value with units. If unavailable, use 'none'.",
      "Maximum Benefit Period": "Duration of benefits. Format: Numeric value with units. If unavailable, use 'none'.",
      "Termination Age": "Age at which short-term disability coverage ends. Format: Numeric value. If unavailable, use 'none'.",
      "Taxability of Benefits": "Whether benefits are taxable. Format: 'Taxable' or 'Non-taxable'. If unavailable, use 'none'."
    },
    "Long Term Disability": {
      "Benefit Amount": "Amount paid during long-term disability. Format: Currency. If unavailable, use '$0'.",
      "Non-evidence Maximum": "Maximum coverage without medical proof. Format: Currency. If unavailable, use '$0'.",
      "Overall Maximum": "Total coverage limit. Format: Currency. If unavailable, use '$0'.",
      "Waiting Period": "Time before long-term disability benefits start. Format: Numeric value with units. If unavailable, use 'none'.",
      "Definition of Disability": "Definition of what qualifies as a disability. Format: String. If unavailable, use 'none'.",
      "Maximum Benefit Period": "Duration of long-term disability benefits. Format: Numeric value with units. If unavailable, use 'none'.",
      "COLA": "Cost of Living Adjustment details. Format: String. If unavailable, use 'none'.",
      "Termination Age": "Age at which long-term disability coverage ends. Format: Numeric value. If unavailable, use 'none'.",
      "Taxability of Benefits": "Whether benefits are taxable. Format: 'Taxable' or 'Non-taxable'. If unavailable, use 'none'."
    },
    "Critical Illness": {
      "Benefit Amount": "Payout amount for critical illness. Format: Currency. If unavailable, use '$0'.",
      "Covered Conditions": "List of illnesses covered. Format: String. If unavailable, use 'none'.",
      "Multi Occurrence": "Whether multiple claims are allowed. Format: 'Yes' or 'No'. If unavailable, use 'none'.",
      "Dependent Coverage": "Whether dependents are covered. Format: 'Yes' or 'No'. If unavailable, use 'none'.",
      "Termination Age": "Age at which critical illness coverage ends. Format: Numeric value. If unavailable, use 'none'."
    },
    "Health Care": {
      "Preferred Provider Arrangement": "Network of preferred providers. Format: String. If unavailable, use 'none'.",
      "Deductible - Drugs/Dispensing Fee Cap": "Annual deductible for drug costs. Format: Currency. If unavailable, use '$0'.",
      "Deductible - Other Health Care": "Annual deductible for other health services. Format: Currency. If unavailable, use '$0'.",
      "Co-insurance - Drugs/Prof. Services/Other Health Care": "Percentage covered by insurance. Format: Numeric percentage (e.g., '80%'). If unavailable, use 'none'.",
      "Drug Coverage": "Description of drug coverage. Format: String. If unavailable, use 'none'."
    }
}



