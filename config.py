import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_DB_PATH = os.path.join(CACHE_DIR, "feedback_db.pkl")

# New Insurance Schema based on provided JSON
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

# Schema description for Gemini prompt (more detailed)
SCHEMA_DESCRIPTION = {
    "Company Name": "Name of the company or employer offering the policy. It is not the insurance provider name, usually in the first few pages of the policy. Format: String. If unavailable, use 'none'.",
    "Policy #": "Unique identifier for the insurance policy. Format: Alphanumeric string. If unavailable, use 'none'.",
    "General Provisions": {
      "Eligibility Period": "Timeframe before an individual becomes eligible for coverage. Format: Numeric value with units (e.g., '30 days'). If unavailable, use 'none'.",
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