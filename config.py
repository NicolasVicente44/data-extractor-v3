import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data")
FEEDBACK_DB_PATH = os.path.join(CACHE_DIR, "feedback_db.pkl")

# Insurance Schema definition
INSURANCE_SCHEMA = {
    "General Information": ["Company Name", "Policy Number"],
    "Benefit Summary": [
        "Benefit Amount",
        "Eligibility Period",
        "Definition of Salary",
        "Child Coverage",
        "Student Extension",
    ],
    "Life Insurance": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Reduction Schedule",
        "Termination Age",
    ],
    "Optional Life Insurance": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Termination Age",
        "Spouse Amount",
        "Child Amount",
    ],
    "STD": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Waiting Period",
        "Maximum Benefit Period",
        "Termination Age",
    ],
    "LTD": [
        "Benefit Amount",
        "Non-Evidence Maximum",
        "Overall Maximum",
        "Waiting Period",
        "Definition of Disability",
        "Maximum Benefit Period",
        "COLA",
        "Termination Age",
    ],
    "Critical Illness": [
        "Benefit Amount",
        "Covered Conditions",
        "Multi-Occurrence",
        "Dependent Coverage",
        "Termination Age",
    ],
    "Health Coverage": [
        "Preferred Provider",
        "Deductibles",
        "Drug Coverage",
        "Preventative Services",
        "Out-of-Province Coverage",
        "Hospital Care",
        "Termination Age",
    ],
    "Dental Benefits": [
        "Dental Coverage",
        "Employee Assistance Plan",
        "Virtual Health Care",
        "Termination Age",
    ],
}

# Value format specifications
VALUE_FORMAT_SPECS = {
    "percentage": {
        "pattern": r"(\d+(?:\.\d+)?)%",
        "format": "{0}% of Salary",
        "validation": lambda x: 0 <= float(x) <= 100,
    },
    "multiplier": {
        "pattern": r"(\d+(?:\.\d+)?)x",
        "format": "{0}x Annual Salary",
        "validation": lambda x: 0 <= float(x) <= 10,
    },
    "currency": {
        "pattern": r"\$?([\d,]+(?:\.\d+)?)",
        "format": "${0:,.2f}",
        "validation": lambda x: float(x.replace(",", "")) >= 0,
    },
    "age": {
        "pattern": r"(?:age|to age)?\s*(\d{2,3})",
        "format": "Age {0}",
        "validation": lambda x: 0 <= int(x) <= 100,
    },
    "days": {
        "pattern": r"(\d+)\s*days?",
        "format": "{0} days",
        "validation": lambda x: 0 <= int(x) <= 365,
    },
    "weeks": {
        "pattern": r"(\d+)\s*weeks?",
        "format": "{0} weeks",
        "validation": lambda x: 0 <= int(x) <= 52,
    },
    "months": {
        "pattern": r"(\d+)\s*months?",
        "format": "{0} months",
        "validation": lambda x: 0 <= int(x) <= 48,
    },
    "years": {
        "pattern": r"(\d+)\s*years?",
        "format": "{0} years",
        "validation": lambda x: 0 <= int(x) <= 20,
    },
    "to_age": {
        "pattern": r"to\s*age\s*(\d{2,3})",
        "format": "To Age {0}",
        "validation": lambda x: 0 <= int(x) <= 100,
    },
}