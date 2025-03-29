import re
from config import VALUE_FORMAT_SPECS

def detect_field_type(value, field):
    """
    Detects the type of value (percentage, multiplier, currency, etc.)
    
    Args:
        value: String value to analyze
        field: Field name for context
        
    Returns:
        tuple: (field_type, extracted_value)
    """
    lower_value = value.lower()

    if "%" in lower_value:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", lower_value)
        if match:
            return "percentage", match.group(1)

    if "x" in lower_value or "times" in lower_value:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)", lower_value)
        if match:
            return "multiplier", match.group(1)

    if "$" in lower_value or re.search(r"\d+,\d{3}", lower_value):
        match = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", lower_value)
        if match:
            return "currency", match.group(1).replace(",", "")

    if "age" in lower_value or field == "Termination Age":
        match = re.search(r"(?:age|to age)?\s*(\d{2,3})", lower_value)
        if match:
            return "age", match.group(1)

    if "day" in lower_value:
        match = re.search(r"(\d+)\s*days?", lower_value)
        if match:
            return "days", match.group(1)

    if "week" in lower_value:
        match = re.search(r"(\d+)\s*weeks?", lower_value)
        if match:
            return "weeks", match.group(1)

    if "month" in lower_value:
        match = re.search(r"(\d+)\s*months?", lower_value)
        if match:
            return "months", match.group(1)

    if "year" in lower_value:
        match = re.search(r"(\d+)\s*years?", lower_value)
        if match:
            return "years", match.group(1)

    match = re.search(r"(\d+(?:\.\d+)?)", lower_value)
    if match:
        return "numeric", match.group(1)

    return "text", value.strip()

def format_value(value, field, category):
    """
    Format an extracted value consistently based on its type and context
    
    Args:
        value: Value to format
        field: Field name for context
        category: Category for context
        
    Returns:
        str: Formatted value
    """
    if value == "Not Found":
        return value

    field_type, extracted_value = detect_field_type(value, field)

    if field_type in VALUE_FORMAT_SPECS:
        format_spec = VALUE_FORMAT_SPECS[field_type]
        try:
            if "validation" in format_spec and format_spec["validation"](
                extracted_value
            ):
                return format_spec["format"].format(extracted_value)
        except:
            pass

    if field == "Benefit Amount":
        if field_type == "percentage":
            return f"{extracted_value}% of Salary"
        elif field_type == "multiplier":
            return f"{extracted_value}x Annual Salary"
        elif field_type == "currency":
            try:
                amount = float(extracted_value)
                return f"${amount:,.2f}"
            except:
                return value

    elif field == "Waiting Period":
        if field_type in ["days", "weeks", "months"]:
            unit = field_type
            return f"{extracted_value} {unit}"

    elif field == "Termination Age":
        if field_type == "age":
            return f"Age {extracted_value}"
        elif field_type == "to_age":
            return f"To Age {extracted_value}"

    elif field == "Maximum Benefit Period":
        if field_type in ["weeks", "months", "years"]:
            unit = field_type
            return f"{extracted_value} {unit}"
        elif field_type == "to_age":
            return f"To Age {extracted_value}"

    return value