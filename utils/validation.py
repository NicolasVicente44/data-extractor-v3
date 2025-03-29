import re
from config import VALUE_FORMAT_SPECS
from utils.formatting import detect_field_type

def is_valid_value(value, field, category=None):
    """
    Validate a field value based on field type and expected constraints
    
    Args:
        value: Value to validate
        field: Field name for context
        category: Category name for context
        
    Returns:
        bool: True if value is valid, False otherwise
    """
    try:
        field_type, extracted_value = detect_field_type(value, field)

        if (
            field_type in VALUE_FORMAT_SPECS
            and "validation" in VALUE_FORMAT_SPECS[field_type]
        ):
            try:
                return VALUE_FORMAT_SPECS[field_type]["validation"](extracted_value)
            except:
                pass

        if field == "Benefit Amount":
            if "%" in value:
                percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", value, re.IGNORECASE)
                if percent_match:
                    percentage = float(percent_match.group(1))
                    return percentage <= 100

            elif "x" in value.lower() or "times" in value.lower():
                multiplier_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:x|times)", value.lower()
                )
                if multiplier_match:
                    multiplier = float(multiplier_match.group(1))
                    if category == "Life Insurance":
                        return 0.5 <= multiplier <= 5
                    else:
                        return multiplier <= 10

        elif field == "Termination Age":
            age_match = re.search(r"(?:Age|age)\s*(\d{2,3})", value)
            if age_match:
                age = int(age_match.group(1))
                if 55 <= age <= 100:
                    return True
                return False

        elif field == "Waiting Period":
            if category == "STD":
                day_match = re.search(r"(\d+)\s*days?", value, re.IGNORECASE)
                if day_match:
                    days = int(day_match.group(1))
                    return 0 <= days <= 180

            elif category == "LTD":
                day_match = re.search(r"(\d+)\s*days?", value, re.IGNORECASE)
                if day_match:
                    days = int(day_match.group(1))
                    return 30 <= days <= 365

                month_match = re.search(r"(\d+)\s*months?", value, re.IGNORECASE)
                if month_match:
                    months = int(month_match.group(1))
                    return 1 <= months <= 12

        return True
    except Exception:
        return True


def apply_cross_field_validation(extracted_values):
    """
    Apply cross-field validation to check for consistency between related fields
    
    Args:
        extracted_values: Dictionary of extracted values by category and field
        
    Returns:
        dict: Updated extracted values
    """
    field_relations = [
        # If life is X times salary, optional life might be similar
        (
            "Life Insurance,Benefit Amount",
            "Optional Life Insurance,Benefit Amount",
            lambda v1, v2: (
                v1 if "Annual Salary" in v1 and v2 == "Not Found" and "x" in v1 else v2
            ),
        ),
        # STD and LTD termination ages are often the same
        (
            "STD,Termination Age",
            "LTD,Termination Age",
            lambda v1, v2: (
                v1
                if v1 != "Not Found"
                and v2 == "Not Found"
                and re.search(r"Age \d{2}", v1)
                else v2
            ),
        ),
        # LTD waiting period often matches STD maximum benefit period
        (
            "STD,Maximum Benefit Period",
            "LTD,Waiting Period",
            lambda v1, v2: (
                v1
                if v1 != "Not Found"
                and v2 == "Not Found"
                and any(unit in v1.lower() for unit in ["week", "day", "month"])
                else v2
            ),
        ),
        # Various termination ages are often the same
        (
            "Life Insurance,Termination Age",
            "Critical Illness,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        # Health and Dental often have the same termination age
        (
            "Health Coverage,Termination Age",
            "Dental Benefits,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        # Other common termination ages
        (
            "Life Insurance,Termination Age",
            "STD,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
        (
            "Life Insurance,Termination Age",
            "LTD,Termination Age",
            lambda v1, v2: (
                v1 if v1 != "Not Found" and v2 == "Not Found" and "Age" in v1 else v2
            ),
        ),
    ]

    # Apply field relation rules
    for relation in field_relations:
        source_path, target_path, rule_func = relation
        source_category, source_field = source_path.split(",")

        # Skip if category is "sources"
        if source_category == "sources":
            continue

        if (
            source_category in extracted_values
            and source_field in extracted_values[source_category]
        ):
            source_value = extracted_values[source_category][source_field]

            if target_path:
                target_category, target_field = target_path.split(",")

                # Skip if category is "sources"
                if target_category == "sources":
                    continue

                if (
                    target_category in extracted_values
                    and target_field in extracted_values[target_category]
                ):
                    target_value = extracted_values[target_category][target_field]
                    new_target_value = rule_func(source_value, target_value)
                    extracted_values[target_category][target_field] = new_target_value

    # Apply standard formatting to all fields
    from utils.formatting import format_value
    
    for category in list(extracted_values.keys()):
        # Skip the sources dictionary
        if category == "sources":
            continue
            
        for field in extracted_values[category]:
            value = extracted_values[category][field]

            if value == "Not Found":
                continue

            standardized_value = format_value(value, field, category)
            extracted_values[category][field] = standardized_value

    return extracted_values