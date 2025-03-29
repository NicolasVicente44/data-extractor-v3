import re
from utils.validation import is_valid_value
from utils.formatting import format_value

def extract_values(chunks_with_scores, category, field):
    """
    Extract field values from text chunks using pattern matching
    
    Args:
        chunks_with_scores: List of tuples (chunk, similarity_score, chunk_idx)
        category: Insurance category
        field: Field name
        
    Returns:
        list: List of tuples (formatted_value, score, original_value, source_chunk)
    """
    base_patterns = {
        "dollar_amount": r"\$\s*([\d,]+(?:\.\d+)?)",
        "percentage": r"(\d{1,3}(?:\.\d{1,2})?)%(?:\s*of\s*(?:your)?(?:\s*(?:monthly|annual))?(?:\s*(?:earnings|salary|income)))?",
        "multiplier": r"(\d{1,2}(?:\.\d{1,2})?)(?:\s*|\-)(x|times)(?:\s*(?:your)?(?:\s*annual)?(?:\s*(?:earnings|salary|income)))?",
        "age": r"(?:age|at)\s*(\d{2,3})",
        "days": r"(\d{1,3})\s*(?:calendar|consecutive)?\s*(?:day|calendar day)s?",
        "weeks": r"(\d{1,3})\s*weeks?",
        "months": r"(\d{1,3})\s*months?",
        "years": r"(\d{1,3})\s*years?",
        "text_value": r"(?:is|are|:)\s*([A-Za-z0-9\s,]+)(?:\.|\n|$)",
    }

    # Generic field patterns
    field_patterns = {
        "Benefit Amount": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": base_patterns["multiplier"],
                "priority": 3,
                "format": "multiplier",
            },
            {
                "pattern": base_patterns["dollar_amount"],
                "priority": 2,
                "format": "currency",
            },
            {
                "pattern": r"benefit(?:\s*amount)?[\s:]*(\d{1,3}(?:\.\d{1,2})?)%",
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(\d{1,3})%\s*of\s*(?:your|the)?\s*(?:monthly|annual)?\s*(?:salary|earnings|income)",
                "priority": 4,
                "format": "percentage",
            },
            {
                "pattern": r"coverage(?:\s*amount)?[\s:]*(\d{1,2})\s*(?:x|times)",
                "priority": 3,
                "format": "multiplier",
            },
        ],
        "Termination Age": [
            {"pattern": base_patterns["age"], "priority": 3, "format": "age"},
            {
                "pattern": r"(?:terminates|ceases|ends)[\s:]*(?:at)?\s*age\s*(\d{2,3})",
                "priority": 4,
                "format": "age",
            },
            {
                "pattern": r"(?:to|until)\s*age\s*(\d{2,3})",
                "priority": 3,
                "format": "age",
            },
            {"pattern": r"age\s*(\d{2,3})", "priority": 2, "format": "age"},
        ],
        "Waiting Period": [
            {"pattern": base_patterns["days"], "priority": 4, "format": "days"},
            {"pattern": base_patterns["weeks"], "priority": 3, "format": "weeks"},
            {"pattern": base_patterns["months"], "priority": 3, "format": "months"},
            {
                "pattern": r"(?:waiting|elimination)\s*period[\s:]*(\d{1,3})\s*(?:calendar|consecutive|business)?\s*(?:day|week|month)s?",
                "priority": 4,
                "format": "auto",
            },
        ],
        "Maximum Benefit Period": [
            {"pattern": base_patterns["weeks"], "priority": 3, "format": "weeks"},
            {"pattern": base_patterns["months"], "priority": 3, "format": "months"},
            {"pattern": base_patterns["years"], "priority": 3, "format": "years"},
            {"pattern": r"to\s*age\s*(\d{2,3})", "priority": 4, "format": "to_age"},
            {
                "pattern": r"(?:maximum|benefit)\s*period[\s:]*(\d{1,3})\s*(?:week|month|year)s?",
                "priority": 4,
                "format": "auto",
            },
        ],
        "Definition of Disability": [
            {
                "pattern": r"own\s*occupation[^.]*?(?:for|period|of)?\s*(\d{1,3})\s*(?:month|year)s?",
                "priority": 4,
                "format": "own_occupation",
            },
            {
                "pattern": r"first\s*(\d{1,3})\s*(?:month|year)s?[^.]*?own\s*occupation",
                "priority": 4,
                "format": "own_occupation",
            },
            {
                "pattern": r"(?:definition|disabled|disability)[^.]*?((?:own|regular|any)\s*occupation)",
                "priority": 3,
                "format": "text",
            },
        ],
        "Covered Conditions": [
            {
                "pattern": r"(?:covered\s*conditions|covered\s*illnesses)[^:]*:([^.]+)",
                "priority": 4,
                "format": "conditions",
            },
            {
                "pattern": r"(?:conditions|illnesses)\s*(?:covered|included)(?:\s*are)?:([^.]+)",
                "priority": 3,
                "format": "conditions",
            },
        ],
        "Company Name": [
            {
                "pattern": r"(?:insurer|insurance\s*company|carrier)[^:]*:\s*([A-Z][A-Za-z\s&.,\']+)",
                "priority": 4,
                "format": "text",
            },
            {
                "pattern": r"(?:underwritten|issued)\s*by[^:]*:\s*([A-Z][A-Za-z\s&.,\']+)",
                "priority": 3,
                "format": "text",
            },
            {
                "pattern": r"([A-Z][A-Za-z\s\.,\']+(?:Inc\.|LLC|Ltd\.|Corporation|Company|Co\.))",
                "priority": 4,
                "format": "text",
            },
            {
                "pattern": r"([A-Z][A-Za-z\s]+(?:Insurance|Life|Financial)(?:\s[A-Za-z\s]+)?)",
                "priority": 4,
                "format": "text",
            },
            {
                "pattern": r"(?:your plan sponsor|sponsored by)\s+([A-Z][A-Za-z\s\.,\']+)",
                "priority": 5,
                "format": "text",
            },
            {
                "pattern": r"^([A-Z][A-Za-z\s\.,\']+(?:Inc\.|LLC|Ltd\.))\s+(?:is|has|Plan)",
                "priority": 5,
                "format": "text",
            },
            {
                "pattern": r"(?:from|by|with)\s+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})",
                "priority": 3,
                "format": "text",
            },
            {
                "pattern": r"selected\s+([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3})",
                "priority": 5,
                "format": "text",
            },
        ],
        "Policy Number": [
            {
                "pattern": r"policy\s*(?:number|no|#)[\s:]+([A-Z0-9][\w\-\.\/]{3,20})",
                "priority": 4,
                "format": "text",
            },
            {
                "pattern": r"certificate\s*(?:number|no|#)[\s:]+([A-Z0-9][\w\-\.\/]{3,20})",
                "priority": 3,
                "format": "text",
            },
            {
                "pattern": r"Plan\s*Number:?\s*([A-Z0-9][\w\-\.\/]{3,20})",
                "priority": 5,
                "format": "text",
            },
            {
                "pattern": r"(?:Group|Contract|Plan|Policy)\s*(?:Number|No\.|#)?\s*:?\s*([A-Z0-9][\w\-\.\/]{3,20})",
                "priority": 4,
                "format": "text",
            },
            {
                "pattern": r"(?:Group|Contract|Plan|Policy)\s*(?:Number|No\.|#)?\s*:?\s*([0-9]{5,7})",
                "priority": 4,
                "format": "text",
            },
            {"pattern": r"([G][0-9]{5,10})", "priority": 3, "format": "text"},
            {
                "pattern": r"(?:number|no|#)\s*([0-9]{5,7})",
                "priority": 3,
                "format": "text",
            },
        ],
        "Deductibles": [
            {
                "pattern": base_patterns["dollar_amount"],
                "priority": 3,
                "format": "currency",
            },
            {
                "pattern": r"deductible[\s:]*\$?\s*([\d,]+)",
                "priority": 4,
                "format": "currency",
            },
        ],
        "Drug Coverage": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(?:drug|prescription|medication)[^.]*?(?:covered|reimbursed|paid)[^.]*?(\d{1,3})%",
                "priority": 4,
                "format": "percentage",
            },
        ],
        "Dental Coverage": [
            {
                "pattern": base_patterns["percentage"],
                "priority": 3,
                "format": "percentage",
            },
            {
                "pattern": r"(?:dental|basic\s*dental)[^.]*?(?:covered|reimbursed|paid)[^.]*?(\d{1,3})%",
                "priority": 4,
                "format": "percentage",
            },
        ],
    }

    # Default patterns for fields without specific patterns
    default_patterns = [
        {
            "pattern": base_patterns["dollar_amount"],
            "priority": 2,
            "format": "currency",
        },
        {"pattern": base_patterns["percentage"], "priority": 2, "format": "percentage"},
        {
            "pattern": r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
            "priority": 1,
            "format": "numeric",
        },
        {"pattern": base_patterns["text_value"], "priority": 1, "format": "text"},
    ]

    # Get the appropriate patterns for this field
    field_specific_patterns = field_patterns.get(field, default_patterns)

    results = []

    # Context boost factors
    context_boost = {
        "field_in_line": 2.0,
        "category_in_context": 1.5,
        "table_header": 1.8,
    }

    # Process each chunk
    for chunk_data in chunks_with_scores:
        chunk, similarity_score, chunk_idx = chunk_data
        chunk_lower = chunk.lower()

        boost = 1.0
        field_lower = field.lower()
        category_lower = category.lower()

        # Check if field name appears in same line as potential values
        for line in chunk.split("\n"):
            line_lower = line.lower()
            if field_lower in line_lower:
                boost *= context_boost["field_in_line"]
                break

        # Check if category appears in context
        if category_lower in chunk_lower:
            boost *= context_boost["category_in_context"]

        # Check for table structure with this field/category
        table_header_pattern = (
            r"(?:"
            + re.escape(field_lower)
            + r"|"
            + re.escape(category_lower)
            + r")\s*(?:\||\t|:)"
        )
        if re.search(table_header_pattern, chunk_lower):
            boost *= context_boost["table_header"]

        # Try each pattern for this field
        for pattern_info in field_specific_patterns:
            pattern = pattern_info["pattern"]
            priority = pattern_info["priority"]
            format_type = pattern_info["format"]

            try:
                # Find all matches in the chunk
                matches = re.findall(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    # If match is a tuple (from capturing groups), take first non-empty group
                    if isinstance(match, tuple):
                        match = next((m for m in match if m), "")

                    # Clean up the match
                    match_value = match.strip()
                    if match_value:
                        # Skip very short values unless they're numbers
                        if len(match_value) < 2 and not match_value.isdigit():
                            continue

                        # Format the value based on the pattern type
                        original_value = match_value
                        formatted_value = format_value_from_pattern(
                            chunk, format_type, match_value
                        )

                        # Apply standard formatting for consistency
                        standardized_value = format_value(
                            formatted_value, field, category
                        )

                        # Calculate final score: similarity * boost * priority
                        final_score = similarity_score * boost * priority

                        # Validate the value
                        if is_valid_value(standardized_value, field, category):
                            results.append(
                                (standardized_value, final_score, original_value, chunk)
                            )

            except Exception:
                continue

    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates, keeping highest score
    unique_values = {}
    for formatted_value, score, original_value, source in results:
        if (
            formatted_value not in unique_values
            or score > unique_values[formatted_value]["score"]
        ):
            unique_values[formatted_value] = {
                "score": score,
                "original": original_value,
                "source": source
            }
    
    return [(value, data["score"], data["original"], data["source"]) 
            for value, data in unique_values.items()]


def format_value_from_pattern(value, format_type, match_value):
    """
    Format extracted value based on pattern format type
    
    Args:
        value: Original text context
        format_type: Type of format to apply
        match_value: Extracted match value
        
    Returns:
        str: Formatted value
    """
    try:
        if format_type == "percentage":
            percentage_match = re.search(r"(\d+(?:\.\d+)?)", match_value)
            if percentage_match:
                percentage = percentage_match.group(1)
                return f"{percentage}% of Salary"
            return (
                f"{match_value}% of Salary"
                if not "%" in match_value
                else match_value
            )

        elif format_type == "multiplier":
            multiplier_match = re.search(r"(\d+(?:\.\d+)?)", match_value)
            if multiplier_match:
                multiplier = multiplier_match.group(1)
                return f"{multiplier}x Annual Salary"
            return (
                f"{match_value}x Annual Salary"
                if not "x" in match_value.lower()
                else match_value
            )

        elif format_type == "currency":
            amount_match = re.search(r"([\d,]+(?:\.\d+)?)", match_value)
            if amount_match:
                amount_str = amount_match.group(1).replace(",", "")
                try:
                    amount = float(amount_str)
                    return f"${amount:,.2f}"
                except:
                    pass
            return f"${match_value}" if not "$" in match_value else match_value

        elif format_type == "age":
            age_match = re.search(r"(\d{2,3})", match_value)
            if age_match:
                age = age_match.group(1)
                return f"Age {age}"
            return match_value

        elif format_type == "days":
            return f"{match_value} days"

        elif format_type == "weeks":
            return f"{match_value} weeks"

        elif format_type == "months":
            return f"{match_value} months"

        elif format_type == "years":
            return f"{match_value} years"

        elif format_type == "auto":
            if "day" in value.lower():
                return f"{match_value} days"
            elif "week" in value.lower():
                return f"{match_value} weeks"
            elif "month" in value.lower():
                return f"{match_value} months"
            elif "year" in value.lower():
                return f"{match_value} years"
            return match_value

        elif format_type == "to_age":
            return f"To Age {match_value}"

        elif format_type == "own_occupation":
            time_unit = "months"
            if "year" in value.lower():
                time_unit = "years"
            return f"Own Occupation for {match_value} {time_unit}"

        elif format_type == "conditions":
            conditions = match_value.strip()
            conditions = re.sub(r"\s+", " ", conditions)
            return conditions

        return match_value
    except:
        return value


def generate_field_queries():
    """
    Generate comprehensive field queries for semantic search
    
    Returns:
        dict: Nested dictionary of queries by category and field
    """
    from config import INSURANCE_SCHEMA
    
    queries = {}

    # Generate standard queries for each field in schema
    for category, fields in INSURANCE_SCHEMA.items():
        queries[category] = {}
        for field in fields:
            # Generate multiple queries for each field with variants
            queries[category][field] = [
                f"{category} {field}",
                f"{field} {category}",
                f"{field}",
                f"{category} {field} insurance",
            ]

    # Add specific queries for common fields
    field_specific_queries = {
        "General Information": {
            "Company Name": [
                "insurance company name",
                "insurance carrier",
                "insurer name",
                "underwritten by",
                "provided by insurance company",
            ],
            "Policy Number": [
                "policy number",
                "certificate number",
                "group policy number",
                "contract number",
                "policy identification",
            ],
        },
        "Life Insurance": {
            "Benefit Amount": [
                "life insurance benefit amount",
                "basic life coverage amount",
                "life insurance coverage",
                "group life insurance benefit",
                "life benefit amount",
            ],
            "Termination Age": [
                "life insurance termination age",
                "coverage ceases at age",
                "life benefits terminate",
                "life insurance terminates",
                "life coverage ends",
            ],
        },
        "STD": {
            "Benefit Amount": [
                "short-term disability benefit amount",
                "STD benefit percentage",
                "weekly indemnity amount",
                "short term disability pays",
                "STD coverage amount",
            ],
            "Waiting Period": [
                "STD waiting period",
                "short-term disability elimination period",
                "days before STD begins",
                "elimination period STD",
                "when STD benefits begin",
            ],
        },
        "LTD": {
            "Benefit Amount": [
                "long-term disability benefit amount",
                "LTD benefit percentage",
                "monthly disability benefit",
                "LTD pays",
                "long term disability amount",
            ],
            "Waiting Period": [
                "LTD waiting period",
                "long-term disability elimination period",
                "days before LTD begins",
                "elimination period LTD",
                "when LTD benefits begin",
            ],
        },
    }

    # Merge field-specific queries into main queries
    for category, fields in field_specific_queries.items():
        if category in queries:
            for field, additional_queries in fields.items():
                if field in queries[category]:
                    queries[category][field].extend(additional_queries)

    return queries