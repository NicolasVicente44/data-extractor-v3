import re
import streamlit as st


class DocumentStructure:
    """
    Analyzes the structure of an insurance document to extract meaningful patterns
    """

    def __init__(self, text, page_texts):
        self.text = text
        self.page_texts = page_texts
        self.structure_info = self._analyze_document_structure()

    def _analyze_document_structure(self):
        """
        Analyze document structure to identify document type, format, sections and fields
        """
        structure_info = {
            "document_type": "unknown",
            "format_type": "unknown",
            "section_markers": [],
            "table_regions": [],
            "field_locations": {},
        }

        # Try to identify document type
        doc_type_patterns = [
            (
                "booklet",
                [r"\bbooklet\b", r"\bemployee\s+booklet\b", r"\bbenefit\s+booklet\b"],
            ),
            (
                "summary",
                [
                    r"\bsummary\b",
                    r"\bbenefits?\s+summary\b",
                    r"\bsummary\s+of\s+benefits\b",
                ],
            ),
            ("certificate", [r"\bcertificate\b", r"\binsurance\s+certificate\b"]),
            (
                "policy",
                [r"\bpolicy\b", r"\binsurance\s+policy\b", r"\bgroup\s+policy\b"],
            ),
        ]

        for doc_type, patterns in doc_type_patterns:
            for pattern in patterns:
                if re.search(pattern, self.text, re.IGNORECASE):
                    structure_info["document_type"] = doc_type
                    break
            if structure_info["document_type"] != "unknown":
                break

        # Identify format type
        if len(re.findall(r"[\.]{4,}", self.text)) > 5:
            structure_info["format_type"] = "dot_leader"
        elif len(re.findall(r"[^\n]+\n[^\n]+", self.text)) > len(self.text) / 300:
            structure_info["format_type"] = "line_break"
        elif len(re.findall(r":\s+", self.text)) > len(self.text) / 400:
            structure_info["format_type"] = "colon_label"
        elif len(re.findall(r"\|\s+\|", self.text)) > 5:
            structure_info["format_type"] = "table_based"

        # Detect section markers
        section_patterns = [
            (r"([A-Z][A-Z\s]{5,30}[A-Z])", "uppercase_title"),
            (r"(?:^|\n)((?:[A-Z][a-z]+\s){1,4}[A-Z][a-z]+)(?:\n|:)", "title_case"),
            (r"(?:^|\n)(\d+\.\s+[A-Z][a-zA-Z\s]{5,30})(?:\n|:)", "numbered_section"),
        ]

        for pattern, section_type in section_patterns:
            for match in re.finditer(pattern, self.text):
                section_name = match.group(1).strip()
                if len(section_name.split()) >= 2 or section_type == "uppercase_title":
                    structure_info["section_markers"].append(
                        {
                            "text": section_name,
                            "type": section_type,
                            "position": match.start(),
                        }
                    )

        # Sort section markers by position
        structure_info["section_markers"].sort(key=lambda x: x["position"])

        # Find table regions
        table_patterns = [
            r"(?:\|\s+){2,}",
            r"(?:[-]+\+[-]+){2,}",
            r"(?:[^\n]+\n){2,}(?:\s{2,}[^\s][^\n]+\n){2,}",
        ]

        for pattern in table_patterns:
            for match in re.finditer(pattern, self.text, re.MULTILINE):
                start = max(0, match.start() - 200)
                end = min(len(self.text), match.end() + 200)

                table_start = match.start()
                table_end = match.end()

                structure_info["table_regions"].append(
                    {
                        "start": table_start,
                        "end": table_end,
                        "context": self.text[table_start:table_end],
                    }
                )

        # Find common insurance field locations
        self._find_field_locations(structure_info)

        return structure_info

    def _find_field_locations(self, structure_info):
        """
        Find common insurance field locations in the document
        """
        common_field_patterns = {
            "Life Insurance,Benefit Amount": [
                r"basic\s+life[^.]*?(?:amount|benefit|coverage)[^.]*?(\d)",
                r"life\s+insurance[^.]*?(?:amount|benefit|coverage)[^.]*?(\d)",
            ],
            "STD,Benefit Amount": [
                r"(?:short\s*term|std)[^.]*?(?:benefit|percentage|amount)[^.]*?(\d{1,3}%)",
                r"(?:short\s*term|std)[^.]*?(?:pays|provides)[^.]*?(\d{1,3}%)",
            ],
            "LTD,Benefit Amount": [
                r"(?:long\s*term|ltd)[^.]*?(?:benefit|percentage|amount)[^.]*?(\d{1,3}%)",
                r"(?:long\s*term|ltd)[^.]*?(?:pays|provides)[^.]*?(\d{1,3}%)",
            ],
            "Life Insurance,Termination Age": [
                r"(?:life|coverage)[^.]*?(?:terminates|ceases)[^.]*?(?:age|at)\s*(\d{2})",
            ],
            "STD,Waiting Period": [
                r"(?:short\s*term|std)[^.]*?(?:waiting|elimination)[^.]*?(?:period|days)[^.]*?(\d{1,3})",
            ],
            "LTD,Waiting Period": [
                r"(?:long\s*term|ltd)[^.]*?(?:waiting|elimination)[^.]*?(?:period|days)[^.]*?(\d{1,3})",
            ],
        }

        for field_key, patterns in common_field_patterns.items():
            category, field = field_key.split(",")
            for pattern in patterns:
                for match in re.finditer(pattern, self.text, re.IGNORECASE):
                    if field_key not in structure_info["field_locations"]:
                        structure_info["field_locations"][field_key] = []

                    start = max(0, match.start() - 100)
                    end = min(len(self.text), match.end() + 100)

                    structure_info["field_locations"][field_key].append(
                        {"position": match.start(), "context": self.text[start:end]}
                    )
