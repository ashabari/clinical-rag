"""
src/pseudonymise.py
Two-pass PII removal using spaCy NER.
Pass 1: applied to chunks before indexing (nothing sensitive ever stored)
Pass 2: applied to user queries before hitting the LLM
"""
import re
import spacy
from pathlib import Path

# Load once at module level — expensive to reload
print("[pseudonymiser] Loading spaCy model...")
nlp = spacy.load("en_core_web_trf")
print("[pseudonymiser] Model ready.")

# Entity types to redact and their placeholder prefix
REDACT_LABELS = {
    "PERSON":   "PATIENT",
    "GPE":      "LOCATION",
    "LOC":      "LOCATION",
    "ORG":      "FACILITY",
    "DATE":     "DATE",
    "TIME":     "TIME",
    "AGE":      "AGE",
}

# Regex patterns for things spaCy misses
REGEX_PATTERNS = [
    # MRN / patient ID patterns
    (re.compile(r'\bMRN[\s:#]*\d+\b', re.I),       '[MRN]'),
    (re.compile(r'\bPatient\s+ID[\s:#]*\d+\b', re.I), '[PATIENT_ID]'),
    # Phone numbers
    (re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]\d{4}\b'), '[PHONE]'),  # handles (312) 555-0177 too
    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),           '[SSN]'),
    # Email
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I), '[EMAIL]'),
    # URLs / patient-portal links -- after EMAIL deliberately (see above)
    (re.compile(r'\b(?:https?://)?(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|health|info)\b(?:/\S*)?', re.I), '[URL]'),
    # Dates (catches formats spaCy sometimes misses)
    (re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),    '[DATE]'),
]


# Titles that mark a nearby PERSON entity as a clinician, not a patient.
# Checked immediately before ("Dr. Sarah Johnson") or after
# ("Jane Doe, MD" / "Jennifer Patel RN") the entity span.
PROVIDER_TITLE_PREFIXES = {"dr", "prof", "professor", "rn", "np", "pa"}
PROVIDER_TITLE_SUFFIXES = {"md", "do", "rn", "np", "pa", "pa-c", "phd"}


def _is_provider_context(doc, ent) -> bool:
    """Heuristic: does this PERSON entity look like a clinician, not a patient?"""
    if ent.start > 0:
        prev = doc[ent.start - 1].text.strip(".,").lower()
        if prev in PROVIDER_TITLE_PREFIXES:
            return True
    if ent.end < len(doc):
        nxt = doc[ent.end].text.strip(".,").lower()
        if nxt in PROVIDER_TITLE_SUFFIXES:
            return True
        if doc[ent.end].text == "," and ent.end + 1 < len(doc):
            nxt2 = doc[ent.end + 1].text.strip(".,").lower()
            if nxt2 in PROVIDER_TITLE_SUFFIXES:
                return True
    return False


class Pseudonymiser:
    """
    Deterministic PII replacer.
    Same entity always maps to the same token within a session
    so the text remains internally consistent.
    e.g. "John Smith" → [PATIENT_1] consistently throughout all chunks.
    """

    def __init__(self):
        self._entity_map: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def _get_token(self, entity_text: str, label: str) -> str:
        """Return a stable token for this entity, creating one if needed."""
        key = (entity_text.lower().strip(), label)
        if key not in self._entity_map:
            self._counters[label] = self._counters.get(label, 0) + 1
            self._entity_map[key] = f"[{label}_{self._counters[label]}]"
        return self._entity_map[key]

    def pseudonymise(self, text: str) -> str:
        """Replace PII in text with stable placeholder tokens."""
        if not text or not text.strip():
            return text

        # Pass 1: spaCy NER
        doc = nlp(text)
        # Process entities in reverse order to preserve character offsets
        replacements = []
        for ent in doc.ents:
            if ent.label_ in REDACT_LABELS:
                placeholder_label = REDACT_LABELS[ent.label_]
                if ent.label_ == "PERSON" and _is_provider_context(doc, ent):
                    placeholder_label = "PROVIDER"
                token = self._get_token(ent.text, placeholder_label)
                replacements.append((ent.start_char, ent.end_char, token))

        # Apply replacements in reverse order
        result = text
        for start, end, token in sorted(replacements, reverse=True):
            result = result[:start] + token + result[end:]

        # Pass 2: regex patterns for structured PII spaCy misses
        for pattern, placeholder in REGEX_PATTERNS:
            result = pattern.sub(placeholder, result)

        return result

    def pseudonymise_query(self, query: str) -> str:
        """
        Lightweight version for user queries — uses a fresh context
        so query tokens don't collide with document tokens.
        """
        temp = Pseudonymiser()
        return temp.pseudonymise(query)

    @property
    def entity_map(self) -> dict:
        """Read-only view of the mapping (for verification only, never persisted)."""
        return {f"{k[0]} ({k[1]})": v for k, v in self._entity_map.items()}


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    p = Pseudonymiser()

    test_cases = [
        "Patient John Smith, DOB 03/15/1978, MRN: 123456, presented to St. Mary's Hospital in Boston.",
        "Dr. Sarah Johnson referred the patient from Seattle Medical Center on 01/20/2024.",
        "Contact: jane.doe@email.com or 555-867-5309",
        "John Smith was seen again today.",  # should reuse [PATIENT_1]
    ]

    print("\n--- Pseudonymisation smoke test ---")
    for text in test_cases:
        result = p.pseudonymise(text)
        print(f"\nIN:  {text}")
        print(f"OUT: {result}")

    print("\n--- Entity map (in-memory only, never saved) ---")
    for k, v in p.entity_map.items():
        print(f"  {v:20s} ← {k}")
