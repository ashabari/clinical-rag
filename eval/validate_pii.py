"""
eval/validate_pii.py

Measures recall of src.pseudonymise.Pseudonymiser against a small,
hand-labelled synthetic test set covering common clinical-note PII
categories. Each test case lists the exact substrings that SHOULD be
redacted; we check whether they survive in the output.

This does NOT prove HIPAA Safe Harbor compliance -- it's a bounded,
honest spot-check: "of these known identifiers, what fraction did our
pipeline catch, and how were they labelled?"

Run: python -m eval.validate_pii
"""

from src.pseudonymise import Pseudonymiser

# Each test case: (input text, [(expected substring, category), ...])
# "expected substring" must disappear from the output for a hit.
TEST_CASES = [
    # --- Patient names ---
    ("Patient Maria Gonzalez was admitted for evaluation.",
     [("Maria Gonzalez", "patient_name")]),
    ("Mr. Robert Chen presented with chest pain radiating to the left arm.",
     [("Robert Chen", "patient_name")]),

    # --- Provider names (testing patient/provider mislabeling) ---
    ("Dr. Sarah Johnson performed the laparoscopic cholecystectomy without complication.",
     [("Sarah Johnson", "provider_name")]),
    ("The patient was seen by Dr. Michael Lee for a follow-up consultation.",
     [("Michael Lee", "provider_name")]),
    ("RN Jennifer Patel administered the medication at 0800.",
     [("Jennifer Patel", "provider_name")]),

    # --- Dates ---
    ("The patient was admitted on 03/14/2023 and discharged on 03/18/2023.",
     [("03/14/2023", "date"), ("03/18/2023", "date")]),
    ("Surgery was performed on January 5, 2022, without incident.",
     [("January 5, 2022", "date")]),

    # --- Ages (HIPAA flags ages over 89 specifically) ---
    ("This 92-year-old male presented with acute confusion.",
     [("92", "age_over_89")]),

    # --- Locations / addresses ---
    ("She resides at 4521 Maple Street, Springfield, Illinois 62704.",
     [("4521 Maple Street", "street_address"), ("Springfield", "city"), ("62704", "zip")]),
    ("He relocated to Denver shortly after his diagnosis.",
     [("Denver", "city")]),

    # --- Facility names ---
    ("The patient was transferred to Memorial Sloan Kettering Cancer Center for further care.",
     [("Memorial Sloan Kettering Cancer Center", "facility")]),
    ("A follow-up appointment was scheduled at Lakeside Family Clinic.",
     [("Lakeside Family Clinic", "facility")]),

    # --- Medical record / patient IDs ---
    ("MRN: 4487213, DOB 06/02/1965.",
     [("4487213", "mrn"), ("06/02/1965", "date")]),
    ("Patient ID #998213 was flagged for chart review.",
     [("998213", "patient_id")]),

    # --- SSN ---
    ("Social security number on file: 512-34-9087.",
     [("512-34-9087", "ssn")]),

    # --- Phone / fax ---
    ("The patient's contact number is 415-555-0192.",
     [("415-555-0192", "phone")]),
    ("Records were faxed to (312) 555-0177 per request.",
     [("(312) 555-0177", "fax")]),

    # --- Email ---
    ("Send lab results to patient.followup@gmail.com.",
     [("patient.followup@gmail.com", "email")]),

    # --- URL ---
    ("Results are available at portal.healthsystem.com/results/8841.",
     [("portal.healthsystem.com/results/8841", "url")]),
]


def run_validation():
    print(f"[pii-validation] Running {len(TEST_CASES)} test cases "
          f"({sum(len(e) for _, e in TEST_CASES)} labelled PII spans)...\n")

    total = 0
    caught = 0
    by_category = {}
    mislabel_examples = []

    for text, expected_spans in TEST_CASES:
        p = Pseudonymiser()  # fresh instance -- clean entity_map per case
        redacted = p.pseudonymise(text)

        for span, category in expected_spans:
            total += 1
            by_category.setdefault(category, {"total": 0, "caught": 0})
            by_category[category]["total"] += 1

            hit = span not in redacted
            if hit:
                caught += 1
                by_category[category]["caught"] += 1

            status = "CAUGHT" if hit else "MISSED"
            print(f"  [{status:6s}] ({category:15s}) {span!r}")

            # Check labelling for name categories
            if category in ("patient_name", "provider_name") and hit:
                for (ent_text, ent_label), token in p._entity_map.items():
                    if ent_text == span.lower():
                        expected_label = "PATIENT" if category == "patient_name" else "PATIENT (but is a PROVIDER)"
                        actual_label = token.strip("[]").rsplit("_", 1)[0]
                        if category == "provider_name" and actual_label == "PATIENT":
                            mislabel_examples.append((text, span, redacted))

        print(f"    IN:  {text}")
        print(f"    OUT: {redacted}\n")

    print("=" * 60)
    print("PII VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall recall: {caught}/{total} ({100*caught/total:.1f}%)\n")
    print("By category:")
    for cat, counts in sorted(by_category.items()):
        c, t = counts["caught"], counts["total"]
        print(f"  {cat:18s} {c}/{t} ({100*c/t:.0f}%)")

    if mislabel_examples:
        print(f"\n{'='*60}")
        print(f"PROVIDER/PATIENT MISLABELLING ({len(mislabel_examples)} examples)")
        print("=" * 60)
        print("Provider names are correctly redacted but labelled [PATIENT_n],")
        print("which could cause a reader to misattribute statements:\n")
        for text, span, redacted in mislabel_examples:
            print(f"  Provider name: {span!r}")
            print(f"    OUT: {redacted}\n")


if __name__ == "__main__":
    run_validation()
