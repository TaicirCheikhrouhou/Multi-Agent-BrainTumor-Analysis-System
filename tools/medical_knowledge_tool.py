from crewai.tools import tool

@tool("search_medical_knowledge")
def search_medical_knowledge(query: str) -> str:
    """
    Search medical knowledge base for information about brain tumors.
    This simulates a GraphRAG system for medical information retrieval.

    Args:
        query: Medical query about brain tumors
    """
    # Medical knowledge base
    knowledge_base = {
        "tumor_types": """
Primary brain tumor types:
1. Gliomas (50% of cases)
   - Glioblastoma (Grade IV): most aggressive, median survival 15 months
   - Astrocytoma (Grade II-III): survival 5-10 years
   - Oligodendroglioma: better prognosis

2. Meningiomas (30% of cases)
   - Generally benign
   - Slow growth
   - Good prognosis after surgery

3. Pituitary adenomas (10%)
   - Often functional
   - Medical or surgical treatment
        """,

        "diagnostic_procedures": """
Recommended additional examinations:
1. Brain MRI with gadolinium injection (gold standard)
2. Magnetic resonance spectroscopy (MRS)
3. FDG or methionine PET scan
4. Stereotactic biopsy if necessary
5. Complete neurological tests
6. Blood tests (tumor markers)
        """,

        "treatment_protocols": """
Treatment protocols:
1. Surgery:
   - Maximal resection if possible
   - Preservation of functional areas

2. Radiotherapy:
   - 3D conformal radiotherapy
   - Stereotactic radiosurgery (small tumors)
   - Dose: 54-60 Gy in 1.8-2 Gy fractions

3. Chemotherapy:
   - Temozolomide (standard for glioblastoma)
   - Adjuvant and neoadjuvant protocols

4. Targeted therapies and immunotherapy (clinical trials)
        """,

        "prognosis": """
Prognostic factors:
1. Tumor histological type
2. Tumor grade (WHO I-IV)
3. Location and size
4. Patient age (< 50 years: better prognosis)
5. Performance status (Karnofsky score)
6. Complete resectability
7. Genetic mutations (IDH, MGMT, 1p/19q)

5-year survival rates:
- Grade I Meningioma: >90%
- Grade II Astrocytoma: 50-70%
- Glioblastoma: 5-10%
        """
    }

    # Search in knowledge base
    results = []
    query_lower = query.lower()

    if "type" in query_lower or "classification" in query_lower:
        results.append(knowledge_base["tumor_types"])

    if "diagnostic" in query_lower or "examination" in query_lower:
        results.append(knowledge_base["diagnostic_procedures"])

    if "treatment" in query_lower or "therapy" in query_lower:
        results.append(knowledge_base["treatment_protocols"])

    if "prognosis" in query_lower or "survival" in query_lower:
        results.append(knowledge_base["prognosis"])

    if not results:
        results = list(knowledge_base.values())

    return "\n\n".join(results)