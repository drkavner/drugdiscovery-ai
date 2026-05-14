"""
DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline
Module 1: Literature Mining & RAG

Searches arXiv for recent papers on drug repurposing for a target disease,
extracts key findings, and builds a lightweight knowledge base for downstream
LLM reasoning.
"""

import arxiv
import json
import re
import time
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────

MAX_RESULTS = 20
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Search ─────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = MAX_RESULTS, sort_by: str = "relevance") -> list[dict]:
    """
    Search arXiv for papers matching the query.
    Returns a list of dicts with title, authors, summary, published, pdf_url, arxiv_id.
    """
    import time
    
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "date": arxiv.SortCriterion.SubmittedDate,
    }
    
    client = arxiv.Client(
        page_size=min(max_results, 10),
        delay_seconds=5,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        sort_order=arxiv.SortOrder.Descending,
    )
    
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "summary": result.summary.strip().replace("\n", " "),
                "published": result.published.isoformat(),
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            })
    except Exception as e:
        print(f"   ⚠️  arXiv search issue: {e}")
        print("   This is usually a rate-limit issue. Try again in a few minutes.")
    
    return papers


# ── Extraction ─────────────────────────────────────────────────────────────

# Common drug name patterns and known drug lists for extraction
KNOWN_DRUGS = {
    # Approved drugs commonly discussed in repurposing
    "metformin", "aspirin", "ibuprofen", "acetaminophen", "paracetamol",
    "atorvastatin", "simvastatin", "rosuvastatin", "lisinopril", "amlodipine",
    "metoprolol", "losartan", "valsartan", "hydrochlorothiazide", "furosemide",
    "omeprazole", "pantoprazole", "levothyroxine", "prednisone", "dexamethasone",
    "methotrexate", "hydroxychloroquine", "azithromycin", "doxycycline",
    "baricitinib", "remdesivir", "molnupiravir", "nirmatrelvir", "ritonavir",
    "thalidomide", "lenalidomide", "pomalidomide",
    "memantine", "donepezil", "rivastigmine", "galantamine", "aducanumab",
    "lecanemab", "donanemab", "semaglutide", "liraglutide",
    "naltrexone", "bupropion", "fluoxetine", "sertraline", "escitalopram",
    "duloxetine", "venlafaxine", "amitriptyline", "nortriptyline",
    "carbamazepine", "valproate", "lamotrigine", "levetiracetam",
    "phenformin", "buformin", "rosiglitazone", "pioglitazone",
    "colchicine", "allopurinol", "febuxostat",
    "propranolol", "atenolol", "carvedilol",
    "warfarin", "rivaroxaban", "apixaban", "dabigatran",
    "clopidogrel", "prasugrel", "ticagrelor",
    "insulin", "glibenclamide", "gliclazide", "pioglitazone",
    "dapsone", "minocycline", "tetracycline", "clindamycin",
    "ketamine", "esketamine", "psilocybin", "MDMA",
    "niclosamide", "ivermectin", "fenbendazole", "mebendazole",
    "rapamycin", "sirolimus", "everolimus", "temsirolimus",
    "valproic acid", "sodium butyrate", "vorinostat", "romidepsin",
    "cisplatin", "carboplatin", "oxaliplatin", "paclitaxel", "docetaxel",
    "doxorubicin", "cyclophosphamide", "5-fluorouracil", "capecitabine",
    "imatinib", "erlotinib", "gefitinib", "osimertinib", "crizotinib",
    "nivolumab", "pembrolizumab", "atezolizumab", "ipilimumab",
    "tocilizumab", "sarilumab", "anakinra", "canakinumab",
    "dexamethasone", "methylprednisolone", "hydrocortisone",
    "lithium", "quetiapine", "olanzapine", "risperidone", "aripiprazole",
    "haloperidol", "chlorpromazine", "clozapine",
    "caffeine", "nicotine", "resveratrol", "curcumin", "quercetin",
    "epigallocatechin", "EGCG", "sulforaphane",
    "vitamin D", "calcifediol", "cholecalciferol",
    "zinc", "magnesium", "selenium", "coenzyme Q10", "CoQ10",
    "NAC", "N-acetylcysteine", "alpha-lipoic acid", "ALA",
    "berberine", "artemisinin", "chloroquine", "mefloquine",
    "disulfiram", "auranofin", "auranofin", "tideglusib",
    "tamoxifen", "raloxifene", "fulvestrant", "anastrozole",
    "spironolactone", "finasteride", "dutasteride",
    "minoxidil", "sildenafil", "tadalafil", "vardenafil",
    "cimetidine", "ranitidine", "famotidine",
    "probenecid", "penicillin", "amoxicillin", "cephalexin",
    "chloramphenicol", "erythromycin", "clarithromycin",
    "linezolid", "daptomycin", "vancomycin", "teicoplanin",
    "fluconazole", "itraconazole", "voriconazole", "posaconazole",
    "acyclovir", "valacyclovir", "ganciclovir", "foscarnet",
    "sofosbuvir", "ledipasvir", "velpatasvir", "glecaprevir",
    "bictegravir", "dolutegravir", "raltegravir", "elvitegravir",
    "tenofovir", "emtricitabine", "lamivudine", "abacavir",
    "efavirenz", "nevirapine", "rilpivirine", "doravirine",
    "darunavir", "atazanavir", "lopinavir", "fosamprenavir",
    "enfuvirtide", "maraviroc", "ibalizumab", "fostemsavir",
    "baloxavir", "oseltamivir", "zanamivir", "peramivir",
    "favipiravir", "galidesivir", "molnupiravir",
    "remdesivir", "nirmatrelvir", "ensitrelvir",
    "bebtelovimab", "casirivimab", "imdevimab", "sotrovimab",
    "bamlanivimab", "etesevimab", "tixagevimab", "cilgavimi",
}

# Common disease/condition keywords
DISEASE_KEYWORDS = {
    "alzheimer", "parkinson", "huntington", "als", "multiple sclerosis",
    "cancer", "carcinoma", "tumor", "tumour", "melanoma", "leukemia",
    "lymphoma", "sarcoma", "glioblastoma", "mesothelioma",
    "diabetes", "obesity", "metabolic syndrome",
    "covid", "sars-cov-2", "coronavirus", "influenza", "hiv", "aids",
    "hepatitis", "tuberculosis", "malaria", "dengue", "zika",
    "depression", "anxiety", "schizophrenia", "bipolar", "ptsd",
    "epilepsy", "migraine", "stroke", "ischemia",
    "arthritis", "lupus", "scleroderma", "fibromyalgia",
    "asthma", "copd", "pulmonary fibrosis", "emphysema",
    "hypertension", "atherosclerosis", "heart failure", "arrhythmia",
    "nephritis", "kidney disease", "renal failure",
    "cirrhosis", "nash", "fatty liver", "pancreatitis",
    "osteoporosis", "osteoarthritis", "gout",
    "macular degeneration", "glaucoma", "cataract",
    "psoriasis", "eczema", "dermatitis", "vitiligo",
    "crohn", "colitis", "ibs", "celiac",
    "endometriosis", "pcos", "infertility",
    "anemia", "sickle cell", "thalassemia", "hemophilia",
    "dementia", "cognitive impairment", "mci",
    "insomnia", "narcolepsy", "sleep apnea",
    "chronic pain", "neuropathy", "fibrosis",
    "sepsis", "ards", "multi-organ failure",
    "aging", "longevity", "senescence", "telomere",
}


def extract_drugs_from_text(text: str) -> list[str]:
    """Extract known drug names from text using dictionary matching."""
    text_lower = text.lower()
    found = set()
    for drug in KNOWN_DRUGS:
        # Use word boundary matching
        pattern = r'\b' + re.escape(drug.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(drug)
    return sorted(found)


def extract_diseases_from_text(text: str) -> list[str]:
    """Extract disease/condition mentions from text."""
    text_lower = text.lower()
    found = set()
    for disease in DISEASE_KEYWORDS:
        pattern = r'\b' + re.escape(disease.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(disease)
    return sorted(found)


def extract_key_findings(summary: str) -> list[str]:
    """Extract sentences that look like key findings from a paper summary."""
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    finding_indicators = [
        "found", "showed", "demonstrated", "revealed", "suggest",
        "indicate", "propose", "identified", "discovered", "report",
        "observed", "confirmed", "established", "evidence", "result",
        "significant", "effective", "potential", "promising", "novel",
        "repurposing", "repurpose", "repositioning", "reposition",
        "inhibit", "reduce", "improve", "decrease", "increase",
        "pathway", "mechanism", "target", "biomarker",
    ]
    
    findings = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(1 for ind in finding_indicators if ind in sent_lower)
        if score >= 2 and len(sent) > 30:
            findings.append(sent.strip())
    
    return findings[:5]  # Top 5 most relevant sentences


# ── Knowledge Base ─────────────────────────────────────────────────────────

def build_knowledge_base(papers: list[dict], disease: str) -> dict:
    """
    Build a structured knowledge base from extracted papers.
    This serves as the RAG context for Module 3.
    """
    kb = {
        "disease": disease,
        "query_date": datetime.now().isoformat(),
        "total_papers": len(papers),
        "papers": [],
        "drug_mentions": {},
        "disease_mentions": {},
        "all_findings": [],
    }
    
    for paper in papers:
        drugs = extract_drugs_from_text(paper["title"] + " " + paper["summary"])
        diseases = extract_diseases_from_text(paper["title"] + " " + paper["summary"])
        findings = extract_key_findings(paper["summary"])
        
        paper_entry = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"][:3],  # First 3 authors
            "published": paper["published"][:10],
            "drugs_mentioned": drugs,
            "diseases_mentioned": diseases,
            "key_findings": findings,
            "summary_short": paper["summary"][:300] + "...",
        }
        kb["papers"].append(paper_entry)
        kb["all_findings"].extend(findings)
        
        # Aggregate drug mentions
        for drug in drugs:
            if drug not in kb["drug_mentions"]:
                kb["drug_mentions"][drug] = {"count": 0, "papers": []}
            kb["drug_mentions"][drug]["count"] += 1
            kb["drug_mentions"][drug]["papers"].append(paper["arxiv_id"])
        
        # Aggregate disease mentions
        for disease in diseases:
            if disease not in kb["disease_mentions"]:
                kb["disease_mentions"][disease] = 0
            kb["disease_mentions"][disease] += 1
    
    # Sort drugs by mention frequency
    kb["drug_mentions"] = dict(
        sorted(kb["drug_mentions"].items(), key=lambda x: x[1]["count"], reverse=True)
    )
    
    return kb


def save_knowledge_base(kb: dict, filename: str = None) -> Path:
    """Save knowledge base to JSON file."""
    if filename is None:
        safe_disease = kb["disease"].replace(" ", "_").replace("'", "")
        filename = f"kb_{safe_disease}_{datetime.now().strftime('%Y%m%d')}.json"
    
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(kb, f, indent=2)
    return path


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_literature_mining(disease: str, max_results: int = MAX_RESULTS) -> dict:
    """
    Full literature mining pipeline for a target disease.
    
    Args:
        disease: Target disease/condition (e.g., "Alzheimer's disease")
        max_results: Maximum number of papers to retrieve
    
    Returns:
        Structured knowledge base dict
    """
    print(f"\n{'='*60}")
    print(f"  DrugDiscovery.ai — Literature Mining")
    print(f"  Target: {disease}")
    print(f"{'='*60}\n")
    
    # Build search query
    query = f"drug repurposing AND ({disease})"
    print(f"🔍 Searching arXiv for: '{query}'...")
    
    papers = search_arxiv(query, max_results=max_results)
    print(f"   Found {len(papers)} papers\n")
    
    if not papers:
        print("⚠️  No papers found. Trying broader query...")
        query = f"drug repositioning {disease}"
        papers = search_arxiv(query, max_results=max_results)
        print(f"   Found {len(papers)} papers with broader query\n")
    
    # Build knowledge base
    print("🧠 Building knowledge base...")
    kb = build_knowledge_base(papers, disease)
    
    # Save
    path = save_knowledge_base(kb)
    print(f"   Saved to: {path}\n")
    
    # Print summary
    print(f"📊 Summary:")
    print(f"   Papers analyzed: {kb['total_papers']}")
    print(f"   Unique drugs mentioned: {len(kb['drug_mentions'])}")
    print(f"   Key findings extracted: {len(kb['all_findings'])}")
    
    if kb["drug_mentions"]:
        print(f"\n💊 Top drug mentions:")
        for drug, info in list(kb["drug_mentions"].items())[:10]:
            print(f"   {drug}: {info['count']} paper(s)")
    
    return kb


if __name__ == "__main__":
    import sys
    disease = sys.argv[1] if len(sys.argv) > 1 else "Alzheimer's disease"
    kb = run_literature_mining(disease)
