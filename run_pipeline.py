#!/usr/bin/env python3
"""
DrugDiscovery.ai — Full Pipeline Runner

Runs all 4 modules end-to-end for a target disease.

Usage:
    python run_pipeline.py [disease]

Example:
    python run_pipeline.py "Alzheimer's disease"
    python run_pipeline.py "Parkinson's disease"
    python run_pipeline.py "COVID-19"
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from literature_mining import run_literature_mining
from molecular_analysis import run_molecular_analysis
from llm_reasoning import run_llm_reasoning
from report_generator import generate_report


def main():
    disease = sys.argv[1] if len(sys.argv) > 1 else "Alzheimer's disease"
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           🧬 DrugDiscovery.ai Pipeline v1.0            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Target disease: {disease}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    t_start = time.time()
    
    # ── Module 1: Literature Mining ────────────────────────────────────
    print("\n" + "━"*60)
    print("  PHASE 1/4: Literature Mining & RAG")
    print("━"*60)
    
    try:
        kb = run_literature_mining(disease, max_results=10)
    except Exception as e:
        print(f"\n⚠️  Literature mining encountered an issue: {e}")
        print("   Creating minimal knowledge base for demo purposes...")
        kb = {
            "disease": disease,
            "query_date": datetime.now().isoformat(),
            "total_papers": 0,
            "papers": [],
            "drug_mentions": {},
            "disease_mentions": {},
            "all_findings": [],
        }
    
    # ── Module 2: Molecular Property Analysis ──────────────────────────
    print("\n" + "━"*60)
    print("  PHASE 2/4: Molecular Property Analysis")
    print("━"*60)
    
    # Get candidates from literature, or use defaults
    literature_drugs = list(kb.get("drug_mentions", {}).keys())[:8]
    
    # Default candidates for common diseases
    defaults = {
        "alzheimer": ["metformin", "rapamycin", "lithium", "naltrexone"],
        "parkinson": ["metformin", "rapamycin", "lithium", "exenatide"],
        "covid": ["baricitinib", "remdesivir", "ivermectin", "fluvoxamine"],
    }
    
    disease_lower = disease.lower()
    if not literature_drugs:
        for key, drugs in defaults.items():
            if key in disease_lower:
                literature_drugs = drugs
                break
        if not literature_drugs:
            literature_drugs = ["metformin", "rapamycin", "lithium"]
    
    # Approved drugs for comparison
    approved_map = {
        "alzheimer": ["donepezil", "rivastigmine", "memantine"],
        "parkinson": ["levodopa", "carbidopa", "ropinirole"],
        "covid": ["remdesivir", "nirmatrelvir", "molnupiravir"],
    }
    approved_drugs = []
    for key, drugs in approved_map.items():
        if key in disease_lower:
            approved_drugs = drugs
            break
    if not approved_drugs:
        approved_drugs = ["metformin"]  # fallback
    
    df = run_molecular_analysis(
        candidate_drugs=literature_drugs,
        disease=disease,
        approved_drugs=approved_drugs,
    )
    
    # ── Module 3: LLM Reasoning ────────────────────────────────────────
    print("\n" + "━"*60)
    print("  PHASE 3/4: AI Candidate Evaluation")
    print("━"*60)
    
    evaluations = run_llm_reasoning(kb, df, disease)
    
    # ── Module 4: Report Generation ────────────────────────────────────
    print("\n" + "━"*60)
    print("  PHASE 4/4: Report Generation")
    print("━"*60)
    
    report_path = generate_report(disease, kb, df, evaluations)
    
    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    
    print("\n" + "╔══════════════════════════════════════════════════════════╗")
    print("║                    ✅ Pipeline Complete                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  ⏱  Total time: {elapsed:.1f}s")
    print(f"  📊 Papers analyzed: {kb.get('total_papers', 0)}")
    print(f"  💊 Candidates evaluated: {len(evaluations)}")
    print(f"  📄 Report: {report_path}")
    print(f"\n  Output files:")
    
    output_dir = Path("output")
    data_dir = Path("data")
    
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    output/{f.name} ({size:.0f} KB)")
    for f in sorted(data_dir.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    data/{f.name} ({size:.0f} KB)")
    
    print(f"\n  Open the report in your browser:")
    print(f"    file://{report_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
