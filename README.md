# 🧬 DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline

> **Demonstration project for the UGA Institute for AI — Drug Research & Development Certificate**

An end-to-end AI pipeline that identifies and evaluates drug repurposing candidates by combining literature mining, molecular property analysis, and structured LLM reasoning.

## Overview

Drug repurposing — finding new therapeutic uses for existing drugs — is one of the most promising applications of AI in pharmaceutical research. This project demonstrates a complete pipeline that:

1. **Mines scientific literature** (arXiv) for repurposing evidence
2. **Extracts molecular properties** (PubChem) and evaluates drug-likeness
3. **Reasons about candidates** using structured LLM output
4. **Generates professional reports** summarizing findings

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Module 1    │───▶│  Module 2    │───▶│  Module 3    │───▶│  Module 4    │
│  Literature  │    │  Molecular   │    │  LLM         │    │  Report      │
│  Mining &    │    │  Property    │    │  Reasoning   │    │  Generation  │
│  RAG         │    │  Analysis    │    │  Engine      │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
   arXiv API          PubChem API        Local/API LLM       HTML/PDF
   + embedding         + RDKit            + DSPy/RAG          + Charts
     search             descriptors       + Outlines
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
git clone https://github.com/yourusername/drugdiscovery-ai.git
cd drugdiscovery-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Demo Notebook

```bash
jupyter notebook notebooks/DrugDiscovery_Demo.ipynb
```

### Run Modules Individually

```bash
# Module 1: Literature mining
python -m src.literature_mining "Alzheimer's disease"

# Module 2: Molecular property analysis
python -m src.molecular_analysis
```

## Project Structure

```
drugdiscovery-ai/
├── src/
│   ├── literature_mining.py    # Module 1: arXiv search, drug/disease extraction, knowledge base
│   ├── molecular_analysis.py   # Module 2: PubChem API, Lipinski rules, visualizations
│   ├── llm_reasoning.py        # Module 3: Structured LLM evaluation (coming soon)
│   └── report_generator.py     # Module 4: HTML report generation (coming soon)
├── notebooks/
│   └── DrugDiscovery_Demo.ipynb  # End-to-end demo notebook
├── data/                         # Generated data (CSV, JSON knowledge bases)
├── output/                       # Generated visualizations and reports
├── tests/                        # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Module Details

### Module 1: Literature Mining & RAG
- Searches arXiv for drug repurposing papers on a target disease
- Extracts drug names, disease mentions, and key findings using NLP patterns
- Builds a structured knowledge base (JSON) for downstream LLM reasoning
- **Skills demonstrated:** Academic research, RAG, knowledge extraction

### Module 2: Molecular Property Analysis
- Fetches compound properties from PubChem PUG-REST API
- Evaluates Lipinski's Rule of Five for drug-likeness
- Compares repurposing candidates against approved drugs
- Generates publication-quality visualizations (box plots, heatmaps)
- **Skills demonstrated:** Data science, API integration, cheminformatics, visualization

### Module 3: LLM Reasoning Engine (Preview)
- Takes literature findings + molecular data as context
- Uses structured output constraints (Outlines/Guidance) for reliable JSON output
- Generates ranked candidate evaluations with confidence scores
- **Skills demonstrated:** LLM inference, structured output, prompt engineering

### Module 4: Report Generation (Preview)
- Generates professional HTML reports with embedded visualizations
- Includes: literature summary, property comparison tables, ranked candidates
- **Skills demonstrated:** Technical communication, data presentation

## Example Output

### Drug Mentions from Literature (Alzheimer's Disease)
```
metformin: 8 papers
rapamycin: 5 papers
lithium: 4 papers
thalidomide: 3 papers
```

### Lipinski's Rule of Five Compliance
```
Name          MW (Da)  LogP   HBD  HBA  Violations  Drug-like
metformin     129.16   -1.3    3    1      0         ✅ Yes
rapamycin     914.20    6.0    3   13      3         ❌ No
naltrexone    341.40    1.9    2    5      0         ✅ Yes
donepezil     379.50    4.3    0    4      0         ✅ Yes
```

## Technologies Used

| Category | Tools |
|----------|-------|
| Data Sources | arXiv API, PubChem PUG-REST API |
| Data Science | pandas, numpy, matplotlib, seaborn |
| ML/AI | DSPy, Outlines, llama.cpp (optional) |
| Research | arxiv Python library, web scraping |
| Visualization | matplotlib, seaborn |
| Reporting | Jinja2 HTML templates |

## Relevance to UGA Institute for AI

This project directly demonstrates competencies taught in the Drug Research & Development certificate:

| Certificate Topic | Project Demonstration |
|---|---|
| Laboratory data analysis | Module 2: Molecular property computation from PubChem |
| Clinical trial data | Module 1: Literature mining of trial results |
| AI tool proficiency | Modules 1, 3, 4 — RAG, structured LLM output, automated reporting |
| Real-world applicability | Drug repurposing is a $3B+ industry application |
| Data pipeline thinking | Full pipeline from raw data → analysis → reasoning → report |

## Future Enhancements

- [ ] Integrate ChEMBL API for bioactivity data
- [ ] Add RDKit for molecular fingerprinting and similarity search
- [ ] Implement full RAG with vector embeddings (FAISS/ChromaDB)
- [ ] Add clinical trial data from ClinicalTrials.gov
- [ ] Build interactive Streamlit dashboard
- [ ] Add ADMET prediction models

## License

MIT License — see LICENSE file for details.

## Author

**Amanda Kavner, PhD**  
Interested in AI applications for drug discovery and development.
