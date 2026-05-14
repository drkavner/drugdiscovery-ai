"""
DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline
Module 3: LLM Reasoning Engine

Takes the knowledge base (Module 1) + molecular properties (Module 2) as context,
uses structured output to generate ranked candidate evaluations.

Supports four backends, auto-detected from environment (see .env.example):
  claude       — Anthropic direct (anthropic SDK, output_config.format)
  openrouter   — OpenRouter (openai SDK, json_schema response_format)
  ollama       — Local Ollama (openai SDK pointed at localhost)
  heuristic    — Built-in deterministic scorer (no LLM, no cost)
"""

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Structured Output Schema ──────────────────────────────────────────────

DRUG_EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "drug_name": {"type": "string"},
        "original_indication": {"type": "string"},
        "proposed_mechanism": {"type": "string"},
        "evidence_from_literature": {"type": "string"},
        "drug_likeness_assessment": {"type": "string"},
        "confidence_score": {"type": "number"},
        "key_risks": {
            "type": "array",
            "items": {"type": "string"}
        },
        "recommended_next_steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "overall_recommendation": {
            "type": "string",
            "enum": ["Strong candidate", "Promising — needs validation", "Moderate potential", "Weak candidate — not recommended"]
        }
    },
    "required": [
        "drug_name", "original_indication", "proposed_mechanism",
        "evidence_from_literature", "drug_likeness_assessment",
        "confidence_score", "key_risks", "recommended_next_steps",
        "overall_recommendation",
    ],
}

FINAL_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "disease": {"type": "string"},
        "analysis_date": {"type": "string"},
        "total_candidates_evaluated": {"type": "integer"},
        "top_candidates": {
            "type": "array",
            "items": DRUG_EVALUATION_SCHEMA
        },
        "summary": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}


# ── Prompt Templates ──────────────────────────────────────────────────────

CANDIDATE_EVALUATION_PROMPT = """You are a pharmaceutical AI research assistant specializing in drug repurposing analysis.

## Target Disease
{disease}

## Literature Evidence
{literature_summary}

## Molecular Property Data
{molecular_data}

## Your Task
Evaluate the following drug as a repurposing candidate for {drug_name}:

**Drug:** {drug_name}
**Molecular Weight:** {mw} Da
**LogP:** {logp}
**Lipinski Drug-like:** {drug_like}
**Literature Mention Count:** {mention_count} papers

Generate a structured evaluation with the following information:
1. Original approved/known indication
2. Proposed mechanism of action for {disease}
3. Key evidence from the literature
4. Drug-likeness assessment
5. Confidence score (0-10)
6. Key risks or concerns
7. Recommended next steps
8. Overall recommendation category

Respond ONLY with valid JSON. Do not include any text outside the JSON object.
"""

FINAL_SYNTHESIS_PROMPT = """You are a pharmaceutical AI research assistant. You have evaluated multiple drug candidates for repurposing to treat {disease}.

## Candidate Evaluations
{candidate_evaluations}

## Key Literature Findings
{key_findings}

## Your Task
Synthesize the evaluations into a final ranked report. Identify the top 3-5 candidates with the strongest evidence and drug-like profiles. Provide:
1. A brief executive summary
2. Ranked list of top candidates with justification
3. Key findings from the analysis
4. Recommended next research steps

Respond ONLY with valid JSON matching the required schema.
"""


# ─ Evaluation Logic (works with any LLM backend) ──────────────────────────

def build_candidate_context(
    drug_name: str,
    kb: dict,
    mol_data: dict,
    disease: str,
) -> dict:
    """Build the context dictionary for a single candidate evaluation."""
    # Get literature mentions for this drug
    drug_info = kb.get("drug_mentions", {}).get(drug_name, {"count": 0, "papers": []})
    
    # Find relevant papers
    relevant_findings = []
    for paper in kb.get("papers", []):
        if drug_name.lower() in paper.get("title", "").lower() or \
           drug_name.lower() in paper.get("summary_short", "").lower():
            relevant_findings.extend(paper.get("key_findings", []))
    
    return {
        "disease": disease,
        "drug_name": drug_name,
        "mw": mol_data.get("molecular_weight", "N/A"),
        "logp": mol_data.get("log_p", "N/A"),
        "drug_like": "Yes" if mol_data.get("drug_like") else "No",
        "mention_count": drug_info["count"],
        "literature_summary": "\n".join(relevant_findings[:3]) if relevant_findings else "No specific findings in retrieved papers.",
        "molecular_data": json.dumps(mol_data, default=str),
    }


def generate_evaluation_heuristic(
    drug_name: str,
    kb: dict,
    mol_data: dict,
    disease: str,
) -> dict:
    """
    Generate a heuristic-based evaluation when no LLM is available.
    This demonstrates the structured output format and scoring logic.
    """
    drug_info = kb.get("drug_mentions", {}).get(drug_name.lower(), {"count": 0, "papers": []})
    mention_count = drug_info["count"]
    
    # Score based on available evidence
    score = 0
    factors = []
    
    # Literature evidence (0-3 points)
    if mention_count >= 5:
        score += 3
        factors.append(f"Strong literature evidence ({mention_count} papers)")
    elif mention_count >= 2:
        score += 2
        factors.append(f"Moderate literature evidence ({mention_count} papers)")
    elif mention_count >= 1:
        score += 1
        factors.append(f"Limited literature evidence ({mention_count} paper)")
    else:
        factors.append("No direct literature mentions in retrieved papers")
    
    # Drug-likeness (0-3 points)
    if mol_data.get("drug_like"):
        score += 3
        factors.append("Passes Lipinski's Rule of Five")
    else:
        violations = mol_data.get("lipinski_violations", 0)
        if violations == 1:
            score += 2
            factors.append("Minor Lipinski violation (may still be viable)")
        elif violations <= 2:
            score += 1
            factors.append(f"Moderate Lipinski violations ({violations})")
        else:
            factors.append(f"Poor drug-like properties ({violations} Lipinski violations)")
    
    # MW in good oral drug range (0-2 points)
    mw = mol_data.get("molecular_weight", 0)
    if 150 <= mw <= 500:
        score += 2
        factors.append(f"Optimal molecular weight ({mw} Da)")
    elif 100 <= mw <= 600:
        score += 1
        factors.append(f"Acceptable molecular weight ({mw} Da)")
    else:
        factors.append(f"Extreme molecular weight ({mw} Da)")
    
    # LogP in good range (0-2 points)
    logp = mol_data.get("log_p")
    if logp is not None and 0 <= logp <= 4:
        score += 2
        factors.append(f"Favorable lipophilicity (LogP={logp})")
    elif logp is not None and -1 <= logp <= 5:
        score += 1
        factors.append(f"Acceptable lipophilicity (LogP={logp})")
    elif logp is not None:
        factors.append(f"Suboptimal lipophilicity (LogP={logp})")
    
    # Determine recommendation
    if score >= 8:
        recommendation = "Strong candidate"
    elif score >= 6:
        recommendation = "Promising — needs validation"
    elif score >= 4:
        recommendation = "Moderate potential"
    else:
        recommendation = "Weak candidate — not recommended"
    
    return {
        "drug_name": drug_name,
        "original_indication": "See literature for original use",
        "proposed_mechanism": f"Potential mechanism for {disease} based on literature analysis",
        "evidence_from_literature": "; ".join(factors),
        "drug_likeness_assessment": "Drug-like" if mol_data.get("drug_like") else f"{mol_data.get('lipinski_violations', '?')} Lipinski violation(s)",
        "confidence_score": score,
        "key_risks": [
            "Requires experimental validation",
            "Off-target effects unknown",
            "Dosing for new indication TBD",
        ],
        "recommended_next_steps": [
            "In vitro validation in disease-relevant cell models",
            "In vivo efficacy studies in animal models",
            "PK/PD modeling for new indication",
            "Literature deep-dive on mechanism of action",
        ],
        "overall_recommendation": recommendation,
    }


# ── Backend Dispatch ───────────────────────────────────────────────────────

def _detect_backend() -> str:
    """Pick a backend based on environment configuration (see .env.example)."""
    explicit = os.getenv("LLM_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OLLAMA_HOST"):
        return "ollama"
    return "heuristic"


def _ollama_base_url() -> str:
    host = os.environ["OLLAMA_HOST"].rstrip("/")
    if not host.endswith("/v1"):
        host += "/v1"
    return host


def _call_claude(prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-opus-4-7"),
        max_tokens=1024,
        output_config={
            "format": {"type": "json_schema", "schema": DRUG_EVALUATION_SCHEMA}
        },
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _call_openai_compatible(prompt: str, base_url: str, api_key: str, model: str) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "drug_evaluation",
                "schema": DRUG_EVALUATION_SCHEMA,
                "strict": True,
            },
        },
    )
    return json.loads(response.choices[0].message.content)


def generate_evaluation(
    drug_name: str,
    kb: dict,
    mol_data: dict,
    disease: str,
) -> dict:
    """Dispatch to the configured backend; fall back to the heuristic on failure."""
    backend = _detect_backend()

    if backend == "heuristic":
        return generate_evaluation_heuristic(drug_name, kb, mol_data, disease)

    ctx = build_candidate_context(drug_name, kb, mol_data, disease)
    prompt = CANDIDATE_EVALUATION_PROMPT.format(**ctx)

    try:
        if backend == "claude":
            return _call_claude(prompt)
        if backend == "openrouter":
            return _call_openai_compatible(
                prompt,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            )
        if backend == "ollama":
            return _call_openai_compatible(
                prompt,
                base_url=_ollama_base_url(),
                api_key="ollama",
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            )
        print(f"   ⚠️  Unknown LLM_BACKEND={backend!r}, using heuristic")
    except Exception as e:
        print(f"   ⚠️  {backend} backend failed ({type(e).__name__}: {e}); using heuristic")

    return generate_evaluation_heuristic(drug_name, kb, mol_data, disease)


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_llm_reasoning(
    kb: dict,
    mol_df,  # pandas DataFrame
    disease: str,
    top_n: int = 5,
) -> list[dict]:
    """
    Run the full evaluation pipeline for all candidates.
    
    Args:
        kb: Knowledge base from Module 1
        mol_df: Molecular properties DataFrame from Module 2
        disease: Target disease
    
    Returns:
        List of evaluation dicts, sorted by confidence score
    """
    import pandas as pd
    
    backend = _detect_backend()
    print(f"\n{'='*60}")
    print(f"  DrugDiscovery.ai — LLM Reasoning Engine")
    print(f"  Disease: {disease}")
    print(f"  Backend: {backend}")
    print(f"{'='*60}\n")

    # Filter to repurposing candidates only
    candidates = mol_df[mol_df["status"] == "repurposing_candidate"] if "status" in mol_df.columns else mol_df

    evaluations = []
    for _, row in candidates.iterrows():
        drug_name = row["name"]
        print(f"  Evaluating: {drug_name}...")

        mol_data = row.to_dict()
        evaluation = generate_evaluation(drug_name, kb, mol_data, disease)
        evaluations.append(evaluation)
        
        print(f"    Score: {evaluation['confidence_score']}/10 — {evaluation['overall_recommendation']}")
    
    # Sort by confidence score
    evaluations.sort(key=lambda x: x["confidence_score"], reverse=True)
    
    # Save
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "evaluations.json"
    with open(path, "w") as f:
        json.dump(evaluations, f, indent=2)
    
    print(f"\n💾 Saved evaluations to: {path}")
    print(f"\n🏆 Top {min(top_n, len(evaluations))} Candidates:")
    for i, ev in enumerate(evaluations[:top_n]):
        print(f"  {i+1}. {ev['drug_name']:20s} — Score: {ev['confidence_score']}/10 — {ev['overall_recommendation']}")
    
    return evaluations


if __name__ == "__main__":
    print("This module is designed to be imported and used with the full pipeline.")
    print("See notebooks/DrugDiscovery_Demo.ipynb for the end-to-end demo.")
