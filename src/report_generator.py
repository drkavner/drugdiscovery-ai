"""
DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline
Module 4: HTML Report Generator

Generates a professional HTML report combining literature findings,
molecular property analysis, and LLM evaluations.
"""

import json
from pathlib import Path
from datetime import datetime

# ── HTML Template ──────────────────────────────────────────────────────────

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrugDiscovery.ai — {disease} Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        background: #f5f7fa;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 20px;
    }}
    header {{
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 40px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(26, 35, 126, 0.3);
    }}
    header h1 {{
        font-size: 2.2em;
        margin-bottom: 8px;
    }}
    header .subtitle {{
        font-size: 1.1em;
        opacity: 0.9;
    }}
    header .meta {{
        margin-top: 15px;
        font-size: 0.9em;
        opacity: 0.8;
    }}
    .card {{
        background: white;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    .card h2 {{
        color: #1a237e;
        font-size: 1.4em;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e8eaf6;
    }}
    .card h3 {{
        color: #283593;
        font-size: 1.1em;
        margin: 15px 0 10px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.9em;
    }}
    th, td {{
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }}
    th {{
        background: #e8eaf6;
        color: #1a237e;
        font-weight: 600;
    }}
    tr:hover {{
        background: #f5f5f5;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }}
    .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
    .badge-warning {{ background: #fff3e0; color: #e65100; }}
    .badge-danger {{ background: #ffebee; color: #c62828; }}
    .badge-info {{ background: #e3f2fd; color: #1565c0; }}
    .candidate-card {{
        border-left: 4px solid #3949ab;
        padding: 15px 20px;
        margin: 15px 0;
        background: #fafafa;
        border-radius: 0 8px 8px 0;
    }}
    .candidate-card h4 {{
        color: #1a237e;
        font-size: 1.15em;
        margin-bottom: 8px;
    }}
    .score-bar {{
        display: inline-block;
        width: 100px;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        vertical-align: middle;
        margin-left: 8px;
    }}
    .score-fill {{
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
    }}
    .finding {{
        padding: 8px 12px;
        margin: 6px 0;
        background: #f3e5f5;
        border-radius: 6px;
        font-size: 0.9em;
        border-left: 3px solid #9c27b0;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
    .stat-box {{
        text-align: center;
        padding: 20px;
        background: #e8eaf6;
        border-radius: 8px;
    }}
    .stat-box .number {{
        font-size: 2em;
        font-weight: 700;
        color: #1a237e;
    }}
    .stat-box .label {{
        font-size: 0.85em;
        color: #666;
        margin-top: 5px;
    }}
    img {{
        max-width: 100%;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    footer {{
        text-align: center;
        padding: 20px;
        color: #999;
        font-size: 0.85em;
    }}
    @media (max-width: 768px) {{
        .grid-2 {{ grid-template-columns: 1fr; }}
        header h1 {{ font-size: 1.6em; }}
    }}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>🧬 DrugDiscovery.ai</h1>
    <div class="subtitle">AI-Powered Drug Repurposing Analysis Report</div>
    <div style="margin-top: 10px; font-size: 1.3em; font-weight: 600;">{disease}</div>
    <div class="meta">
        Generated: {date} | Pipeline v1.0
    </div>
</header>

<!-- Executive Summary -->
<div class="card">
    <h2>📋 Executive Summary</h2>
    <div class="grid-2">
        <div class="stat-box">
            <div class="number">{total_papers}</div>
            <div class="label">Papers Analyzed</div>
        </div>
        <div class="stat-box">
            <div class="number">{total_candidates}</div>
            <div class="label">Candidates Evaluated</div>
        </div>
        <div class="stat-box">
            <div class="number">{drug_like_count}</div>
            <div class="label">Drug-like Candidates</div>
        </div>
        <div class="stat-box">
            <div class="number">{top_score}</div>
            <div class="label">Top Confidence Score</div>
        </div>
    </div>
    <p style="margin-top: 15px;">{summary}</p>
</div>

<!-- Literature Findings -->
<div class="card">
    <h2>📚 Literature Analysis</h2>
    <h3>Top Drug Mentions in Recent Papers</h3>
    <table>
        <thead>
            <tr><th>Drug</th><th>Paper Count</th><th>Relevance</th></tr>
        </thead>
        <tbody>
            {drug_mention_rows}
        </tbody>
    </table>
    
    <h3>Key Findings from Literature</h3>
    {key_findings_html}
</div>

<!-- Molecular Properties -->
<div class="card">
    <h2>⚗️ Molecular Property Analysis</h2>
    <table>
        <thead>
            <tr>
                <th>Drug</th><th>MW (Da)</th><th>LogP</th>
                <th>HBD</th><th>HBA</th><th>Lipinski</th><th>Status</th>
            </tr>
        </thead>
        <tbody>
            {property_rows}
        </tbody>
    </table>
    
    {visualization_html}
</div>

<!-- LLM Evaluations -->
<div class="card">
    <h2>🤖 AI Candidate Evaluations</h2>
    {candidate_evaluations_html}
</div>

<!-- Methodology -->
<div class="card">
    <h2>🔬 Methodology</h2>
    <p>This analysis was generated using the DrugDiscovery.ai pipeline:</p>
    <ol style="margin: 10px 0 10px 20px;">
        <li><strong>Literature Mining:</strong> arXiv API search for drug repurposing papers, NLP-based extraction of drug names, diseases, and key findings</li>
        <li><strong>Molecular Analysis:</strong> PubChem PUG-REST API for compound properties, Lipinski's Rule of Five evaluation</li>
        <li><strong>AI Reasoning:</strong> Structured evaluation combining literature evidence with molecular properties</li>
        <li><strong>Report Generation:</strong> Automated HTML report with interactive visualizations</li>
    </ol>
    <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
        <em>This is an AI-assisted analysis for research purposes. All candidates require experimental validation.
        This tool does not provide medical advice.</em>
    </p>
</div>

<footer>
    <p>DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline</p>
    <p>Generated on {date}</p>
</footer>

</div>
</body>
</html>
"""


# ── Report Builder ─────────────────────────────────────────────────────────

def generate_report(
    disease: str,
    kb: dict,
    mol_df,  # pandas DataFrame
    evaluations: list[dict],
    output_dir: str = None,
) -> Path:
    """
    Generate a complete HTML report.
    
    Args:
        disease: Target disease name
        kb: Knowledge base dict from Module 1
        mol_df: Molecular properties DataFrame from Module 2
        evaluations: List of evaluation dicts from Module 3
        output_dir: Output directory (default: project output/)
    
    Returns:
        Path to generated HTML file
    """
    import pandas as pd
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # ── Build template variables ────────────────────────────────────────
    
    # Drug mention rows
    drug_mention_rows = ""
    for drug, info in list(kb.get("drug_mentions", {}).items())[:10]:
        count = info["count"]
        if count >= 5:
            relevance = '<span class="badge badge-success">High</span>'
        elif count >= 2:
            relevance = '<span class="badge badge-warning">Medium</span>'
        else:
            relevance = '<span class="badge badge-info">Low</span>'
        drug_mention_rows += f"<tr><td><strong>{drug.capitalize()}</strong></td><td>{count}</td><td>{relevance}</td></tr>\n"
    
    # Key findings
    findings_html = ""
    for finding in kb.get("all_findings", [])[:8]:
        findings_html += f'<div class="finding">{finding}</div>\n'
    if not findings_html:
        findings_html = "<p><em>No specific findings extracted.</em></p>"
    
    # Property table rows
    property_rows = ""
    for _, row in mol_df.iterrows():
        name = row.get("name", "?")
        mw = row.get("molecular_weight", "?")
        logp = row.get("log_p", "?")
        hbd = row.get("h_bond_donors", "?")
        hba = row.get("h_bond_acceptors", "?")
        violations = row.get("lipinski_violations", "?")
        drug_like = row.get("drug_like", False)
        status = row.get("status", "candidate")
        
        lipinski_badge = '<span class="badge badge-success">✓ Pass</span>' if drug_like else f'<span class="badge badge-danger">✗ {violations} violations</span>'
        status_badge = '<span class="badge badge-info">Candidate</span>' if status == "repurposing_candidate" else '<span class="badge badge-success">Approved</span>'
        
        property_rows += f"<tr><td><strong>{name.capitalize()}</strong></td><td>{mw}</td><td>{logp}</td><td>{hbd}</td><td>{hba}</td><td>{lipinski_badge}</td><td>{status_badge}</td></tr>\n"
    
    # Visualization references
    safe_disease = disease.replace(" ", "_").replace("'", "")
    viz_html = ""
    prop_plot = output_dir / f"property_comparison_{safe_disease}.png"
    heatmap_plot = output_dir / f"lipinski_heatmap_{safe_disease}.png"
    
    if prop_plot.exists():
        # Copy to output with relative path
        viz_html += f'<h3>Property Comparison</h3>\n<img src="property_comparison_{safe_disease}.png" alt="Property Comparison">\n'
    if heatmap_plot.exists():
        viz_html += f'<h3>Lipinski Compliance Heatmap</h3>\n<img src="lipinski_heatmap_{safe_disease}.png" alt="Lipinski Heatmap">\n'
    
    if not viz_html:
        viz_html = "<p><em>Run Module 2 to generate visualizations.</em></p>"
    
    # Candidate evaluations
    eval_html = ""
    for i, ev in enumerate(evaluations):
        score = ev.get("confidence_score", 0)
        rec = ev.get("overall_recommendation", "Unknown")
        
        if "Strong" in rec:
            rec_badge = "badge-success"
        elif "Promising" in rec:
            rec_badge = "badge-warning"
        elif "Moderate" in rec:
            rec_badge = "badge-info"
        else:
            rec_badge = "badge-danger"
        
        risks = "".join(f"<li>{r}</li>" for r in ev.get("key_risks", []))
        steps = "".join(f"<li>{s}</li>" for s in ev.get("recommended_next_steps", []))
        
        eval_html += f"""
        <div class="candidate-card">
            <h4>#{i+1} {ev['drug_name'].capitalize()} 
                <span class="badge {rec_badge}">{rec}</span>
                <span style="float:right; font-size:0.9em;">
                    Score: {score}/10
                    <span class="score-bar"><span class="score-fill" style="width:{score*10}%"></span></span>
                </span>
            </h4>
            <p><strong>Original indication:</strong> {ev.get('original_indication', 'N/A')}</p>
            <p><strong>Proposed mechanism:</strong> {ev.get('proposed_mechanism', 'N/A')}</p>
            <p><strong>Evidence:</strong> {ev.get('evidence_from_literature', 'N/A')}</p>
            <div class="grid-2" style="margin-top:10px;">
                <div>
                    <h4>⚠️ Key Risks</h4>
                    <ul style="margin-left:20px; font-size:0.9em;">{risks}</ul>
                </div>
                <div>
                    <h4>📋 Next Steps</h4>
                    <ul style="margin-left:20px; font-size:0.9em;">{steps}</ul>
                </div>
            </div>
        </div>
        """
    
    if not eval_html:
        eval_html = "<p><em>Run Module 3 to generate evaluations.</em></p>"
    
    # Summary stats
    drug_like_count = sum(1 for e in evaluations if e.get("drug_likeness_assessment", "").startswith("Drug"))
    top_score = evaluations[0]["confidence_score"] if evaluations else 0
    
    summary = (
        f"This analysis evaluated {len(evaluations)} drug repurposing candidates for {disease}. "
        f"{drug_like_count} candidates pass Lipinski's Rule of Five for drug-likeness. "
        f"The top-ranked candidate is <strong>{evaluations[0]['drug_name'].capitalize() if evaluations else 'N/A'}</strong> "
        f"with a confidence score of {top_score}/10. "
        f"These results should be validated with experimental studies."
    )
    
    # ── Render ──────────────────────────────────────────────────────────
    
    html = REPORT_TEMPLATE.format(
        disease=disease,
        date=datetime.now().strftime("%B %d, %Y"),
        total_papers=kb.get("total_papers", 0),
        total_candidates=len(evaluations),
        drug_like_count=drug_like_count,
        top_score=top_score,
        summary=summary,
        drug_mention_rows=drug_mention_rows,
        key_findings_html=findings_html,
        property_rows=property_rows,
        visualization_html=viz_html,
        candidate_evaluations_html=eval_html,
    )
    
    # Save
    report_path = output_dir / f"report_{safe_disease}.html"
    with open(report_path, "w") as f:
        f.write(html)
    
    print(f"📄 Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    print("This module is designed to be imported and used with the full pipeline.")
    print("See notebooks/DrugDiscovery_Demo.ipynb for the end-to-end demo.")
