# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.10+ required for `int | None` syntax)
pip install -r requirements.txt

# Run the full 4-phase pipeline (defaults to "Alzheimer's disease")
python run_pipeline.py "Alzheimer's disease"

# Run individual modules
python -m src.literature_mining "Alzheimer's disease"   # writes data/kb_*.json
python -m src.molecular_analysis "Alzheimer's disease"  # writes data/molecular_properties.csv + output/*.png

# Open the demo notebook
jupyter notebook notebooks/DrugDiscovery_Demo.ipynb
```

No test suite, linter, or type checker is configured — the `tests/` directory referenced in the README does not exist.

## Architecture

The pipeline is four modules chained by `run_pipeline.py`. Each module exposes a top-level `run_*` function that takes the previous module's output and produces an artifact on disk. The runner is the only place the modules are wired together — they do not import each other.

```
literature_mining.run_literature_mining(disease)
    → kb: dict                                 → data/kb_<disease>_<date>.json
molecular_analysis.run_molecular_analysis(candidates, disease, approved)
    → pd.DataFrame                             → data/molecular_properties.csv
                                                 output/property_comparison_<disease>.png
                                                 output/lipinski_heatmap_<disease>.png
llm_reasoning.run_llm_reasoning(kb, df, disease)
    → list[dict] (evaluations)                 → data/evaluations.json
report_generator.generate_report(disease, kb, df, evaluations)
    → Path to HTML                             → output/report_<disease>.html
```

### Data contract between modules

The `kb` dict (Module 1 → 3) carries `drug_mentions: {drug: {count, papers}}`, `papers: [...]`, and `all_findings: [...]`. `run_pipeline.py` pulls candidate drug names from `kb["drug_mentions"].keys()[:8]`, so any change to that key shape breaks the runner.

The molecular DataFrame uses a `status` column with values `repurposing_candidate` or `approved_drug` — Module 3 filters on this, and Module 4's plotting logic colors by it.

### Module-specific notes

- **`literature_mining.py`**: drug/disease extraction is plain dictionary matching against the hard-coded `KNOWN_DRUGS` and `DISEASE_KEYWORDS` sets with word-boundary regex — not NER. New drugs must be added to `KNOWN_DRUGS` or they won't be picked up from arXiv abstracts. arXiv calls go through `arxiv.Client(delay_seconds=5, num_retries=3)`; expect rate-limiting on repeated runs.
- **`molecular_analysis.py`**: PubChem PUG-REST is rate-limited to ≤5 req/s — the 0.35s `time.sleep()` after every call enforces this and must be preserved. Each compound takes 2 calls (properties by name, fallback CID lookup), so 10 compounds ≈ 7s minimum. Uses `matplotlib.use("Agg")` for headless plotting.
- **`llm_reasoning.py`**: dispatches to one of four backends, auto-detected from environment. Priority: `ANTHROPIC_API_KEY` → claude, else `OPENROUTER_API_KEY` → openrouter, else `OLLAMA_HOST` → ollama, else the deterministic `generate_evaluation_heuristic` (0–10 score from literature count + Lipinski + MW/LogP ranges). `LLM_BACKEND=` overrides detection. Claude path uses the Anthropic SDK with `output_config.format=json_schema`; OpenRouter and Ollama share the OpenAI SDK with `response_format=json_schema,strict=true`. Any backend exception falls back to the heuristic so the pipeline never crashes on a misconfigured key. Schema (`DRUG_EVALUATION_SCHEMA`) is strict-mode compliant: `additionalProperties: false` and every property listed in `required`.
- **`report_generator.py`**: HTML template is a single inline f-string with CSS — no Jinja2 despite the README. Images are referenced by relative path and must already exist in `output/` next to the report.

### Disease-specific defaults

`run_pipeline.py` has hard-coded default candidate and approved-drug lists for keys `alzheimer`, `parkinson`, and `covid` (substring match on the disease arg). When literature mining returns zero drugs, these defaults are used instead — so for a new disease, the pipeline falls back to `["metformin", "rapamycin", "lithium"]` with approved `["metformin"]`.

### Paths

All modules resolve paths relative to `__file__.parent.parent`, so `data/` and `output/` are always at the repo root regardless of CWD. The `.gitignore` excludes generated CSVs/JSONs/PNGs/HTMLs but keeps the directories.
