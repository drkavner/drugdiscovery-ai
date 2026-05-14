"""
DrugDiscovery.ai — AI-Powered Drug Repurposing Pipeline
Module 2: Molecular Property Analysis

Fetches drug/compound data from PubChem PUG-REST API,
computes ADMET-relevant properties, checks Lipinski's Rule of Five,
and generates comparison visualizations.
"""

import requests
import json
import time
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ──────────────────────────────────────────────────────────

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Lipinski's Rule of Five thresholds
LIPINSKI = {
    "molecular_weight": ("≤ 500 Da", 500),
    "log_p": ("≤ 5", 5),
    "h_bond_donors": ("≤ 5", 5),
    "h_bond_acceptors": ("≤ 10", 10),
}

# ── PubChem API ────────────────────────────────────────────────────────────

def get_cid_by_name(name: str) -> int | None:
    """Get PubChem CID for a compound by name."""
    import urllib.parse
    url = f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception as e:
        print(f"   ⚠️  Error looking up '{name}': {e}")
    return None


def get_compound_properties(cid: int) -> dict | None:
    """Fetch key molecular properties from PubChem for a given CID."""
    props = [
        "MolecularWeight", "XLogP", "HBondDonorCount", "HBondAcceptorCount",
        "TPSA", "RotatableBondCount", "MolecularFormula", "IUPACName",
        "CanonicalSMILES", "IsomericSMILES", "Charge",
        "HeavyAtomCount", "AtomStereoCount", "DefinedAtomStereoCount",
        "BondStereoCount", "DefinedBondStereoCount", "CovalentUnitCount",
        "Volume3D", "XStericQuadrupole3D", "YStericQuadrupole3D",
        "ZStericQuadrupole3D", "FeatureCount3D", "FeatureAcceptorCount3D",
        "FeatureDonorCount3D", "FeatureAnionCount3D", "FeatureCationCount3D",
        "FeatureRingCount3D", "FeatureHydrophobeCount3D", "ConformerModelRMSD3D",
        "EffectiveRotorCount3D", "ConformerCount3D",
    ]
    
    prop_str = ",".join(props)
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/{prop_str}/JSON"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if props_list:
                return props_list[0]
    except Exception as e:
        print(f"   ⚠️  Error fetching properties for CID {cid}: {e}")
    return None


def get_compound_properties_by_name(name: str) -> dict | None:
    """Fetch properties directly by compound name."""
    import urllib.parse
    props = [
        "MolecularWeight", "XLogP", "HBondDonorCount", "HBondAcceptorCount",
        "TPSA", "RotatableBondCount", "MolecularFormula", "IUPACName",
        "CanonicalSMILES",
    ]
    prop_str = ",".join(props)
    url = f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name)}/property/{prop_str}/JSON"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if props_list:
                return props_list[0]
    except Exception as e:
        print(f"   ⚠️  Error fetching properties for '{name}': {e}")
    return None


# ── Lipinski's Rule of Five ────────────────────────────────────────────────

def check_lipinski(props: dict) -> dict:
    """
    Evaluate Lipinski's Rule of Five compliance.
    Returns dict with pass/fail for each rule and overall assessment.
    """
    results = {}
    violations = 0
    
    mw = safe_float(props.get("MolecularWeight"))
    logp = safe_float(props.get("XLogP"))
    hbd = safe_int(props.get("HBondDonorCount"))
    hba = safe_int(props.get("HBondAcceptorCount"))
    
    if mw is not None:
        passed = mw <= LIPINSKI["molecular_weight"][1]
        results["molecular_weight"] = {
            "value": mw,
            "threshold": LIPINSKI["molecular_weight"][0],
            "passed": passed,
        }
        if not passed:
            violations += 1
    
    if logp is not None:
        passed = logp <= LIPINSKI["log_p"][1]
        results["log_p"] = {
            "value": logp,
            "threshold": LIPINSKI["log_p"][0],
            "passed": passed,
        }
        if not passed:
            violations += 1
    
    if hbd is not None:
        passed = hbd <= LIPINSKI["h_bond_donors"][1]
        results["h_bond_donors"] = {
            "value": hbd,
            "threshold": LIPINSKI["h_bond_donors"][0],
            "passed": passed,
        }
        if not passed:
            violations += 1
    
    if hba is not None:
        passed = hba <= LIPINSKI["h_bond_acceptors"][1]
        results["h_bond_acceptors"] = {
            "value": hba,
            "threshold": LIPINSKI["h_bond_acceptors"][0],
            "passed": passed,
        }
        if not passed:
            violations += 1
    
    results["total_violations"] = violations
    results["drug_like"] = violations <= 1  # Lipinski: ≤1 violation = drug-like
    results["verdict"] = "✅ Drug-like" if violations <= 1 else f"❌ {violations} violation(s)"
    
    return results


# ── Helper Functions ───────────────────────────────────────────────────────

def safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None

def safe_int(val) -> int | None:
    try:
        return int(float(val)) if val is not None else None
    except (ValueError, TypeError):
        return None


# ── Batch Analysis ─────────────────────────────────────────────────────────

def analyze_drugs(drug_names: list[str], approved_drugs: list[str] = None) -> pd.DataFrame:
    """
    Fetch properties for a list of drug names and return a DataFrame.
    
    Args:
        drug_names: List of drug/compound names to analyze
        approved_drugs: Optional list of approved drugs for comparison
    
    Returns:
        pandas DataFrame with molecular properties and Lipinski assessment
    """
    all_names = list(drug_names)
    if approved_drugs:
        for d in approved_drugs:
            if d.lower() not in [n.lower() for n in all_names]:
                all_names.append(d)
    
    print(f"\n{'='*60}")
    print(f"  DrugDiscovery.ai — Molecular Property Analysis")
    print(f"  Compounds: {len(all_names)}")
    print(f"{'='*60}\n")
    
    records = []
    for i, name in enumerate(all_names):
        print(f"  [{i+1}/{len(all_names)}] Fetching: {name}...")
        
        props = get_compound_properties_by_name(name)
        time.sleep(0.35)  # PubChem rate limit: ≤5 requests/second
        
        if props is None:
            print(f"         ⚠️  Not found in PubChem, trying CID lookup...")
            cid = get_cid_by_name(name)
            time.sleep(0.35)
            if cid:
                props = get_compound_properties(cid)
                time.sleep(0.35)
        
        if props:
            lipinski = check_lipinski(props)
            record = {
                "name": name,
                "cid": props.get("CID"),
                "molecular_formula": props.get("MolecularFormula"),
                "molecular_weight": safe_float(props.get("MolecularWeight")),
                "log_p": safe_float(props.get("XLogP")),
                "h_bond_donors": safe_int(props.get("HBondDonorCount")),
                "h_bond_acceptors": safe_int(props.get("HBondAcceptorCount")),
                "tpsa": safe_float(props.get("TPSA")),
                "rotatable_bonds": safe_int(props.get("RotatableBondCount")),
                "smiles": props.get("CanonicalSMILES"),
                "lipinski_violations": lipinski.get("total_violations", None),
                "drug_like": lipinski.get("drug_like", None),
                "status": "repurposing_candidate" if name in drug_names and (approved_drugs is None or name not in approved_drugs) else "approved_drug",
            }
            records.append(record)
            print(f"         ✓ MW={record['molecular_weight']}, LogP={record['log_p']}, Lipinski={lipinski.get('verdict')}")
        else:
            print(f"         ✗ Could not retrieve data")
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        # Save to CSV
        csv_path = DATA_DIR / "molecular_properties.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved to: {csv_path}")
    
    return df


# ── Visualization ──────────────────────────────────────────────────────────

def plot_property_comparison(df: pd.DataFrame, disease: str, save: bool = True) -> Path:
    """
    Generate comparison plots for molecular properties.
    Returns path to saved figure.
    """
    if df.empty:
        print("⚠️  No data to plot")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"DrugDiscovery.ai — Molecular Property Analysis\n{disease}", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    candidate_color = "#2196F3"  # Blue
    approved_color = "#4CAF50"   # Green
    colors = {
        "repurposing_candidate": candidate_color,
        "approved_drug": approved_color,
    }
    
    plot_configs = [
        ("molecular_weight", "Molecular Weight (Da)", "MW (Da)"),
        ("log_p", "LogP (Partition Coefficient)", "LogP"),
        ("h_bond_donors", "H-Bond Donors", "Count"),
        ("h_bond_acceptors", "H-Bond Acceptors", "Count"),
        ("tpsa", "Topological Polar Surface Area (Ų)", "TPSA (Ų)"),
        ("rotatable_bonds", "Rotatable Bonds", "Count"),
    ]
    
    for ax, (col, title, xlabel) in zip(axes.flatten(), plot_configs):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        
        # Separate candidates and approved
        candidates = df[df["status"] == "repurposing_candidate"]
        approved = df[df["status"] == "approved_drug"]
        
        # Box plots
        data_to_plot = []
        labels = []
        box_colors = []
        
        if not candidates.empty and candidates[col].notna().any():
            data_to_plot.append(candidates[col].dropna().values)
            labels.append(f"Candidates\n(n={len(candidates)})")
            box_colors.append(candidate_color)
        
        if not approved.empty and approved[col].notna().any():
            data_to_plot.append(approved[col].dropna().values)
            labels.append(f"Approved\n(n={len(approved)})")
            box_colors.append(approved_color)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           widths=0.6, showmeans=True,
                           meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add individual points
            for j, (data, color) in enumerate(zip(data_to_plot, box_colors)):
                x = np.random.normal(j + 1, 0.06, size=len(data))
                ax.scatter(x, data, alpha=0.7, s=40, color=color, edgecolors="white", linewidth=0.5, zorder=5)
        
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(xlabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        
        # Add Lipinski threshold lines where applicable
        if col == "molecular_weight":
            ax.axhline(y=500, color="red", linestyle="--", alpha=0.5, label="Lipinski threshold")
            ax.legend(fontsize=8)
        elif col == "log_p":
            ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Lipinski threshold")
            ax.legend(fontsize=8)
        elif col == "h_bond_donors":
            ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Lipinski threshold")
            ax.legend(fontsize=8)
        elif col == "h_bond_acceptors":
            ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Lipinski threshold")
            ax.legend(fontsize=8)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=candidate_color, alpha=0.7, label="Repurposing Candidates"),
        mpatches.Patch(color=approved_color, alpha=0.7, label="Approved Drugs"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, -0.04))
    
    plt.tight_layout()
    
    if save:
        safe_disease = disease.replace(" ", "_").replace("'", "")
        path = OUTPUT_DIR / f"property_comparison_{safe_disease}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"📊 Saved plot: {path}")
        return path
    
    return None


def plot_lipinski_heatmap(df: pd.DataFrame, disease: str, save: bool = True) -> Path:
    """
    Generate a Lipinski compliance heatmap.
    """
    if df.empty:
        return None
    
    lipinski_cols = ["molecular_weight", "log_p", "h_bond_donors", "h_bond_acceptors"]
    thresholds = {"molecular_weight": 500, "log_p": 5, "h_bond_donors": 5, "h_bond_acceptors": 10}
    display_names = {
        "molecular_weight": "MW ≤ 500",
        "log_p": "LogP ≤ 5",
        "h_bond_donors": "HBD ≤ 5",
        "h_bond_acceptors": "HBA ≤ 10",
    }
    
    # Build pass/fail matrix
    matrix = []
    labels = []
    for _, row in df.iterrows():
        drug_row = []
        for col in lipinski_cols:
            val = row.get(col)
            if val is None:
                drug_row.append(None)
            else:
                drug_row.append(1 if val <= thresholds[col] else 0)
        matrix.append(drug_row)
        label = f"{row['name']}"
        if row.get("status") == "approved_drug":
            label += " (approved)"
        labels.append(label)
    
    matrix = np.array(matrix, dtype=float)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))
    
    # Custom colormap: green=pass, red=fail
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f44336", "#4CAF50"])
    
    # Mask None values
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(lipinski_cols)))
    ax.set_xticklabels([display_names[c] for c in lipinski_cols], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(lipinski_cols)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = "✓" if val == 1 else "✗"
                color = "white" if val == 0 else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=14,
                       fontweight="bold", color=color)
    
    ax.set_title(f"Lipinski's Rule of Five Compliance\n{disease}",
                fontsize=14, fontweight="bold", pad=15)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Pass"),
        mpatches.Patch(color="#f44336", label="Fail"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    
    if save:
        safe_disease = disease.replace(" ", "_").replace("'", "")
        path = OUTPUT_DIR / f"lipinski_heatmap_{safe_disease}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"📊 Saved heatmap: {path}")
        return path
    
    return None


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_molecular_analysis(
    candidate_drugs: list[str],
    disease: str,
    approved_drugs: list[str] = None,
) -> pd.DataFrame:
    """
    Full molecular property analysis pipeline.
    
    Args:
        candidate_drugs: List of repurposing candidate drug names
        disease: Target disease (for labeling)
        approved_drugs: Optional list of approved drugs for comparison
    
    Returns:
        pandas DataFrame with all properties
    """
    df = analyze_drugs(candidate_drugs, approved_drugs)
    
    if not df.empty:
        print(f"\n📋 Property Summary:")
        print(df[["name", "molecular_weight", "log_p", "h_bond_donors", 
                   "h_bond_acceptors", "lipinski_violations", "drug_like", "status"]].to_string(index=False))
        
        # Generate plots
        print(f"\n📊 Generating visualizations...")
        plot_property_comparison(df, disease)
        plot_lipinski_heatmap(df, disease)
    
    return df


if __name__ == "__main__":
    import sys
    
    # Default demo: Alzheimer's repurposing candidates
    candidates = ["metformin", "rapamycin", "lithium", "memantine"]
    approved = ["donepezil", "rivastigmine", "galantamine"]
    disease = "Alzheimer's disease"
    
    if len(sys.argv) > 1:
        disease = sys.argv[1]
    
    df = run_molecular_analysis(candidates, disease, approved)
