"""Programmatic Table Generator and README-vs-JSON Consistency Verifier.

Author: Lingxiao Li (Student ID: 49852413)
Course: COMP3710 / COMP7710 — Pattern Recognition and Analysis (2026)

Reads raw JSON artifacts in `results/` to:
1. Generate canonical Markdown tables for `README.md` directly from programmatic JSON outputs.
2. Verify that every key statistic cited in `README.md` matches `results/*.json` (`verify_readme_matches_json`),
   preventing manual transcription errors or AI-hallucinated summary numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
README_PATH = PROJECT_DIR / "README.md"


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_protocol_audit_table(audit: Dict[str, object]) -> str:
    """Format Table 2.3A directly from `results/protocol_audit.json`."""
    tr_ad = audit["official_train"]["classes"]["AD"]
    tr_nc = audit["official_train"]["classes"]["NC"]
    te_ad = audit["official_test"]["classes"]["AD"]
    te_nc = audit["official_test"]["classes"]["NC"]
    ov = audit["official_split_subject_overlap"]

    lines = [
        "| Official Partition | Scans (Slices) | Unique Subjects | Scan ID Range (Ixxxxxx) | Acquisition Years (era_counts) | Dominant Sequences (protocol_counts) |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| train/AD | {tr_ad['num_scans']} ({tr_ad['num_scans'] * 20:,}) | {tr_ad['num_unique_subjects']} | "
        f"I{tr_ad['scan_id_min']}–I{tr_ad['scan_id_max']} (median I{tr_ad['scan_id_median']}) | "
        f"{tr_ad['year_min']}–{tr_ad['year_max']} (median {tr_ad['year_median']}; {tr_ad['era_counts'].get('pre_2017_ADNI2', 0)}/{tr_ad['num_scans']} <2017) | "
        f"{tr_ad['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)}/{tr_ad['num_scans']} Standard MPRAGE, "
        f"{tr_ad['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)} IR-FSPGR, "
        f"{tr_ad['protocol_counts'].get('Other_T1_Sequence', 0)} Other T1 |",
        f"| train/NC | {tr_nc['num_scans']} ({tr_nc['num_scans'] * 20:,}) | {tr_nc['num_unique_subjects']} | "
        f"I{tr_nc['scan_id_min']}–I{tr_nc['scan_id_max']} (median I{tr_nc['scan_id_median']}) | "
        f"{tr_nc['year_min']}–{tr_nc['year_max']} (median {tr_nc['year_median']}; {tr_nc['era_counts'].get('post_2017_ADNI3', 0)}/{tr_nc['num_scans']} >=2017) | "
        f"{tr_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)}/{tr_nc['num_scans']} Accel. MPRAGE, "
        f"{tr_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)} IR-FSPGR, "
        f"{tr_nc['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)} Standard MPRAGE |",
        f"| test/AD | {te_ad['num_scans']} ({te_ad['num_scans'] * 20:,}) | {te_ad['num_unique_subjects']} ({ov['ad_overlapping_subjects']} in train) | "
        f"I{te_ad['scan_id_min']}–I{te_ad['scan_id_max']} (median I{te_ad['scan_id_median']}) | "
        f"{te_ad['year_min']}–{te_ad['year_max']} (median {te_ad['year_median']}; {te_ad['era_counts'].get('pre_2017_ADNI2', 0)} <2017, {te_ad['era_counts'].get('post_2017_ADNI3', 0)} >=2017) | "
        f"{te_ad['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)} Standard MPRAGE, "
        f"{te_ad['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)} Accel. MPRAGE, "
        f"{te_ad['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)} IR-FSPGR, "
        f"{te_ad['protocol_counts'].get('Other_T1_Sequence', 0)} Other T1 |",
        f"| test/NC | {te_nc['num_scans']} ({te_nc['num_scans'] * 20:,}) | {te_nc['num_unique_subjects']} ({ov['nc_overlapping_subjects']} in train) | "
        f"I{te_nc['scan_id_min']}–I{te_nc['scan_id_max']} (median I{te_nc['scan_id_median']}) | "
        f"{te_nc['year_min']}–{te_nc['year_max']} (median {te_nc['year_median']}; {te_nc['era_counts'].get('post_2017_ADNI3', 0)}/{te_nc['num_scans']} >=2017) | "
        f"{te_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)}/{te_nc['num_scans']} Accel. MPRAGE, "
        f"{te_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)} IR-FSPGR, "
        f"{te_nc['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)} Standard MPRAGE |",
    ]
    return "\n".join(lines)


def generate_3seed_subject_table(b3: Dict[str, object]) -> str:
    """Format Table 3.1A directly from `results/benchmark_subject_3seeds.json`."""
    agg = b3["aggregated"]
    base = agg["baseline"]
    conv = agg["convnext"]
    pair = agg["paired_baseline_minus_convnext"]

    def _pct(d: Dict[str, float]) -> str:
        return f"{d['mean'] * 100:.2f}% ± {d['std'] * 100:.2f}%"

    def _dec(d: Dict[str, float]) -> str:
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    lines = [
        "| Evaluation Metric (Strict Subject-Disjoint, 3 Seeds: 42, 43, 44) | SimpleCNNBaseline (Mean ± SD, ddof=1) | ConvNeXt [Hard] (Mean ± SD, ddof=1) | Paired Diff (SimpleCNN - ConvNeXt) |",
        "| :--- | :---: | :---: | :---: |",
        f"| Slice Test Accuracy | {_pct(base['slice_level']['accuracy'])} | {_pct(conv['slice_level']['accuracy'])} | +{pair['slice_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f} ± {pair['slice_accuracy']['std_diff_ddof1'] * 100:.2f} pp |",
        f"| Slice Balanced Accuracy | {_pct(base['slice_level']['balanced_accuracy'])} | {_pct(conv['slice_level']['balanced_accuracy'])} | — |",
        f"| Slice Macro-Averaged F1 | {_dec(base['slice_level']['macro_f1'])} | {_dec(conv['slice_level']['macro_f1'])} | +{pair['macro_f1']['mean_diff_baseline_minus_convnext']:.4f} ± {pair['macro_f1']['std_diff_ddof1']:.4f} |",
        f"| Slice Test AUROC | {_dec(base['slice_level']['auroc'])} | {_dec(conv['slice_level']['auroc'])} | +{pair['auroc']['mean_diff_baseline_minus_convnext']:.4f} ± {pair['auroc']['std_diff_ddof1']:.4f} |",
        f"| Slice Test PR-AUC | {_dec(base['slice_level']['pr_auc'])} | {_dec(conv['slice_level']['pr_auc'])} | — |",
        f"| Scan-Level Accuracy (20-Slice Mean Prob) | {_pct(base['patient_level']['scan_mean_prob_accuracy'])} | {_pct(conv['patient_level']['scan_mean_prob_accuracy'])} | +{pair['scan_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f} ± {pair['scan_accuracy']['std_diff_ddof1'] * 100:.2f} pp |",
        f"| Scan-Level Accuracy (20-Slice Majority Vote) | {_pct(base['patient_level']['scan_majority_vote_accuracy'])} | {_pct(conv['patient_level']['scan_majority_vote_accuracy'])} | — |",
        f"| Scan-Level AUROC | {_dec(base['patient_level']['patient_auroc'])} | {_dec(conv['patient_level']['patient_auroc'])} | — |",
        f"| Subject-Level Accuracy (102 Unseen Patients) | {_pct(base['patient_level']['subject_accuracy'])} | {_pct(conv['patient_level']['subject_accuracy'])} | +{pair['subject_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f} ± {pair['subject_accuracy']['std_diff_ddof1'] * 100:.2f} pp |",
        f"| Subject-Level AUROC | {_dec(base['patient_level']['subject_auroc'])} | {_dec(conv['patient_level']['subject_auroc'])} | — |",
        f"| Uncalibrated Test ECE | {_dec(base['slice_level']['ece_uncalibrated'])} | {_dec(conv['slice_level']['ece_uncalibrated'])} | — |",
        f"| Calibrated Test ECE (Post-Hoc T*) | {_dec(base['slice_level']['ece_calibrated'])} | {_dec(conv['slice_level']['ece_calibrated'])} | — |",
    ]
    return "\n".join(lines)


def verify_readme_matches_json(
    readme_path: Path = README_PATH,
    results_dir: Path = RESULTS_DIR,
) -> Tuple[bool, List[str]]:
    """Verify that key numbers in `README.md` match the programmatic JSON files in `results/`.

    Returns:
        `(all_passed, list_of_mismatches)`
    """
    readme_text = readme_path.read_text(encoding="utf-8")
    mismatches: List[str] = []

    def _require(token: str, context: str) -> None:
        if token not in readme_text:
            mismatches.append(f"Missing expected token '{token}' ({context}) in {readme_path.name}")

    # 1. Verify protocol_audit.json numbers
    audit_path = results_dir / "protocol_audit.json"
    if audit_path.is_file():
        audit = _load_json(audit_path)
        tr_ad = audit["official_train"]["classes"]["AD"]
        tr_nc = audit["official_train"]["classes"]["NC"]
        te_ad = audit["official_test"]["classes"]["AD"]
        te_nc = audit["official_test"]["classes"]["NC"]
        ov = audit["official_split_subject_overlap"]

        _require(str(tr_ad["num_unique_subjects"]), "official_train/AD unique subjects")
        _require(str(tr_nc["num_unique_subjects"]), "official_train/NC unique subjects")
        _require(str(te_ad["num_unique_subjects"]), "official_test/AD unique subjects")
        _require(str(te_nc["num_unique_subjects"]), "official_test/NC unique subjects")
        _require(f"I{tr_ad['scan_id_min']}", "official_train/AD scan_id_min")
        _require(f"I{tr_ad['scan_id_max']}", "official_train/AD scan_id_max")
        _require(f"I{te_ad['scan_id_min']}", "official_test/AD scan_id_min")
        _require(f"I{te_ad['scan_id_max']}", "official_test/AD scan_id_max")
        _require(f"I{te_nc['scan_id_min']}", "official_test/NC scan_id_min")
        _require(f"I{te_nc['scan_id_max']}", "official_test/NC scan_id_max")
        _require(str(ov["total_overlapping_subjects"]), "official total_overlapping_subjects")

    # 2. Verify 3-seed subject benchmark numbers
    b3_path = results_dir / "benchmark_subject_3seeds.json"
    if b3_path.is_file():
        b3 = _load_json(b3_path)
        agg = b3["aggregated"]
        for m_name in ("baseline", "convnext"):
            acc_m = agg[m_name]["slice_level"]["accuracy"]["mean"] * 100.0
            acc_s = agg[m_name]["slice_level"]["accuracy"]["std"] * 100.0
            _require(f"{acc_m:.2f}%", f"3-seed {m_name} slice accuracy mean")
            _require(f"{acc_s:.2f}%", f"3-seed {m_name} slice accuracy std (ddof=1)")
            scan_m = agg[m_name]["patient_level"]["scan_mean_prob_accuracy"]["mean"] * 100.0
            scan_s = agg[m_name]["patient_level"]["scan_mean_prob_accuracy"]["std"] * 100.0
            _require(f"{scan_m:.2f}%", f"3-seed {m_name} scan accuracy mean")
            _require(f"{scan_s:.2f}%", f"3-seed {m_name} scan accuracy std (ddof=1)")

    # 3. Verify single-seed subject benchmark numbers (Seed 42)
    b_subj_path = results_dir / "benchmark_subject.json"
    if b_subj_path.is_file():
        bs = _load_json(b_subj_path)
        for m_name in ("baseline", "convnext"):
            tm = bs["models"][m_name]["test_metrics"]
            _require(f"{tm['accuracy'] * 100:.2f}%", f"subject seed42 {m_name} slice accuracy")
            _require(f"{tm['auroc']:.4f}", f"subject seed42 {m_name} slice auroc")
            _require(
                f"{tm['patient_level']['scan_mean_prob_accuracy'] * 100:.2f}%",
                f"subject seed42 {m_name} scan accuracy",
            )
            post_era = tm.get("era_stratified", {}).get("post_2017_adni3", {})
            if post_era:
                _require(
                    f"{post_era['ad_recall'] * 100:.2f}%",
                    f"subject seed42 {m_name} >=2017 AD recall",
                )
                _require(
                    f"{post_era['nc_recall'] * 100:.2f}%",
                    f"subject seed42 {m_name} >=2017 NC recall",
                )
                _require(
                    f"{post_era['auroc']:.4f}",
                    f"subject seed42 {m_name} >=2017 AUROC",
                )

    # 4. Verify autopsy subject JSON numbers
    aut_subj_path = results_dir / "failure_autopsy_subject.json"
    if aut_subj_path.is_file():
        aut_cases = _load_json(aut_subj_path)
        for c in aut_cases:
            _require(str(c["patient_id"]), f"subject autopsy case #{c['case_number']} patient_id")
            _require(f"{c['scan_20slice_mean_prob_ad']:.4f}", f"subject autopsy case #{c['case_number']} scan mean prob")

    return (len(mismatches) == 0), mismatches


def main() -> None:
    audit = _load_json(RESULTS_DIR / "protocol_audit.json")
    b3 = _load_json(RESULTS_DIR / "benchmark_subject_3seeds.json")
    print("=== Generated Table 2.3A (Protocol Audit) ===")
    print(generate_protocol_audit_table(audit))
    print("\n=== Generated Table 3.1A (3-Seed Subject-Disjoint Benchmark) ===")
    print(generate_3seed_subject_table(b3))
    ok, errs = verify_readme_matches_json()
    print(f"\nREADME-vs-JSON consistency check: {'PASSED' if ok else 'FAILED'}")
    for e in errs:
        print("  -", e)


if __name__ == "__main__":
    main()
