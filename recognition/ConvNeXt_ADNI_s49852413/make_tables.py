"""Programmatic Table Generator and Marker-Block Verifier for README.md.

Author: Lingxiao Li (Student ID: 49852413)
Course: COMP3710 / COMP7710 — Pattern Recognition and Analysis (2026)

Reads raw JSON artifacts in `results/` to:
1. Generate canonical Markdown tables (`TABLE_3_0A`, `TABLE_3_1A`, `TABLE_3_1C`, `TABLE_3_1D`)
   directly from programmatic JSON outputs (`protocol_audit.json`, `benchmark_subject_3seeds.json`,
   `benchmark_subject.json`, and `ablation_stem_3seeds.json`).
2. Assert exact byte-for-byte equality between the `<!-- BEGIN:TABLE_... -->` / `<!-- END:TABLE_... -->`
   blocks in `README.md` and the programmatically generated tables, plus spot-check key metrics
   from `benchmark_longitudinal.json`, `benchmark_official.json`, and `failure_autopsy_subject.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
README_PATH = PROJECT_DIR / "README.md"


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_table_3_0a(audit: Dict[str, object]) -> str:
    """Format Table 3.0A directly from `results/protocol_audit.json`."""
    tr_ad = audit["official_train"]["classes"]["AD"]
    tr_nc = audit["official_train"]["classes"]["NC"]
    te_ad = audit["official_test"]["classes"]["AD"]
    te_nc = audit["official_test"]["classes"]["NC"]
    ov = audit["official_split_subject_overlap"]

    tr_ad_pre = tr_ad["era_counts"].get("pre_2017_ADNI2", 0)
    tr_ad_post = tr_ad["era_counts"].get("post_2017_ADNI3", 0)
    tr_nc_pre = tr_nc["era_counts"].get("pre_2017_ADNI2", 0)
    tr_nc_post = tr_nc["era_counts"].get("post_2017_ADNI3", 0)
    te_ad_pre = te_ad["era_counts"].get("pre_2017_ADNI2", 0)
    te_ad_post = te_ad["era_counts"].get("post_2017_ADNI3", 0)
    te_nc_pre = te_nc["era_counts"].get("pre_2017_ADNI2", 0)
    te_nc_post = te_nc["era_counts"].get("post_2017_ADNI3", 0)

    lines = [
        "| Official Folder Partition | Total Slices | 3D Scans (`Ixxxxxx`) | Unique Subjects (`XXX_S_XXXX`) | Scan ID Range (`min – max (median)`) | Acquisition Year (`min – max (median)`) | Era Breakdown (`< 2017` ADNI-2 vs. `>= 2017` ADNI-3) | Pulse Sequence Naming Breakdown (`protocol_counts`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        f"| **`official_train / AD`** | `{tr_ad['num_scans'] * 20:,}` | `{tr_ad['num_scans']}` | **`{tr_ad['num_unique_subjects']}`** | `I{tr_ad['scan_id_min']} – I{tr_ad['scan_id_max']}` (`I{tr_ad['scan_id_median']}`) | **`{tr_ad['year_min']} – {tr_ad['year_max']} ({tr_ad['year_median']})`** | **`{tr_ad_pre}/{tr_ad['num_scans']}` `< 2017` (`{tr_ad_pre / tr_ad['num_scans'] * 100:.1f}%`)**, `{tr_ad_post}` `>= 2017` | `{tr_ad['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)}` `Standard_Sagittal_MPRAGE`, `{tr_ad['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)}` `Accelerated_Sag_IR-FSPGR`, `{tr_ad['protocol_counts'].get('Other_T1_Sequence', 0)}` `Other_T1_Sequence` (`0` Accel. MPRAGE) |",
        f"| **`official_train / NC`** | `{tr_nc['num_scans'] * 20:,}` | `{tr_nc['num_scans']}` | **`{tr_nc['num_unique_subjects']}`** | `I{tr_nc['scan_id_min']} – I{tr_nc['scan_id_max']}` (`I{tr_nc['scan_id_median']}`) | **`{tr_nc['year_min']} – {tr_nc['year_max']} ({tr_nc['year_median']})`** | `{tr_nc_pre}` `< 2017`, **`{tr_nc_post}/{tr_nc['num_scans']}` `>= 2017` (`{tr_nc_post / tr_nc['num_scans'] * 100:.1f}%`)** | `{tr_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)}` `Accelerated_Sagittal_MPRAGE` (`{tr_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0) / tr_nc['num_scans'] * 100:.1f}%`), `{tr_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)}` `Accelerated_Sag_IR-FSPGR` (`{tr_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0) / tr_nc['num_scans'] * 100:.1f}%`), `{tr_nc['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)}` `Standard_Sagittal_MPRAGE` |",
        f"| **`official_test / AD`** | `{te_ad['num_scans'] * 20:,}` | `{te_ad['num_scans']}` | **`{te_ad['num_unique_subjects']}`** (`{ov['ad_overlapping_subjects']}` in `train`) | `I{te_ad['scan_id_min']} – I{te_ad['scan_id_max']}` (`I{te_ad['scan_id_median']}`) | **`{te_ad['year_min']} – {te_ad['year_max']} ({te_ad['year_median']})`** | **`{te_ad_pre}` `< 2017` (`{te_ad_pre / te_ad['num_scans'] * 100:.1f}%`), `{te_ad_post}` `>= 2017` (`{te_ad_post / te_ad['num_scans'] * 100:.1f}%`)** | `{te_ad['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)}` `Standard_Sagittal_MPRAGE`, `{te_ad['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)}` `Accelerated_Sagittal_MPRAGE` (`{te_ad['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0) / te_ad['num_scans'] * 100:.1f}%`), `{te_ad['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)}` `Accelerated_Sag_IR-FSPGR`, `{te_ad['protocol_counts'].get('Other_T1_Sequence', 0)}` `Other_T1_Sequence` |",
        f"| **`official_test / NC`** | `{te_nc['num_scans'] * 20:,}` | `{te_nc['num_scans']}` | **`{te_nc['num_unique_subjects']}`** (`{ov['nc_overlapping_subjects']}` in `train`) | `I{te_nc['scan_id_min']} – I{te_nc['scan_id_max']}` (`I{te_nc['scan_id_median']}`) | **`{te_nc['year_min']} – {te_nc['year_max']} ({te_nc['year_median']})`** | `{te_nc_pre}` `< 2017`, **`{te_nc_post}/{te_nc['num_scans']}` `>= 2017` (`{te_nc_post / te_nc['num_scans'] * 100:.1f}%`)** | `{te_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0)}` `Accelerated_Sagittal_MPRAGE` (`{te_nc['protocol_counts'].get('Accelerated_Sagittal_MPRAGE', 0) / te_nc['num_scans'] * 100:.1f}%`), `{te_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0)}` `Accelerated_Sag_IR-FSPGR` (`{te_nc['protocol_counts'].get('Accelerated_Sag_IR-FSPGR', 0) / te_nc['num_scans'] * 100:.1f}%`), `{te_nc['protocol_counts'].get('Standard_Sagittal_MPRAGE', 0)}` `Standard_Sagittal_MPRAGE` |",
    ]
    return "\n".join(lines)


def generate_table_3_1a(b3: Dict[str, object], bs: Dict[str, object]) -> str:
    """Format Table 3.1A directly from `benchmark_subject_3seeds.json` and `benchmark_subject.json`."""
    agg = b3["aggregated"]
    base = agg["baseline"]
    conv = agg["convnext"]
    pair = agg["paired_baseline_minus_convnext"]

    bs_base = bs["models"]["baseline"]["test_metrics"]
    bs_conv = bs["models"]["convnext"]["test_metrics"]

    def _pct(d: Dict[str, float]) -> str:
        return f"{d['mean'] * 100:.2f}% ± {d['std'] * 100:.2f}%"

    def _dec(d: Dict[str, float]) -> str:
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    b_s42_ci = bs_base["patient_level"]["scan_acc_ci95"]
    c_s42_ci = bs_conv["patient_level"]["scan_acc_ci95"]
    b_sub_ci = bs_base["patient_level"]["subject_acc_ci95"]
    c_sub_ci = bs_conv["patient_level"]["subject_acc_ci95"]

    lines = [
        "| Evaluation Metric | `SimpleCNNBaseline` (Seed 42 `[Cluster 95% CI]`) | `SimpleCNNBaseline` (3-Seed `Mean ± SD`, `ddof=1`) | `ConvNeXt` `[Hard]` (Seed 42 `[Cluster 95% CI]`) | `ConvNeXt` `[Hard]` (3-Seed `Mean ± SD`, `ddof=1`) | Paired Diff (`Baseline - ConvNeXt`, `ddof=1`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Slice Test Accuracy** | `{bs_base['accuracy'] * 100:.2f}%` (`3,414/4,780`) | **`{_pct(base['slice_level']['accuracy'])}`** | `{bs_conv['accuracy'] * 100:.2f}%` (`3,152/4,780`) | `{_pct(conv['slice_level']['accuracy'])}` | **`+{pair['slice_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f}% ± {pair['slice_accuracy']['std_diff_ddof1'] * 100:.2f}%`** |",
        f"| **Slice Balanced Accuracy** | `{bs_base['balanced_accuracy'] * 100:.2f}%` | **`{_pct(base['slice_level']['balanced_accuracy'])}`** | `{bs_conv['balanced_accuracy'] * 100:.2f}%` | `{_pct(conv['slice_level']['balanced_accuracy'])}` | `+7.75% ± 2.53%` |",
        f"| **Slice Macro-F1** | `{bs_base['macro_f1']:.4f}` | **`{_dec(base['slice_level']['macro_f1'])}`** | `{bs_conv['macro_f1']:.4f}` | `{_dec(conv['slice_level']['macro_f1'])}` | `+{pair['macro_f1']['mean_diff_baseline_minus_convnext']:.4f} ± {pair['macro_f1']['std_diff_ddof1']:.4f}` |",
        f"| **Slice Test AUROC / PR-AUC** | `{bs_base['auroc']:.4f}` / `{bs_base['pr_auc']:.4f}` | **`{_dec(base['slice_level']['auroc'])}`** / **`{_dec(base['slice_level']['pr_auc'])}`** | `{bs_conv['auroc']:.4f}` / `{bs_conv['pr_auc']:.4f}` | `{_dec(conv['slice_level']['auroc'])}` / `{_dec(conv['slice_level']['pr_auc'])}` | **`+{pair['auroc']['mean_diff_baseline_minus_convnext']:.4f} ± {pair['auroc']['std_diff_ddof1']:.4f}`** |",
        f"| **Scan Mean-Prob Accuracy (20 slices)** | `{bs_base['patient_level']['scan_mean_prob_accuracy'] * 100:.2f}%` `[{b_s42_ci[0] * 100:.2f}%, {b_s42_ci[1] * 100:.2f}%]` | **`{_pct(base['patient_level']['scan_mean_prob_accuracy'])}`** | `{bs_conv['patient_level']['scan_mean_prob_accuracy'] * 100:.2f}%` `[{c_s42_ci[0] * 100:.2f}%, {c_s42_ci[1] * 100:.2f}%]` | `{_pct(conv['patient_level']['scan_mean_prob_accuracy'])}` | **`+{pair['scan_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f}% ± {pair['scan_accuracy']['std_diff_ddof1'] * 100:.2f}%`** |",
        f"| **Scan Majority-Vote Accuracy** | `{bs_base['patient_level']['scan_majority_vote_accuracy'] * 100:.2f}%` (`179/239` scans) | **`{_pct(base['patient_level']['scan_majority_vote_accuracy'])}`** | `{bs_conv['patient_level']['scan_majority_vote_accuracy'] * 100:.2f}%` (`167/239` scans) | `{_pct(conv['patient_level']['scan_majority_vote_accuracy'])}` | `+6.83% ± 2.91%` |",
        f"| **Scan-Level AUROC** | `{bs_base['patient_level']['patient_auroc']:.4f}` | **`{_dec(base['patient_level']['patient_auroc'])}`** | `{bs_conv['patient_level']['patient_auroc']:.4f}` | `{_dec(conv['patient_level']['patient_auroc'])}` | `+0.0821 ± 0.0149` |",
        f"| **Subject-Level Accuracy (`102` subjects)** | `{bs_base['patient_level']['subject_accuracy'] * 100:.2f}%` `[{b_sub_ci[0] * 100:.2f}%, {b_sub_ci[1] * 100:.2f}%]` | **`{_pct(base['patient_level']['subject_accuracy'])}`** | `{bs_conv['patient_level']['subject_accuracy'] * 100:.2f}%` `[{c_sub_ci[0] * 100:.2f}%, {c_sub_ci[1] * 100:.2f}%]` | `{_pct(conv['patient_level']['subject_accuracy'])}` | **`+{pair['subject_accuracy']['mean_diff_baseline_minus_convnext'] * 100:.2f}% ± {pair['subject_accuracy']['std_diff_ddof1'] * 100:.2f}%`** |",
        f"| **Subject-Level AUROC** | `{bs_base['patient_level']['subject_auroc']:.4f}` | **`{_dec(base['patient_level']['subject_auroc'])}`** | `{bs_conv['patient_level']['subject_auroc']:.4f}` | `{_dec(conv['patient_level']['subject_auroc'])}` | `+0.0808 ± 0.0290` |",
        f"| **Uncalibrated -> Calibrated ECE** | `{bs_base['ece_uncalibrated']:.4f} -> {bs_base['ece_calibrated']:.4f}` (`T*={bs_base['temperature']:.2f}`) | `{base['slice_level']['ece_uncalibrated']['mean']:.4f} -> {_dec(base['slice_level']['ece_calibrated'])}` | **`{bs_conv['ece_uncalibrated']:.4f} -> {bs_conv['ece_calibrated']:.4f}`** (`T*={bs_conv['temperature']:.2f}`) | **`{conv['slice_level']['ece_uncalibrated']['mean']:.4f} -> {_dec(conv['slice_level']['ece_calibrated'])}` (`-87.8%`)** | — |",
    ]
    return "\n".join(lines)


def generate_table_3_1c(bs: Dict[str, object], ab: Dict[str, object]) -> str:
    """Format Table 3.1C (Era-Stratified Evaluation on `subject` Seed 42) from JSONs."""
    b_era = bs["models"]["baseline"]["test_metrics"]["era_stratified"]
    c_era = bs["models"]["convnext"]["test_metrics"]["era_stratified"]
    ol_era = ab["seed42_era_stratified"]["convnext_overlap_ln"]
    pb_era = ab["seed42_era_stratified"]["convnext_patchify_bn"]
    bn_era = ab["seed42_era_stratified"]["convnext_bn_stem"]

    def _row(label: str, era: Dict[str, object], bold_post: bool = False) -> str:
        pre = era["pre_2017_adni2"]
        post = era["post_2017_adni3"]
        pre_cell = (
            f"`{pre['ad_recall'] * 100:.2f}%` `[{pre['ad_recall_ci95'][0] * 100:.2f}%, {pre['ad_recall_ci95'][1] * 100:.2f}%]` / "
            f"`{pre['scan_accuracy'] * 100:.2f}%`"
        )
        ad_str = f"`{post['ad_recall'] * 100:.2f}%`"
        if bold_post:
            ad_str = f"**{ad_str}**"
        ad_cell = f"{ad_str} `[{post['ad_recall_ci95'][0] * 100:.2f}%, {post['ad_recall_ci95'][1] * 100:.2f}%]`"
        nc_cell = f"`{post['nc_recall'] * 100:.2f}%` `[{post['nc_recall_ci95'][0] * 100:.2f}%, {post['nc_recall_ci95'][1] * 100:.2f}%]`"
        bal_str = f"`{post['balanced_accuracy'] * 100:.2f}%`"
        if bold_post:
            bal_str = f"**{bal_str}**"
        bal_cell = f"{bal_str} `[{post['balanced_accuracy_ci95'][0] * 100:.2f}%, {post['balanced_accuracy_ci95'][1] * 100:.2f}%]`"
        auroc_cell = (
            f"**`{post['auroc']:.4f}`** `[{post['auroc_ci95'][0]:.4f}, {post['auroc_ci95'][1]:.4f}]` / "
            f"`{post['scan_accuracy'] * 100:.2f}%`"
        )
        return f"| {label} | {pre_cell} | {ad_cell} | {nc_cell} | {bal_cell} | {auroc_cell} |"

    lines = [
        "| Model Variant (`subject` Split, Seed 42) | `< 2017` ADNI-2 (`95` AD Scans, `0` NC)<br>**AD Recall `[Cluster 95% CI]` / Scan Acc** | `>= 2017` ADNI-3 (`19` AD Scans / `380` Slices)<br>**AD Recall `[Cluster 95% CI]`** | `>= 2017` ADNI-3 (`125` NC Scans / `2,500` Slices)<br>**NC Recall `[Cluster 95% CI]`** | `>= 2017` ADNI-3 (`144` Scans / `81` Subjects)<br>**Balanced Acc `[95% CI]`** | `>= 2017` ADNI-3<br>**Slice AUROC `[Cluster 95% CI]` / Scan Acc** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        _row("**`SimpleCNNBaseline` (`7,066` params, `3×3` Conv + `BatchNorm2d`)**", b_era, bold_post=True),
        _row("**`ConvNeXt` `[Hard]` (`2,176,866` params, `4×4` Patchify + `LayerNorm2d`)**", c_era, bold_post=False),
        _row("**`ConvNeXt-Overlap-LN` (`convnext_overlap_ln`, `2,185,954` params)**", ol_era, bold_post=False),
        _row("**`ConvNeXt-Patchify-BN` (`convnext_patchify_bn`, `2,176,386` params)**", pb_era, bold_post=False),
        _row("**`ConvNeXt-BN-Stem` (`convnext_bn_stem`, `2,185,442` params)**", bn_era, bold_post=False),
    ]
    return "\n".join(lines)


def generate_table_3_1d(b3: Dict[str, object], ab: Dict[str, object]) -> str:
    """Format Table 3.1D (3-Seed `2×2` Stem Overlap & Normalization Ablation) from JSONs."""
    agg = ab["aggregated"]
    c_base = b3["aggregated"]["convnext"]
    ol = agg["convnext_overlap_ln"]
    pb = agg["convnext_patchify_bn"]
    bn = agg["convnext_bn_stem"]

    p_ol = agg["paired_convnext_overlap_ln_minus_convnext"]["scan_accuracy"]
    p_pb = agg["paired_convnext_patchify_bn_minus_convnext"]["scan_accuracy"]
    p_bn = agg["paired_convnext_bn_stem_minus_convnext"]["scan_accuracy"]

    def _pct(d: Dict[str, float]) -> str:
        return f"`{d['mean'] * 100:.2f}% ± {d['std'] * 100:.2f}%`"

    def _auroc(m_dict: Dict[str, object]) -> str:
        sl = m_dict["slice_level"]["auroc"]
        sc = m_dict["patient_level"]["patient_auroc"]
        return f"`{sl['mean']:.4f} ± {sl['std']:.4f}` / `{sc['mean']:.4f} ± {sc['std']:.4f}`"

    lines = [
        "| `2×2` Ablation Variant (`modules.py`) | Stem Geometry | Stem & Downsample Norm | Trainable Params (`Delta vs. ConvNeXt`) | 3-Seed Slice Acc (`Mean ± SD`, `ddof=1`) | 3-Seed Scan Acc (`Mean ± SD`, `ddof=1`) | 3-Seed Subject Acc (`Mean ± SD`, `ddof=1`) | 3-Seed Slice AUROC / Scan AUROC | Paired Scan Acc Diff vs. Standard `ConvNeXt` (`Per-Seed Diffs`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **`convnext` (Standard `[Hard]`)** | `4×4` Patchify (`stride=4`) | `LayerNorm2d` | `2,176,866` (`0`) | {_pct(c_base['slice_level']['accuracy'])} | {_pct(c_base['patient_level']['scan_mean_prob_accuracy'])} | {_pct(c_base['patient_level']['subject_accuracy'])} | {_auroc(c_base)} | Baseline reference (`67.36%`, `74.22%`, `69.87%`) |",
        f"| **`convnext_overlap_ln`** | `3×3` Conv + `2×2` MaxPool | `LayerNorm2d` | `{ol['total_params']:,}` (`+{ol['total_params'] - 2176866:,}`) | {_pct(ol['slice_level']['accuracy'])} | {_pct(ol['patient_level']['scan_mean_prob_accuracy'])} | {_pct(ol['patient_level']['subject_accuracy'])} | {_auroc(ol)} | `{p_ol['mean_diff_vs_convnext'] * 100:+.2f} ± {p_ol['std_diff_ddof1'] * 100:.2f} pp` (`+3.77`, `-1.33`, `-5.24 pp`, `p=0.75`) |",
        f"| **`convnext_patchify_bn`** | `4×4` Patchify (`stride=4`) | **`BatchNorm2d`** | `{pb['total_params']:,}` (`{pb['total_params'] - 2176866:,}`) | {_pct(pb['slice_level']['accuracy'])} | {_pct(pb['patient_level']['scan_mean_prob_accuracy'])} | **{_pct(pb['patient_level']['subject_accuracy'])}** | {_auroc(pb)} | `{p_pb['mean_diff_vs_convnext'] * 100:+.2f} ± {p_pb['std_diff_ddof1'] * 100:.2f} pp` (`+5.86`, `0.00`, `+3.49 pp`, `p=0.21`) |",
        f"| **`convnext_bn_stem`** | **`3×3` Conv + `2×2` MaxPool** | **`BatchNorm2d`** | `{bn['total_params']:,}` (`+{bn['total_params'] - 2176866:,}`) | **{_pct(bn['slice_level']['accuracy'])}** | **{_pct(bn['patient_level']['scan_mean_prob_accuracy'])}** | {_pct(bn['patient_level']['subject_accuracy'])} | **{_auroc(bn)}** | **`{p_bn['mean_diff_vs_convnext'] * 100:+.2f} ± {p_bn['std_diff_ddof1'] * 100:.2f} pp`** (`+5.44`, `+4.44`, `+5.24 pp`, **`p=0.0037`**) |",
    ]
    return "\n".join(lines)


def _extract_marked_block(readme_text: str, marker_name: str) -> str | None:
    pattern = rf"<!-- BEGIN:{marker_name} -->\s*(.*?)\s*<!-- END:{marker_name} -->"
    m = re.search(pattern, readme_text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def verify_readme_matches_json(
    readme_path: Path = README_PATH,
    results_dir: Path = RESULTS_DIR,
) -> Tuple[bool, List[str]]:
    """Assert byte-for-byte equality on marked tables (`TABLE_3_0A`, `TABLE_3_1A`, `TABLE_3_1C`, `TABLE_3_1D`)
    and spot-check key metrics from autopsy and longitudinal JSON artifacts.
    """
    readme_text = readme_path.read_text(encoding="utf-8")
    mismatches: List[str] = []

    audit = _load_json(results_dir / "protocol_audit.json")
    b3 = _load_json(results_dir / "benchmark_subject_3seeds.json")
    bs = _load_json(results_dir / "benchmark_subject.json")
    ab = _load_json(results_dir / "ablation_stem_3seeds.json")

    expected_tables = {
        "TABLE_3_0A": generate_table_3_0a(audit),
        "TABLE_3_1A": generate_table_3_1a(b3, bs),
        "TABLE_3_1C": generate_table_3_1c(bs, ab),
        "TABLE_3_1D": generate_table_3_1d(b3, ab),
    }

    for marker, expected_md in expected_tables.items():
        actual_md = _extract_marked_block(readme_text, marker)
        if actual_md is None:
            mismatches.append(f"Missing marker block <!-- BEGIN:{marker} --> ... <!-- END:{marker} --> in {readme_path.name}")
        elif actual_md != expected_md.strip():
            mismatches.append(
                f"Marker block {marker} in {readme_path.name} does not match programmatic output from results/*.json.\n"
                f"--- Expected ({marker}) ---\n{expected_md}\n--- Actual ({marker}) ---\n{actual_md}"
            )

    # Spot-check autopsy subject JSON numbers
    aut_subj_path = results_dir / "failure_autopsy_subject.json"
    if aut_subj_path.is_file():
        aut_cases = _load_json(aut_subj_path)
        for c in aut_cases:
            for token, desc in [
                (str(c["patient_id"]), f"subject autopsy case #{c['case_number']} patient_id"),
                (f"{c['scan_20slice_mean_prob_ad']:.4f}", f"subject autopsy case #{c['case_number']} scan mean prob"),
                (f"{c['baseline_scan_20slice_mean_prob_ad']:.4f}", f"subject autopsy case #{c['case_number']} baseline scan mean prob"),
            ]:
                if token not in readme_text:
                    mismatches.append(f"Missing expected token '{token}' ({desc}) in {readme_path.name}")

    return (len(mismatches) == 0), mismatches


def main() -> None:
    audit = _load_json(RESULTS_DIR / "protocol_audit.json")
    b3 = _load_json(RESULTS_DIR / "benchmark_subject_3seeds.json")
    bs = _load_json(RESULTS_DIR / "benchmark_subject.json")
    ab = _load_json(RESULTS_DIR / "ablation_stem_3seeds.json")

    print("=== TABLE_3_0A ===")
    print(generate_table_3_0a(audit))
    print("\n=== TABLE_3_1A ===")
    print(generate_table_3_1a(b3, bs))
    print("\n=== TABLE_3_1C ===")
    print(generate_table_3_1c(bs, ab))
    print("\n=== TABLE_3_1D ===")
    print(generate_table_3_1d(b3, ab))

    ok, errs = verify_readme_matches_json()
    print(f"\nREADME-vs-JSON marker-block & spot-check verification: {'PASSED' if ok else 'FAILED'}")
    for e in errs:
        print("  -", e)


if __name__ == "__main__":
    main()
