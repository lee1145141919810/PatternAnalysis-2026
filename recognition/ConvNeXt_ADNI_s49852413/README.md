# ConvNeXt for Alzheimer's Disease Neuroimaging Classification, Uncertainty Calibration, and Selective Clinical Triage on ADNI Brain MRI

**Course:** COMP3710 / COMP7710 — Pattern Recognition & Analysis (Semester 2, 2026)  
**Specification Version:** Assessment Specification **Version 2.5**  
**Author:** Lingxiao Li  
**Student ID:** `49852413` (`s49852413`)  
**Email:** `s4985241@student.uq.edu.au`  
**Selected Project:** Section 2.1 Classification — Neuroimaging (`ADNI` Brain MRI)  
**Selected Architecture & Tier:** **ConvNeXt** (`[Hard Difficulty]`, eligible for full 20/20 implementation ceiling)  
**Target Repository:** `shakes76/PatternAnalysis-2026`  
**Target Branch:** `topic-recognition`  
**Working Directory:** `recognition/ConvNeXt_ADNI_s49852413/`

---

## 1. Executive Summary & Problem Formulation

### 1.1 Clinical Problem & Objective
Early and accurate differentiation between **Alzheimer's Disease (`AD`)** and **Cognitively Normal Controls (`NC`)** from structural T1-weighted brain Magnetic Resonance Imaging (MRI) is critical for clinical trial cohort enrichment, differential dementia workup, and longitudinal disease tracking. On sagittal T1-weighted brain MRI (`ADNI_T1_3T_MASKED_N4`), Alzheimer's neurodegeneration manifests across two complementary spatial scales:
1. **Focal Medial Temporal & Callosal Atrophy:** Progressive volume loss and cortical ribbon thinning across the hippocampus, parahippocampal gyrus, precuneus, and corpus callosum.
2. **Macroscopic Ventriculomegaly & Sulcal Widening:** Compensatory *ex-vacuo* dilation of the lateral and third ventricles (visible as expanded dark cerebrospinal fluid [CSF] cavities in the central brain ROI) alongside diffuse cerebral sulcal enlargement.

This project designs, implements from scratch in pure PyTorch, and critically evaluates an end-to-end **ConvNeXt** (`[Hard Difficulty]`, Liu et al., *CVPR 2022*) diagnostic prototype on the official ADNI cohort (`/home/groups/comp3710/ADNI/AD_NC`). We benchmark `ConvNeXt` head-to-head against a matched-training `SimpleCNNBaseline` equipped with `BatchNorm2d`, uncover and prove a major **acquisition-era protocol confound and 216-subject cross-folder overlap** in the raw dataset folders via [`audit_protocol.py`](./audit_protocol.py), evaluate both **Strict Subject-Disjoint** (3-seed `mean ± std`) and **Within-Subject Longitudinal Scan-Disjoint** regimes, resolve the **Confidence and Ambiguity Dilemma in Automated Triage** via L-BFGS Temperature Scaling and validation-selected reject-option routing, and conduct a quantitative 4-case **Failure Case Autopsy** comparing intercepted borderline errors (`< τ*`) against high-confidence escaped errors (`≥ τ*`).

### 1.2 Algorithm Working Principles & Pure-PyTorch Architecture
Standard small-kernel CNNs ($3\times 3$ convolutions with `BatchNorm2d`, `ReLU`, and $2\times 2$ `MaxPool2d`) possess strong local translation invariance and batch-level regularization, which are advantageous in low-subject-count medical cohorts, but have limited receptive fields per layer. Conversely, Vision Transformers (ViTs) capture global context via self-attention but lack convolutional inductive biases and require large pre-training corpora.

**ConvNeXt** (Liu et al., *CVPR 2022*; adapted from the official [`facebookresearch/ConvNeXt`](https://github.com/facebookresearch/ConvNeXt) design into pure `torch`/`torch.nn` in [`modules.py`](./modules.py)) modernizes a 4-stage hierarchical ConvNet with Vision Transformer design principles:

$$\mathbf{z}_{\ell} = \mathbf{x}_{\ell-1} + \text{DropPath}\!\left(\boldsymbol{\gamma}_{\ell} \odot W_{\text{pw2}}\,\phi_{\text{GELU}}\!\Big(W_{\text{pw1}}\,\text{LN}\big(\text{DWConv}_{7\times 7}(\mathbf{x}_{\ell-1})\big)\Big)\right)$$

where:
- $\text{DWConv}_{7\times 7}$: Depthwise separable $7\times 7$ convolution (`groups = C`, `padding = 3`) aggregating a 49-pixel spatial neighborhood per block;
- $\text{LN}$: Token-wise Layer Normalization ([`LayerNorm2d`](./modules.py) in NCHW stems/downsamples and `nn.LayerNorm` in NHWC blocks) normalizing across the $C$ channels at each spatial token $(h, w)$;
- $W_{\text{pw1}} \in \mathbb{R}^{4C \times C}, W_{\text{pw2}} \in \mathbb{R}^{C \times 4C}$: Pointwise $1\times 1$ linear projections forming a $4\times$ inverted bottleneck;
- $\phi_{\text{GELU}}$: Single Gaussian Error Linear Unit activation per block;
- $\boldsymbol{\gamma}_{\ell} \in \mathbb{R}^{C}$: Learnable per-channel LayerScale vector ($\boldsymbol{\gamma}_0 = 0.1$) stabilizing deep residual optimization;
- $\text{DropPath}$: Linearly scheduled Stochastic Depth regularization (`drop_path_rate = 0.10` across all 12 blocks).

For our $96\times 96$ single-channel sagittal MRI input, the $4\times 4$ non-overlapping patchify stem (`stride = 4`) and three $2\times 2$ (`stride = 2`) inter-stage downsampling layers produce the exact spatial feature map cascade:
- **`stages[0]` (`2 blocks`, `C = 32`):** $24\times 24$ spatial resolution
- **`stages[1]` (`2 blocks`, `C = 64`):** $12\times 12$ spatial resolution
- **`stages[2]` (`6 blocks`, `C = 128`):** $6\times 6$ spatial resolution *(core semantic representation stage; hooked by [`compute_spatial_attribution()`](./modules.py) for $6\times 6 \to 96\times 96$ Grad-CAM attribution)*
- **`stages[3]` (`2 blocks`, `C = 256`):** $3\times 3$ spatial resolution $\to$ Global Average Pooling (`[B, 256]`) $\to$ `LayerNorm` $\to$ `Linear(256, 2)`

| Architectural Component | `SimpleCNNBaseline` ([`modules.py`](./modules.py)) | `ConvNeXt` `[Hard]` ([`modules.py`](./modules.py)) | Neuroimaging & Inductive-Bias Trade-Off |
| :--- | :--- | :--- | :--- |
| **Stages & Channels** | 3 stages, `C = (8, 16, 32)` ($7,066$ params) | 4 stages, `depths = (2, 2, 6, 2)`, `dims = (32, 64, 128, 256)` ($2,176,866$ params) | Concentrates 6 blocks in Stage 3 (`6×6`, `C=128`) to integrate ventricular-to-cortical ratios |
| **Stem & Downsampling** | Overlapping $3\times 3$ conv (`stride=1`) + $2\times 2$ `MaxPool2d` (`96→48→24→12`) | Non-overlapping $4\times 4$ Patchify Stem (`stride=4`) + $2\times 2$ conv downsamples (`96→24→12→6→3`) | Patchify stem emulates ViT tokenization; overlapping $3\times 3$ + MaxPool retains stronger local translation invariance |
| **Spatial & Channel Mixing** | Dense $3\times 3$ conv (`groups=1`) | $7\times 7$ Depthwise Conv (`groups=C`) + $4\times$ Inverted Bottleneck ($C \to 4C \to C$) | Decouples 49-pixel spatial context aggregation from high-dimensional channel expansion |
| **Normalization & Activation** | `BatchNorm2d` + `ReLU` after every conv | Single pixel-wise `LayerNorm` + single `GELU` per block | `BatchNorm2d` normalizes across `(B, H, W)` (preserving spatial CSF contrast); `LayerNorm` normalizes per spatial token across `C` |
| **Classification Head** | `AdaptiveAvgPool2d((1, 1))` $\to$ `Linear(32, 32)` $\to$ `Linear(32, 2)` | Global Average Pooling (`mean(-2, -1)`) $\to$ `LayerNorm(256)` $\to$ `Linear(256, 2)` | Matched global average pooling (`1×1`) head ensures a fair architectural comparison |

![ConvNeXt Architecture & Block Topology](./images/convnext_architecture.png)

---

## 2. Required Feasibility Check-Off (Compulsory Midpoint Hurdle)

> **Specification Compliance (Section 3 of Assessment Specification v2.5):** This self-contained feasibility review covers the four mandatory check-off pillars: (1) User Need, Scope & Testable Acceptance Criteria, (2) Model Choice & Course Concepts, (3) Preliminary Feasibility Evidence & Forensic Protocol Audit, and (4) Technical Risks, Computing Budget, Workable Fallbacks & Next Planned Experiments.

### 2.1 User Need, Scope & Testable Acceptance Criteria
- **Intended User:** Memory-clinic neuroradiologists and clinical trial screening coordinators triaging structural T1-weighted brain MRI scans for Alzheimer's Disease (`AD`) versus Cognitively Normal (`NC`) aging.
- **Prototype Scope:** An uncertainty-calibrated 2D slice and 20-slice 3D scan classification pipeline that auto-clears high-confidence scans ($\max_c p_c \ge \tau^*$, where $\tau^*$ is selected strictly on the validation set) while referring borderline/ambiguous scans ($\max_c p_c < \tau^*$) to human neuroradiologists.
- **Testable Acceptance Criteria & Empirical Verification Summary:**

| Criterion ID | Target Acceptance Specification |Strict Subject-Disjoint (`subject`, 3-Seed Mean ± Std) | Within-Subject Longitudinal (`longitudinal`, Seed 42) | Status & Honest Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **AC-1 (Discrimination)** | Slice Acc $\ge 80.0\%$, Scan Acc $\ge 85.0\%$, AUROC $\ge 0.85$ | `ConvNeXt`: $67.45\% \pm 1.98\%$ slice / $70.49\% \pm 2.83\%$ scan ($71.61\%$ vote)<br>`SimpleCNN`: $75.22\% \pm 3.97\%$ slice ($80.69\%$ S43) / $78.31\% \pm 4.53\%$ scan ($84.44\%$ S43) | `ConvNeXt`: **$87.18\%$ slice** / **$93.55\%$ scan** ($\text{AUROC}=0.9307$ slice, **$0.9906$ scan**)<br>`SimpleCNN`: $83.23\%$ slice / $90.32\%$ scan | **Met on Longitudinal (`87.18%` / `93.55%`)**; **Below `80%` on Unseen Subjects (`67.45%` / `70.49%`)** due to 476-subject cross-patient inductive-bias gap (disclosed in Sec 3.1) |
| **AC-2 (Calibration)** | Post-hoc Temperature Scaling achieves test $\text{ECE} \le 0.05$ on large validation cohorts | `ConvNeXt` ($T^* = 3.27 \pm 0.42$): **$\text{ECE}: 0.1785 \to 0.0217 \pm 0.0085$** ($-87.8\%$ reduction) | `SimpleCNN` ($T^* = 0.76$): $\text{ECE}: 0.1118 \to 0.0684$<br>`ConvNeXt` ($T^* = 1.73$): $\text{ECE}: 0.0362 \to 0.0698$ | **Met on Subject-Disjoint ($\text{ECE} = 0.0217 \le 0.05$)**; small 60-scan longitudinal val set shows mild calibration overfitting |
| **AC-3 (Selective Triage)** | Validation-selected gate $\tau^*$ lifts scan accuracy $\ge 85\%$ and intercepts $\ge 70\%$ of FN | `ConvNeXt` Scan Triage ($\tau^* = 0.70$, S42): **$85.71\%$ auto-cleared scan acc**, **$97.22\%$ FN intercepted** | `ConvNeXt` Scan Triage ($\tau^* = 0.80$): **$100.00\%$ auto-cleared scan acc**, **$100.00\%$ FN intercepted** | **Met across both protocols** ($\tau^*$ chosen strictly on validation split without test peeking) |
| **AC-4 (Compute & Latency)** | Peak tensor memory $< 2.0\text{ GB}$; batched latency (`batch=128`) $< 1.0\text{ ms/slice}$ | `ConvNeXt`: **$41.90\text{ MB}$** active tensor memory, **$0.435\text{ ms/slice}$** (`2,299 slices/s`) | `ConvNeXt`: **$41.90\text{ MB}$** active tensor memory, **$0.341\text{ ms/slice}$** (`2,933 slices/s`) | **Met with $>40\times$ memory margin** ($6.8\text{ ms}$ per 20-slice 3D patient scan) |
| **AC-5 (Code & Data Hygiene)** | Zero NumPy/3rd-party imports in `modules.py`; programmatic scan & subject leakage audit | AST test + `verify_no_patient_leakage()` verify `scan_overlap = 0` & `subject_overlap = 0` | AST test + `verify_no_patient_leakage()` verify `scan_overlap = 0` (`subject_disjoint = False` flagged) | **Met (`7/7` unit tests passing in [`test_pipeline.py`](./test_pipeline.py))** |

### 2.2 Model Choice, Course Concepts & Difficulty Ceiling
- **Difficulty Ceiling:** **ConvNeXt** (`[Hard Difficulty]`, 20/20 implementation ceiling), implemented from scratch in [`modules.py`](./modules.py) alongside `SimpleCNNBaseline`, `TemperatureScaledClassifier`, and `compute_spatial_attribution`.
- **Course Concepts Applied:** (1) Depthwise-separable receptive-field expansion ($7\times 7$ `groups=C`) vs. dense $3\times 3$ convolutions; (2) Token-wise `LayerNorm2d` vs. batch-wide `BatchNorm2d` and their effect on spatial intensity contrast in single-channel medical imaging; (3) Post-hoc probability calibration via L-BFGS NLL minimization on held-out validation logits (Guo et al., *ICML 2017*); and (4) Reject-option selective classification (El-Yaniv & Wiener, *JMLR 2010*).

### 2.3 Preliminary Feasibility Evidence & Reproducible Protocol Audit
The dataset (`/home/groups/comp3710/ADNI/AD_NC`) contains **30,520 skull-stripped, N4 bias-corrected sagittal T1-weighted slices** ($240\times 256$ pixels, 20 contiguous sagittal slices per 3D scan with slice indices ranging from `67` to `114`, spanning **1,526 3D MRI scans** across **680 unique human subjects** in `meta_data_with_label.json`).

#### ⚠️ Reproducible Forensic Discovery ([`audit_protocol.py`](./audit_protocol.py) $\to$ [`results/protocol_audit.json`](./results/protocol_audit.json))
Running `python recognition/ConvNeXt_ADNI_s49852413/audit_protocol.py` cross-references every JPEG filename (`<scan_id>_<slice_idx>.jpeg`) against `meta_data_with_label.json` (`100%` of `1,526` scans matched; `0` unmapped scans) and proves two structural confounds in the default `train/` vs. `test/` folders:

| Official Folder Partition | Total Slices | 3D Scans (`Ixxxxxx`) | Unique Subjects (`XXX_S_XXXX`) | Scan ID Range (`min – max`) | Acquisition Year Range (`p05 – p95`) | Accelerated Sagittal MPRAGE Scans | Dominant ADNI Acquisition Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`official_train / AD`** | $10,400$ | $520$ | $165$ | `I218391 – I388204` | **`2011 – 2013`** | $0 / 520$ ($0.0\%$) | Legacy **ADNI-2** Standard MPRAGE / IR-SPGR |
| **`official_train / NC`** | $11,120$ | $556$ | $400$ | `I808819 – I1442100` | **`2017 – 2020`** | $377 / 556$ ($67.8\%$) | Modern **ADNI-3** Accelerated Sagittal MPRAGE |
| **`official_test / AD`** | $4,460$ | $223$ | $131$ | `I388208 – I1452495` | **`2013 – 2020`** | $110 / 223$ ($49.3\%$) | Mixed ADNI-2 / ADNI-3 (`64` subjects overlap `train/AD`) |
| **`official_test / NC`** | $4,540$ | $227$ | $200$ | `I423180 – I1443741` | **`2017 – 2020`** | $152 / 227$ ($67.0\%$) | Modern ADNI-3 (`152` subjects overlap `train/NC`) |

1. **Cross-Era Acquisition Protocol Confound in `official_train/`:** Every scan in `train/AD` was acquired in `2011–2013` (`I218391–I388204`, `0%` Accelerated MPRAGE), whereas every scan in `train/NC` was acquired in `2017–2021` (`I808819–I1442100`, `67.8%` Accelerated MPRAGE). Training on the default folders induces shortcut learning on scanner/era contrast rather than Alzheimer's pathology.
2. **Cross-Folder Human Subject Leakage (`216` Overlapping Subjects):** Exactly **216 unique human subjects** (`64` AD + `152` NC) have longitudinal visits split across both `official_train/` and `official_test/`.

#### Three Evaluation Protocols Engineered in [`dataset.py`](./dataset.py)
To evaluate models transparently under both strict cross-patient hygiene and longitudinal follow-up, [`build_adni_dataloaders()`](./dataset.py) implements three auditable protocols verified by [`verify_no_patient_leakage()`](./dataset.py):

| Split Mode (`--split_mode`) | Slices (`train / val / test`) | 3D Scans (`train / val / test`) | Unique Subjects (`train / val / test`) | Scan Overlap | Subject Overlap | Evaluation Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`subject` (Strict Subject-Disjoint)** | $10,420 / 4,900 / 4,780$ | $1,042 / 245 / 239$ | **$476 / 102 / 102$** | **$0$** (`scan_disjoint=True`) | **$0$** (`subject_disjoint=True`) | **Specification Section 2.1 primary hygiene benchmark:** unseen human subjects (`155 AD, 321 NC` train) |
| **`longitudinal` (Within-Subject Scan-Disjoint)** | $14,040 / 1,200 / 1,240$ | $1,404 / 60 / 62$ | $680 / 60 / 62$ | **$0$** (`scan_disjoint=True`) | $60$ val / $62$ test (`subject_disjoint=False`) | Longitudinal follow-up monitoring across separate clinical visits (`Ixxxxxx`) of known cohort members |
| **`official` (Raw Folder Split)** | $18,300 / 3,220 / 9,000$ | $915 / 161 / 450$ | $518 / 139 / 331$ | **$0$** (`scan_disjoint=True`) | $92$ val / $197$ test (`subject_disjoint=False`) | Empirical control proving accuracy collapse ($64.60\%$) under the ADNI-2 vs. ADNI-3 era confound |

#### Preprocessing & Anatomically Valid Sagittal Augmentation
- **Tight Brain ROI Crop & Resize:** Each $240\times 256$ sagittal slice from `ADNI_T1_3T_MASKED_N4` is cropped to $[16:208, 48:224]$ to remove empty background borders while preserving the full cerebral cortex, ventricles, and cerebellum, then bilinearly resized to $96\times 96$.
- **Cohort ROI Standardization:** Normalized via $\tilde{I} = (I - 0.18) / 0.25$ to preserve relative T1 intensity contrast between dark ventricular CSF ($\approx 0.0\text{--}0.28$) and bright cerebral parenchyma.
- **Why Horizontal Flip is Disabled:** Because these images are **sagittal** brain slices (displaying the anterior frontal lobe on one side and the posterior occipital lobe/cerebellum on the other), a horizontal flip would invert **anterior–posterior brain anatomy**, creating biologically impossible training images. We restrict training augmentations to $\pm 2\text{ px}$ spatial translation and mild gamma jitter ($\gamma \in [0.92, 1.08]$).

![ADNI Pre-processing Pipeline and Class Intensity Distributions](./images/preprocessing_and_samples.png)

### 2.4 Technical Risks, Computing Budget, Fallbacks & Next Planned Experiments
- **Risk 1 — High-Capacity Overfitting on Small Subject Counts ($N_{\text{train}}=476$ subjects):** `ConvNeXt` ($2.18\text{ M}$ params) overfits individual subject morphology when trained from scratch on 476 subjects. *Mitigation:* Benchmark across 3 random seeds (`42, 43, 44`), apply L-BFGS Temperature Scaling ($T^* = 3.27 \pm 0.42$, reducing ECE by $87.8\%$), and aggregate 20 slices per scan with validation-selected selective triage.
- **Risk 2 — Calibration Overfitting on Small Validation Splits:** On `longitudinal` ($N_{\text{val}} = 60$ scans), fitting $T^*$ slightly increases test ECE ($0.0362 \to 0.0698$). *Mitigation:* [`train.py`](./train.py) automatically checks whether Temperature Scaling improves validation ECE before activating calibrated probabilities for triage.
- **Computing Budget & Fallback:** Full 12-epoch training of both models takes $\approx 4.8\text{ minutes}$ per seed on Apple Silicon MPS (`41.90 MB` active tensor memory) or $<2\text{ minutes}$ on a single UQ Rangpur A100/V100 GPU ([`slurm_train_rangpur.sh`](./slurm_train_rangpur.sh)). `SimpleCNNBaseline` ($7,066$ params, $3.88\text{ s/epoch}$) provides an ultra-fast fallback.
- **Next Planned Experiments (Phase 2 Roadmap):**
  1. *Overlapping Convolutional Stem for ConvNeXt:* Replace the ViT-style non-overlapping $4\times 4$ `stride=4` patchify stem with a two-layer overlapping $3\times 3$ (`stride=2, stride=1`) stem with `BatchNorm2d` (Xiao et al., *NeurIPS 2021*) to close the cross-subject inductive-bias gap against `SimpleCNNBaseline`.
  2. *2.5D Multi-Slice Cross-Attention Fusion:* Jointly encode all 20 sagittal slices per scan `[B, 20, 96, 96]` via slice-positional attention rather than independent 2D slice training followed by post-hoc averaging.
  3. *Age & Intracranial Volume (ICV) Conditioning:* Concatenate patient age and ICV from ADNI clinical tables into the final linear classifier (`nn.Linear(256 + 2, 2)`) to disambiguate benign senile ventriculomegaly from Alzheimer's atrophy.

---

## 3. Quantitative Benchmark & Resource Profiling

### 3.1 Head-to-Head Quantitative Benchmark Across All Three Protocols
Both `SimpleCNNBaseline` ($7,066$ parameters, `BatchNorm2d`) and `ConvNeXt` ($2,176,866$ parameters, `LayerNorm2d`, `DropPath=0.10`) were trained under **identical optimization hyperparameters**: `AdamW(lr=1.2e-3, weight_decay=1e-2)`, 12 epochs with cosine annealing (`T_max=12`), `batch_size=128`, and $96\times 96$ resolution. All raw JSON outputs are tracked in [`results/`](./results/).

#### Table 3.1A — Primary Hygiene Benchmark: Strict Subject-Disjoint (`--split_mode subject`, Zero Subject Overlap, 3 Seeds `42, 43, 44`)
> Evaluated on $4,780$ test slices ($239$ scans, $102$ unseen subjects) per seed. Full metrics in [`results/benchmark_subject_3seeds.json`](./results/benchmark_subject_3seeds.json) and [`results/benchmark_subject.json`](./results/benchmark_subject.json).

| Evaluation Metric | `SimpleCNNBaseline` (Seed 42 `[95% CI]`) | `SimpleCNNBaseline` (3-Seed `Mean ± Std`) | `ConvNeXt` `[Hard]` (Seed 42 `[95% CI]`) | `ConvNeXt` `[Hard]` (3-Seed `Mean ± Std`) |
| :--- | :---: | :---: | :---: | :---: |
| **Slice Test Accuracy** | $71.42\%$ ($3,414/4,780$) | **$75.22\% \pm 3.97\%$** *(S43: $80.69\%$)* | $65.94\%$ ($3,152/4,780$) | $67.45\% \pm 1.98\%$ *(S43: $70.24\%$)* |
| **Slice Balanced Accuracy** | $71.55\%$ | **$75.34\% \pm 3.90\%$** | $66.05\%$ | $67.59\% \pm 1.89\%$ |
| **Slice Macro-F1** | $0.7142$ | **$0.7521 \pm 0.0396$** | $0.6594$ | $0.6742 \pm 0.0200$ |
| **Slice Test AUROC / PR-AUC** | $0.7824$ / $0.7148$ | **$0.8254 \pm 0.0426$** / **$0.7851 \pm 0.0578$** | $0.7182$ / $0.6803$ | $0.7387 \pm 0.0237$ / $0.7117 \pm 0.0341$ |
| **Scan Mean-Prob Accuracy (20 slices)** | $73.64\%$ `[68.20%, 79.08%]` | **$78.31\% \pm 4.53\%$** *(S43: $84.44\%$)* | $67.36\%$ `[61.09%, 72.80%]` | $70.49\% \pm 2.83\%$ *(S43: $74.22\%$)* |
| **Scan Majority-Vote Accuracy** | $74.90\%$ ($179/239$ scans) | **$78.44\% \pm 3.98\%$** *(S43: $84.00\%$)* | $69.87\%$ ($167/239$ scans) | $71.61\% \pm 1.62\%$ *(S43: $73.78\%$)* |
| **Scan-Level AUROC** | $0.8235$ | **$0.8660 \pm 0.0418$** *(S43: $0.9229$)* | $0.7552$ | $0.7839 \pm 0.0320$ *(S43: $0.8286$)* |
| **Subject-Level Accuracy ($102$ subjects)** | $73.53\%$ `[64.71%, 82.35%]` | **$74.84\% \pm 2.57\%$** *(S43: $78.43\%$)* | $64.71\%$ `[55.88%, 73.53%]` | $69.28\% \pm 5.79\%$ *(S43: $77.45\%$)* |
| **Subject-Level AUROC** | $0.8292$ | **$0.8410 \pm 0.0416$** | $0.7172$ | $0.7602 \pm 0.0441$ *(S43: $0.8208$)* |
| **Uncalibrated $\to$ Calibrated ECE** | $0.0163 \to 0.0138$ ($T^*=0.91$) | $0.0469 \to 0.0265 \pm 0.0124$ | **$0.1956 \to 0.0120$** ($T^*=3.57$) | **$0.1785 \to 0.0217 \pm 0.0085$ ($-87.8\%$)** |
| **Scan Triage Acc @ Val $\tau^*=0.70$ (Seed 42)** | $88.89\%$ ($45.2\%$ cov, $85.2\%$ FN caught) | — | **$85.71\%$ ($23.4\%$ cov, $97.2\%$ FN caught)** | — |

#### Table 3.1B — Within-Subject Longitudinal (`longitudinal`) & Official Confounded (`official`) Benchmarks (Seed 42)
> Full metrics in [`results/benchmark_longitudinal.json`](./results/benchmark_longitudinal.json) and [`results/benchmark_official.json`](./results/benchmark_official.json).

| Evaluation Metric | Longitudinal: `SimpleCNNBaseline` | Longitudinal: `ConvNeXt` `[Hard]` | Official Confounded: `SimpleCNNBaseline` | Official Confounded: `ConvNeXt` `[Hard]` |
| :--- | :---: | :---: | :---: | :---: |
| **Slice Test Accuracy** | $83.23\%$ ($1,032/1,240$) | **$87.18\%$ ($1,081/1,240$)** | $69.42\%$ ($6,248/9,000$) | $64.60\%$ ($5,814/9,000$) |
| **Slice Balanced Accuracy** | $83.23\%$ | **$87.18\%$** | $69.32\%$ | $64.49\%$ |
| **Slice Macro-F1** | $0.8312$ | **$0.8717$** | $0.6894$ | $0.6402$ |
| **Slice Test AUROC / PR-AUC** | $0.9132$ / $0.9005$ | **$0.9307$ / $0.9264$** | $0.7710$ / $0.7647$ | $0.6959$ / $0.6798$ |
| **Scan Mean-Prob Accuracy `[95% CI]`** | $90.32\%$ `[82.26%, 96.77%]` | **$93.55\%$ `[87.10%, 98.39%]` ($58/62$)** | $70.44\%$ `[65.56%, 74.89%]` | $65.33\%$ `[60.67%, 69.78%]` |
| **Scan Majority-Vote Accuracy** | $90.32\%$ ($56/62$ scans) | **$93.55\%$ ($58/62$ scans)** | $72.44\%$ ($326/450$ scans) | $66.00\%$ ($297/450$ scans) |
| **Scan-Level AUROC** | $0.9584$ | **$0.9906$** | $0.8059$ | $0.7253$ |
| **NC Precision / Recall / F1** | $0.8931 / 0.7548 / 0.8182$ | **$0.8585 / 0.8903 / 0.8741$** | $0.6600 / 0.8121 / 0.7282$ | $0.6208 / 0.7663 / 0.6859$ |
| **AD Precision / Recall / F1** | $0.7877 / 0.9097 / 0.8443$ | **$0.8861 / 0.8532 / 0.8694$** | $0.7501 / 0.5742 / 0.6505$ | $0.6876 / 0.5235 / 0.5945$ |
| **Confusion Matrix (`TN, FP, FN, TP`)** | `468, 152, 56, 564` | **`552, 68, 91, 529`** | `3687, 853, 1899, 2561` | `3479, 1061, 2125, 2335` |
| **Scan Triage Acc @ Val-Selected $\tau^*$** | $96.88\%$ ($\tau^*=0.75$, $51.6\%$ cov) | **$100.00\%$ ($\tau^*=0.80$, $43.5\%$ cov)** | $81.70\%$ ($\tau^*=0.75$, $52.2\%$ cov) | $75.68\%$ ($\tau^*=0.85$, $32.9\%$ cov) |

#### Critical Scientific Analysis of the Three Regimes
1. **Why `ConvNeXt` Excels on Within-Subject Longitudinal Follow-Up (`87.18%` Slice / `93.55%` Scan Acc, Scan AUROC `0.9906`):** When tracking longitudinal visits of known cohort subjects (`scan_disjoint = True`), `ConvNeXt` beats the `BatchNorm2d`-equipped `SimpleCNNBaseline` by **$+3.95\%$ slice accuracy** ($87.18\%$ vs. $83.23\%$) and **$+3.23\%$ scan accuracy** ($93.55\%$ vs. $90.32\%$), cutting false positives by $55.3\%$ ($152 \to 68$). Its high capacity ($2.18\text{ M}$ params) and $7\times 7$ depthwise kernels effectively capture fine subject-specific anatomical progression across visits.
2. **Why `SimpleCNNBaseline` (`BatchNorm2d`) Out-Generalizes `ConvNeXt` on Unseen Subjects (`78.31%` vs. `70.49%` Scan Acc) & Honest Disclosure:** Under strict human-subject isolation (`subject_overlap = 0`, only $155$ AD and $321$ NC training subjects), `ConvNeXt` achieves $67.45\% \pm 1.98\%$ slice accuracy and $70.49\% \pm 2.83\%$ scan accuracy ($71.61\% \pm 1.62\%$ majority vote), falling short of the $80\%$ threshold without triage, whereas adding `BatchNorm2d` to `SimpleCNNBaseline` lifts the baseline to $75.22\% \pm 3.97\%$ slice accuracy ($80.69\%$ on Seed 43) and $78.31\% \pm 4.53\%$ scan accuracy ($84.44\%$ on Seed 43). Two architectural mechanisms explain this finding:
   - *Inductive Bias of Overlapping $3\times 3$ Convolutions + MaxPool vs. $4\times 4$ Non-Overlapping Patchify Stem:* Training a $2.18\text{ M}$-parameter ViT-style `stride=4` patchify stem from scratch on only 476 subjects causes rapid subject memorization (`train_acc = 91.8%` vs. `val_acc = 66.4%`), whereas `SimpleCNNBaseline`'s $7,066$ parameters and $2\times 2$ `MaxPool2d` act as a strong structural regularizer. Notice also that the $\approx 20\%$ gap between `longitudinal` ($87.18\%$) and `subject` ($67.45\%$) for `ConvNeXt` is partly driven by within-subject morphological re-identification across visits.
   - *Spatial Contrast Preservation in `BatchNorm2d` vs. Pixel-Wise `LayerNorm`:* In 1-channel T1 MRI, `BatchNorm2d` normalizes across `(B, H, W)` per channel, preserving the spatial intensity difference between dark central CSF ventricles and bright cortical white matter. Pixel-wise `LayerNorm` normalizes across the $C$ channels independently at every $(h, w)$ coordinate, attenuating absolute regional brightness differences.
3. **Empirical Proof of the Official Split Shortcut (`64.60%` Slice Acc):** On `--split_mode official`, `ConvNeXt` reaches $97.05\%$ training accuracy on the confounded `train/` split (2011–2013 AD vs. 2017–2021 NC) but collapses to **$64.60\%$ test slice accuracy** ($52.35\%$ AD recall, uncalibrated $\text{ECE} = 0.2719$), confirming that the default folder layout induces severe acquisition-era shortcut learning.

#### Training & Discrimination Figures — Strict Subject-Disjoint (`./images/subject/`)
![Subject-Disjoint Training and Validation Dynamics](./images/subject/training_curves.png)

![Subject-Disjoint ROC, PR, and Confusion Matrix](./images/subject/roc_pr_comparison.png)

#### Training & Discrimination Figures — Within-Subject Longitudinal (`./images/`)
![Longitudinal Training and Validation Dynamics](./images/training_curves.png)

![Longitudinal ROC, PR, and Confusion Matrix](./images/roc_pr_comparison.png)

### 3.2 Computational Resource & Latency Profiling
Hardware telemetry was profiled on Apple Silicon MPS (`batch_size = 128`, $96\times 96$ resolution) with active tensor memory measured via `torch.mps.current_allocated_memory()` after cache reset ([`results/benchmark_longitudinal.json`](./results/benchmark_longitudinal.json) and [`results/benchmark_subject.json`](./results/benchmark_subject.json)):

| Profiling Dimension | `SimpleCNNBaseline` (`BatchNorm2d`) | `ConvNeXt` (`[Hard Difficulty]`) | Scaling Ratio & Clinical Deployment Feasibility |
| :--- | :---: | :---: | :--- |
| **Trainable Parameters** | $7,066$ ($0.007\text{ M}$) | **$2,176,866$ ($2.18\text{ M}$)** | $11.7\times$ smaller than ResNet-50 ($25.6\text{ M}$); fits easily on edge/clinic workstations |
| **Peak Active Tensor Memory** | $4.67\text{ MB}$ | **$41.90\text{ MB}$** | $>45\times$ below the $2.0\text{ GB}$ VRAM budget during `batch_size=128` training |
| **Mean Training Epoch Time** | $3.88\text{ s}$ (`subject`) / $4.06\text{ s}$ (`long.`) | **$18.32\text{ s}$ (`subject`) / $20.02\text{ s}$ (`long.`)** | Full 12-epoch training finishes in $3.7\text{--}4.0\text{ minutes}$ per run |
| **Single-Slice Latency (`batch=1`)** | $0.326\text{--}0.339\text{ ms}$ | **$1.462\text{--}1.853\text{ ms}$** | Real-time interactive single-slice inspection ($<2\text{ ms}$) |
| **Batched Latency (`batch=128`)** | $0.029\text{--}0.030\text{ ms / slice}$ | **$0.341\text{--}0.435\text{ ms / slice}$** | Full 20-slice 3D patient scan inferred in **$6.82\text{--}8.70\text{ ms}$** |
| **Batched Throughput (`batch=128`)** | $33,816\text{--}34,096\text{ slices/sec}$ | **$2,299\text{--}2,933\text{ slices/sec}$** | Processes $115\text{--}146$ complete 20-slice patient scans per second |

---

## 4. Open Research Dilemma: Uncertainty Calibration & Validation-Selected Clinical Triage

### 4.1 Post-Hoc L-BFGS Temperature Scaling & Validation Safeguard
High-capacity neural networks trained with cross-entropy loss become overconfident when evaluated on out-of-distribution or unseen patient cohorts. In [`TemperatureScaledClassifier`](./modules.py), we optimize a single scalar temperature $T^* > 0$ on the held-out validation logits $\mathbf{z}_i$ via L-BFGS (`max_iter=100`):

$$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \frac{\exp(z_{i, y_i} / T)}{\sum_{c \in \{0,1\}} \exp(z_{i, c} / T)}, \qquad \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \Big|\text{acc}(B_m) - \text{conf}(B_m)\Big|$$

- **Massive Calibration Gain on Unseen Subjects (`subject`, $N_{\text{val}} = 4,900$ slices / $245$ scans):** Across 3 random seeds, `ConvNeXt` exhibits severe raw overconfidence on unseen subjects ($\text{ECE}_{\text{raw}} = 0.1785 \pm 0.0121$; $0.1956$ on Seed 42). Temperature Scaling learns $T^* = 3.2721 \pm 0.4169$ ($3.5730$ on Seed 42), **slashing test ECE by $87.8\%$ down to $0.0217 \pm 0.0085$** ($0.0120$ on Seed 42) and reducing Brier score from $0.2434 \to 0.2083$.
- **Small-Validation Cohort Behavior (`longitudinal`, $N_{\text{val}} = 1,200$ slices / $60$ scans):** On the 60-scan longitudinal validation split, `SimpleCNNBaseline` improves from $\text{ECE}: 0.1118 \to 0.0684$ ($T^* = 0.7584$), whereas `ConvNeXt` is already well-calibrated raw ($\text{ECE}_{\text{raw}} = 0.0362$) and fitting $T^* = 1.7308$ on only 60 scans slightly increases test ECE to $0.0698$.

| Evaluation Protocol | Model | Optimal Temp $T^*$ | Uncalibrated Test ECE | Calibrated Test ECE | Uncal $\to$ Cal Brier | Calibration Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Strict Subject-Disjoint (3-Seed Mean ± Std)** | `SimpleCNNBaseline` | $0.863 \pm 0.144$ | $0.0469 \pm 0.0403$ | **$0.0265 \pm 0.0124$** | $0.1729 \to 0.1695$ | Improved ($-43.5\%$ ECE reduction) |
| **Strict Subject-Disjoint (3-Seed Mean ± Std)** | **`ConvNeXt` `[Hard]`** | **$3.272 \pm 0.417$** | **$0.1785 \pm 0.0121$** | **$0.0217 \pm 0.0085$** | **$0.2434 \to 0.2083$** | **Dramatically Improved ($-87.8\%$ ECE; S42: $0.1956 \to 0.0120$)** |
| **Within-Subject Longitudinal (Seed 42)** | `SimpleCNNBaseline` | $0.758$ | $0.1118$ | **$0.0684$** | $0.1351 \to 0.1261$ | Improved ($-38.8\%$ ECE reduction) |
| **Within-Subject Longitudinal (Seed 42)** | `ConvNeXt` `[Hard]` | $1.731$ | **$0.0362$** | $0.0698$ | **$0.1013 \to 0.1062$** | Already calibrated raw; 60-scan val set too small for post-hoc scaling |
| **Official Confounded Split (Seed 42)** | `ConvNeXt` `[Hard]` | $2.398$ | $0.2719$ | $0.1638$ | $0.3036 \to 0.2550$ | Partial mitigation of severe era-shift miscalibration |

### 4.2 Validation-Selected Reject-Option Triage (Slice-Level & 20-Slice Scan-Level)
To prevent test-set leakage when choosing the clinical decision threshold $\tau^*$, [`select_validation_triage_threshold()`](./train.py) selects $\tau^* \in [0.55, 0.90]$ **strictly on the validation split** by maximizing the utility $\mathcal{U}(\tau) = \text{Acc}_{\text{auto}}(\tau)\sqrt{\text{Cov}(\tau)} + 0.5\,\text{FN}_{\text{intercepted}}(\tau)$ subject to validation coverage $\ge 25\%$. This yields **$\tau^* = 0.70$** on `subject` (across all 3 seeds) and **$\tau^* = 0.80$** on `longitudinal`.

> **Definition of Metrics in Triage Tables:**
> - **Auto-Cleared Accuracy (`Acc_auto`):** Empirical accuracy strictly among accepted cases with $\max_c \hat{p}_c \ge \tau$.
> - **Intercepted FN Rate (`FN_intercepted`):** Fraction of all baseline False Negatives (missed AD cases at $\tau = 0.50$) that have $\max_c \hat{p}_c < \tau$ and are therefore **caught and routed to specialist review**.
> - **Oracle Hybrid Accuracy:** Theoretical upper-bound system accuracy $\text{Cov}(\tau)\cdot\text{Acc}_{\text{auto}}(\tau) + (1 - \text{Cov}(\tau))\cdot 1.0$, **explicitly assuming an oracle human neuroradiologist ($100\%$ accuracy) on referred cases**.

#### Table 4.2A — Strict Subject-Disjoint (`subject`, Seed 42) `ConvNeXt` Slice-Level vs. 20-Slice Scan-Level Triage
| Confidence Gate $\tau$ | Slice Coverage | Slice Auto-Cleared Acc | Slice FN Intercepted | **20-Slice Scan Coverage** | **Scan Auto-Cleared Acc** | **Scan FN Intercepted** | **Scan Oracle Hybrid Acc** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\tau = 0.50$ (No Triage)** | $100.0\%$ | $65.94\%$ | $0.0\%$ | $100.0\%$ ($239/239$) | $67.36\%$ | $0.0\%$ | $67.36\%$ |
| **$\tau^* = 0.70$ (Val-Selected)** | **$37.1\%$** | **$77.55\%$** | **$77.9\%$** | **$23.4\%$ ($56/239$)** | **$85.71\%$** | **$97.2\%$ ($35/36$)** | **$96.65\%$** |
| **$\tau = 0.75$** | $20.5\%$ | $80.37\%$ | $87.6\%$ | $9.2\%$ ($22/239$) | **$90.91\%$** | **$97.2\%$ ($35/36$)** | **$99.16\%$** |

#### Table 4.2B — Within-Subject Longitudinal (`longitudinal`, Seed 42) `ConvNeXt` Slice-Level vs. 20-Slice Scan-Level Triage
| Confidence Gate $\tau$ | Slice Coverage | Slice Auto-Cleared Acc | Slice FN Intercepted | **20-Slice Scan Coverage** | **Scan Auto-Cleared Acc** | **Scan FN Intercepted** | **Scan Oracle Hybrid Acc** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\tau = 0.50$ (No Triage)** | $100.0\%$ | $87.18\%$ | $0.0\%$ | $100.0\%$ ($62/62$) | $93.55\%$ | $0.0\%$ | $93.55\%$ |
| **$\tau = 0.70$** | $75.8\%$ | $92.13\%$ | $52.7\%$ | **$71.0\%$ ($44/62$)** | **$100.00\%$** | **$100.0\%$ ($2/2$)** | **$100.00\%$** |
| **$\tau = 0.75$** | $67.7\%$ | $93.09\%$ | $63.7\%$ | $51.6\%$ ($32/62$) | **$100.00\%$** | **$100.0\%$ ($2/2$)** | **$100.00\%$** |
| **$\tau^* = 0.80$ (Val-Selected)** | **$58.6\%$** | **$94.22\%$** | **$73.6\%$** | **$43.5\%$ ($27/62$)** | **$100.00\%$** | **$100.0\%$ ($2/2$)** | **$100.00\%$** |
| **$\tau = 0.85$** | $46.5\%$ | $95.67\%$ | $84.6\%$ | $30.6\%$ ($19/62$) | **$100.00\%$** | **$100.0\%$ ($2/2$)** | **$100.00\%$** |

> **Resolution of the Research Dilemma:** Even on the challenging **Strict Subject-Disjoint** split where raw cross-subject scan accuracy is $67.36\%$, combining **Temperature Scaling ($T^*=3.57$)** with **20-Slice Scan-Level Triage at the validation-selected gate $\tau^* = 0.70$** lifts auto-cleared scan accuracy to **$85.71\%$** (exceeding $80\%$) while intercepting **$97.22\%$ ($35/36$) of scan-level False Negatives** for specialist review! On the **Longitudinal** split, scan-level triage achieves **$100.00\%$ auto-cleared scan accuracy** with **$0$ unflagged missed AD scans** across $71.0\%$ of scans at $\tau = 0.70$ ($43.5\%$ at $\tau^* = 0.80$).

#### Calibration & Triage Curves — Strict Subject-Disjoint (`./images/subject/`) vs. Longitudinal (`./images/`)
![Subject-Disjoint Calibration and Triage Frontier](./images/subject/calibration_and_uncertainty.png)

![Longitudinal Calibration and Triage Frontier](./images/calibration_and_uncertainty.png)

---

## 5. Interpretability & Quantitative 4-Case Failure Autopsy

### 5.1 Stage-3 ($6\times 6$) Spatial Grad-CAM Attribution
In [`compute_spatial_attribution()`](./modules.py), we hook Grad-CAM onto **`stages[2]`** (Stage 3 of `ConvNeXt`, which has a **$6\times 6$ spatial grid** at $C=128$, rather than `stages[3]` which is only $3\times 3$) and bilinearly upsample the positive gradient-weighted activation map to $96\times 96$:

$$A_{\text{Grad-CAM}}(x, y) = \text{Upsample}_{6\times 6 \to 96\times 96}\!\left(\text{ReLU}\!\left(\sum_{k=1}^{128} \underbrace{\left[\frac{1}{36}\sum_{u=1}^{6}\sum_{v=1}^{6}\frac{\partial z_{c^*}}{\partial F_{k}(u,v)}\right]}_{\alpha_k^{c^*}} F_{k}\right)\right)$$

![Sample Test Predictions with Stage-3 (6x6) Grad-CAM Attribution Overlays](./images/sample_predictions.png)

### 5.2 Quantitative Four-Case Failure Autopsy: Intercepted Borderline (`< τ*`) vs. High-Confidence Escaped (`≥ τ*`) Errors
Rather than selecting only low-confidence errors near $0.50$ (which would make triage interception tautological), [`conduct_failure_case_autopsy()`](./predict.py) explicitly selects **four distinct failure archetypes** on the test set: **two borderline errors successfully intercepted by the validation-selected triage gate ($\hat{p} < \tau^*$)** and **two high-confidence errors that escaped the slice-level triage gate ($\hat{p} \ge \tau^*$)**. For each case, [`predict.py`](./predict.py) computes verifiable quantitative telemetry (exact slice index within that scan's 20-slice sagittal range `[s_min..s_max]`, relative parasagittal position `rel_pos`, brain foreground fraction, central dark-CSF ratio, `6×6` Grad-CAM peak coordinates, and 20-slice scan-level mean AD probability `scan_p(AD)`).

#### Autopsy Grid — Within-Subject Longitudinal (`./images/failure_autopsy.png`, [`results/failure_autopsy_longitudinal.json`](./results/failure_autopsy_longitudinal.json))
![Longitudinal 4-Case Failure Autopsy](./images/failure_autopsy.png)

| Case & Archetype (`longitudinal`, $\tau^*=0.80$) | Scan & Subject (`Slice [Range]`, `rel_pos`) | `SimpleCNN` Conf | `ConvNeXt` Raw $\to$ Cal Conf ($H$) | Slice Gate Verdict ($\tau^*=0.80$) | 20-Slice Scan Mean `p(AD)` & Scan Vote | Measured Quantitative Telemetry & Root Mechanism | Concrete Technical Remedy |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Case #1: Intercepted Borderline FN (`< τ*`)** | `AD_343288` (`019_S_4549`)<br>Slice `80` in `[75..94]` (`rel_pos=0.26`) | $0.866$ (`NC`) | $0.507 \to 0.504$ (`NC`)<br>$H = 1.000\text{ bits}$ | **INTERCEPTED (`< 0.80`)** | **`scan_p(AD) = 0.588` (RESCUED $\to$ `AD`)** | Foreground $= 0.58$, central dark-CSF ratio $= 0.37$, `6×6` Grad-CAM peak $= (3,1)$. Mid-temporal plane where `SimpleCNN` is overconfident wrong ($0.866$), while `ConvNeXt` outputs maximum entropy ($1.000$) and routes to review. | **20-Slice Scan Aggregation:** Already rescues this slice (`scan_p(AD)=0.588 ≥ 0.50`). |
| **Case #2: High-Conf Escaped FN (`≥ τ*`)** | `AD_363795` (`037_S_4879`)<br>Slice `80` in `[78..97]` (`rel_pos=0.11`) | $0.784$ (`NC`) | $0.994 \to 0.952$ (`NC`)<br>$H = 0.278\text{ bits}$ | **ESCAPED SLICE GATE (`≥ 0.80`)** | **`scan_p(AD) = 0.547` (RESCUED $\to$ `AD`)** | **Lateral Parasagittal Boundary Slice (`rel_pos = 0.11`):** Foreground $= 0.40$, central dark-CSF ratio $= 0.11$ (lateral to the lateral ventricle horn!), `6×6` Grad-CAM peak $= (3,2)$. Because the ventricle is not intersected at slice 80, single-slice `ConvNeXt` confidently predicts `NC` ($0.952$). | **Mandatory Scan-Level Triage & Slice-Position Weighting:** Averaging all 20 slices yields `scan_p(AD) = 0.547` (**rescuing the AD diagnosis** and routing the scan at $\tau^*=0.80$!). Down-weight extreme boundary slices (`rel_pos < 0.15` or `> 0.85`). |
| **Case #3: Intercepted Borderline FP (`< τ*`)** | `NC_1262194` (`033_S_1016`)<br>Slice `86` in `[78..97]` (`rel_pos=0.42`) | $0.511$ (`AD`) | $0.502 \to 0.501$ (`AD`)<br>$H = 1.000\text{ bits}$ | **INTERCEPTED (`< 0.80`)** | **`scan_p(AD) = 0.260` (RESCUED $\to$ `NC`)** | Foreground $= 0.47$, central dark-CSF ratio $= 0.41$, `6×6` Grad-CAM peak $= (5,2)$. Prominent central CSF space on single slice `86` causes a $0.501$ borderline split, which is both intercepted by $\tau^*=0.80$ and corrected by 20-slice averaging (`0.260`). | **Multi-Slice Consensus:** 20-slice scan probability (`0.260` AD / `0.740` NC) eliminates single-slice noise. |
| **Case #4: High-Conf Escaped FP (`≥ τ*`)** | `NC_1157071` (`007_S_6255`)<br>Slice `102` in `[94..113]` (`rel_pos=0.42`) | $0.752$ (`AD`) | $0.987 \to 0.924$ (`AD`)<br>$H = 0.388\text{ bits}$ | **ESCAPED SLICE GATE (`≥ 0.80`)** | `scan_p(AD) = 0.547` (**Persistent Scan FP**, but **INTERCEPTED by Scan Gate `< 0.80`!**) | Foreground $= 0.44$, **elevated central dark-CSF ratio $= 0.42$**, `6×6` Grad-CAM peak $= (5,1)$. Visually prominent ventricular/sulcal CSF cavity in a Cognitively Normal control mimics *ex-vacuo* AD atrophy across multiple slices (`scan_p(AD)=0.547`). | **Scan-Level Triage Gate + Demographic Conditioning:** While slice `102` escaped the slice gate ($0.924$), the **20-slice scan confidence is only $0.547 < \tau^*=0.80$**, so **Scan-Level Triage intercepts this patient**! Adding chronological age + ICV conditioning will further separate normal senile ventriculomegaly. |

#### Autopsy Grid — Strict Subject-Disjoint (`./images/subject/failure_autopsy.png`, [`results/failure_autopsy_subject.json`](./results/failure_autopsy_subject.json))
![Subject-Disjoint 4-Case Failure Autopsy](./images/subject/failure_autopsy.png)

> **Crucial Autopsy Insight:** Examining high-confidence escaped slice errors (Cases #2 and #4) reveals *why* 2D slice-level triage alone is insufficient: lateral parasagittal slices (`rel_pos = 0.11`, Case #2) miss the ventricular horn entirely (`central_csf_dark_fraction = 0.11`), producing high-confidence slice False Negatives ($0.952$). However, **20-slice scan-level aggregation + scan-level triage (`scan_p(AD) = 0.547 < τ* = 0.80`) rescues Case #2 to `AD` and intercepts both Case #2 and Case #4 for specialist review**, resulting in **$0\%$ unflagged scan-level False Negatives** on the longitudinal test cohort!

---

## 6. Actionable Manager Recommendation (Deployment Decision Memo)

**To:** Clinical Director of Neuroimaging Diagnostics  
**Recommendation:** **Deploy `ConvNeXt` strictly as a 20-Slice Scan-Aggregated, Temperature-Calibrated Triage Assistant with Validation-Selected Confidence Gating—Never as an Uncalibrated Single-Slice Classifier.**

1. **Enforce 20-Slice 3D Scan Aggregation Over Single 2D Slices:** As proven in Failure Autopsy Case #2, single peripheral slices (`rel_pos < 0.15`) frequently miss the lateral ventricles and produce high-confidence errors. Aggregating probabilities across all 20 slices per scan boosts `ConvNeXt` accuracy from **$87.18\% \to 93.55\%$** ($\text{Scan AUROC} = 0.9906$) on longitudinal follow-up and from **$65.94\% \to 69.87\%$** (majority vote) on unseen subjects at a total cost of only **$6.8\text{ ms}$ per 3D scan**.
2. **Deploy Scan-Level Selective Triage at Validation-Selected $\tau^*$:**
   - For **longitudinal cohort monitoring (`longitudinal`)**, setting $\tau = 0.70\text{--}0.80$ auto-clears **$43.5\%\text{--}71.0\%$ of 3D scans at $100.00\%$ empirical accuracy** while intercepting **$100\%$ of scan-level False Negatives**.
   - For **unseen cross-patient screening (`subject`)**, combining Temperature Scaling ($T^* = 3.57$, cutting ECE from $0.1956 \to 0.0120$) with scan-level triage at $\tau^* = 0.70$ raises auto-cleared scan accuracy from **$67.36\% \to 85.71\%$** while intercepting **$97.22\%$ ($35/36$) of missed AD scans**.
3. **Mitigate Cross-Subject Generalization Risk Before Standalone Unseen-Patient Deployment:** Because `ConvNeXt` trained from scratch on 476 subjects trails the `BatchNorm2d`-equipped `SimpleCNNBaseline` ($70.49\%$ vs. $78.31\%$ scan accuracy) on unseen subjects, standalone cross-patient deployment should wait for Phase 2 integration of an overlapping $3\times 3$ stem and age/ICV clinical covariates.

---

## 7. Repository Structure, Dependencies & Reproducibility Instructions

### 7.1 Directory Layout (`recognition/ConvNeXt_ADNI_s49852413/`)
```text
recognition/ConvNeXt_ADNI_s49852413/
├── README.md                  # Full technical report, v2.5 feasibility check-off & deployment memo
├── modules.py                 # Pure PyTorch ConvNeXt, SimpleCNNBaseline (BatchNorm2d), TempScaling & Stage-3 Grad-CAM
├── dataset.py                 # 3-protocol ADNI loader (subject, longitudinal, official) & scan/subject leakage audit
├── audit_protocol.py          # Standalone reproducible forensic audit of ADNI metadata & folder confounds
├── train.py                   # Multi-seed trainer, validation-only tau* selector, scan-level triage & profiler
├── predict.py                 # Calibrated inference, Stage-3 (6×6) Grad-CAM & 4-archetype quantitative autopsy
├── test_pipeline.py           # 7-test Pytest suite (AST zero-NumPy check, shapes, calibration, subject/scan hygiene)
├── slurm_train_rangpur.sh     # Slurm batch script for UQ Rangpur HPC (/home/groups/comp3710/ADNI/AD_NC)
├── results/                   # Tracked JSON evidence files for all benchmarks, 3-seed CIs, and failure autopsies
│   ├── protocol_audit.json
│   ├── benchmark_subject_3seeds.json
│   ├── benchmark_subject.json
│   ├── benchmark_longitudinal.json
│   ├── benchmark_official.json
│   ├── failure_autopsy_subject.json
│   └── failure_autopsy_longitudinal.json
└── images/                    # Tracked plots for longitudinal (images/) and subject-disjoint (images/subject/)
    ├── convnext_architecture.png
    ├── preprocessing_and_samples.png
    ├── training_curves.png
    ├── roc_pr_comparison.png
    ├── calibration_and_uncertainty.png
    ├── sample_predictions.png
    ├── failure_autopsy.png
    └── subject/
        ├── training_curves.png
        ├── roc_pr_comparison.png
        ├── calibration_and_uncertainty.png
        ├── sample_predictions.png
        └── failure_autopsy.png
```

### 7.2 Environment Setup & Exact Dependencies
Tested with Python `3.9.6` / `3.10+`:
- `torch == 2.8.0` (CUDA 12.x / Apple Silicon MPS)
- `torchvision == 0.23.0`
- `numpy == 2.0.2` (used in `dataset.py`, `train.py`, `predict.py`; **strictly zero imports in `modules.py`**)
- `scikit-learn == 1.6.1`, `matplotlib == 3.9.4`, `Pillow == 11.1.0`, `pytest == 8.4.2`

```bash
pip install torch torchvision numpy scikit-learn matplotlib Pillow pytest
```

### 7.3 Step-by-Step Reproducible Commands

```bash
# 1. Run AST compliance, model shape, calibration, and scan/subject leakage unit tests (7/7 passing)
pytest recognition/ConvNeXt_ADNI_s49852413/test_pipeline.py -v

# 2. Reproduce the forensic metadata & protocol confound audit (writes results/protocol_audit.json)
python recognition/ConvNeXt_ADNI_s49852413/audit_protocol.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC

# 3. Run 3-seed Strict Subject-Disjoint benchmark + Failure Autopsy (writes results/benchmark_subject_3seeds.json)
python recognition/ConvNeXt_ADNI_s49852413/train.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC \
    --split_mode subject --seeds 42,43,44 --epochs 12 --batch_size 128 --image_size 96 --lr 1.2e-3
python recognition/ConvNeXt_ADNI_s49852413/predict.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC --split_mode subject

# 4. Run Within-Subject Longitudinal Scan-Disjoint benchmark + Failure Autopsy
python recognition/ConvNeXt_ADNI_s49852413/train.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC \
    --split_mode longitudinal --seed 42 --epochs 12 --batch_size 128 --image_size 96 --lr 1.2e-3
python recognition/ConvNeXt_ADNI_s49852413/predict.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC --split_mode longitudinal

# 5. Run Official Confounded Split control experiment
python recognition/ConvNeXt_ADNI_s49852413/train.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC --split_mode official --seed 42
```

---

## 8. References & Code Attribution
1. **Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).** *A ConvNet for the 2020s.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986. Official repository adapted for block/stage conventions: [`https://github.com/facebookresearch/ConvNeXt`](https://github.com/facebookresearch/ConvNeXt).
2. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* International Conference on Machine Learning (ICML), PMLR 70:1321–1330. Reference implementation conventions: [`https://github.com/gpleiss/temperature_scaling`](https://github.com/gpleiss/temperature_scaling).
3. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* IEEE International Conference on Computer Vision (ICCV), pp. 618–626.
4. **Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., & Girshick, R. (2021).** *Early Convolutions Help Transformers See Better.* Advances in Neural Information Processing Systems (NeurIPS), 34, 30392–30400.
5. **El-Yaniv, R., & Wiener, Y. (2010).** *On the Foundations of Noise-free Selective Classification.* Journal of Machine Learning Research (JMLR), 11, 1605–1641.
6. **Jack, C. R., et al. (2008).** *The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods.* Journal of Magnetic Resonance Imaging, 27(4), 685–691.

---

## 9. Artificial Intelligence Usage & Verification Disclosure (Section 6 Mandatory Declaration)

In compliance with **Section 6 (Academic Integrity, Attribution & AI Usage Policy)** of the COMP3710 / COMP7710 Assessment Specification (**Version 2.5**):

- **External Code Attribution:** The `ConvNeXtBlock` and `ConvNeXt` class structure and naming conventions (`dwconv`, `pwconv1`, `pwconv2`, `gamma`, `downsample_layers`, `LayerNorm2d`) in [`modules.py`](./modules.py) are adapted from the official PyTorch reference implementation [`facebookresearch/ConvNeXt`](https://github.com/facebookresearch/ConvNeXt) (Liu et al., *CVPR 2022*) and re-implemented from scratch using pure `torch` and `torch.nn` for single-channel $96\times 96$ 2D MRI inputs (`depths=(2,2,6,2)`, `dims=(32,64,128,256)`), with zero `timm`, `torchvision.models`, or `numpy` dependencies.
- **AI Tools Used:** Google Antigravity (AI Pair-Programming Assistant) and Claude (Independent Code & Methodology Auditor).
- **How AI Assistance Was Used:**
  1. Assisted in scaffolding CLI runners, multi-panel Matplotlib plotting routines in [`train.py`](./train.py) and [`predict.py`](./predict.py), and pytest unit tests in [`test_pipeline.py`](./test_pipeline.py).
  2. Assisted in formatting Markdown tables, LaTeX equations, and structuring the technical report in [`README.md`](./README.md).
- **How AI-Assisted Output Was Audited, Debugged & Verified by the Student:**
  1. *Data Hygiene & Subject vs. Scan Leakage Audit:* Initial `verify_no_patient_leakage()` only checked 3D scan IDs (`Ixxxxxx`), returning `leakage_detected: False` on longitudinal splits that shared human subjects (`XXX_S_XXXX`). I upgraded [`dataset.py`](./dataset.py) and [`test_pipeline.py`](./test_pipeline.py) to track both `patient_id` (scan) and `subject_id` (human participant), fail loudly if any scan is unmapped in `meta_data_with_label.json`, and wrote [`audit_protocol.py`](./audit_protocol.py) to export exact reproducible proof ([`results/protocol_audit.json`](./results/protocol_audit.json)).
  2. *Fair Baseline & Hyperparameter Parity:* Upgraded `SimpleCNNBaseline` in [`modules.py`](./modules.py) with `BatchNorm2d` and unified `AdamW(weight_decay=1e-2)` across both models in [`train.py`](./train.py), revealing the genuine inductive-bias trade-off between `SimpleCNNBaseline` and `ConvNeXt` on the 476-subject cross-patient split.
  3. *Spatial Attribution & Sagittal Geometry Verification:* Verified that for $96\times 96$ input with `stem_stride=4`, `stages[3]` has a $3\times 3$ feature map while `stages[2]` has a $6\times 6$ feature map; updated [`compute_spatial_attribution()`](./modules.py) to hook `stages[2]` ($6\times 6$) and corrected slice orientation labels in [`predict.py`](./predict.py) to reflect **sagittal** anterior–posterior geometry.
  4. *Eliminating Threshold Peeking & Template Autopsies:* Updated [`train.py`](./train.py) to select the triage gate $\tau^*$ strictly on the validation set (`val_selected_tau`) and updated [`predict.py`](./predict.py) to audit both intercepted borderline errors (`< τ*`) and escaped high-confidence errors (`≥ τ*`) using measured slice/scan telemetry (`rel_pos` in `[67..114]`, dark-CSF fraction, and 20-slice scan probability rescue).
