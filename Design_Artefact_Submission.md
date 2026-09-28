# Design Artefact: Cross-Dataset Battery RUL Architecture
**Project Title:** Multi-Source Health Indicator Evaluation & Generalized Forecasting Pipeline  
**Student:** Ryan / Rrydicy  
**Course:** EECS Masters Thesis (Semester 2, 2026)  
**Working GitHub Repository:** [https://github.com/Rrydicy/Thesis-project-](https://github.com/Rrydicy/Thesis-project-)

---

## 1. End-to-End System Architecture Pipeline (The Design Artefact)

```
[ Stage 1: Multi-Source Repositories ]
├── Source A: NASA PCoE 18650 Li-ion Benchmark (LiCoO2, 2.0 Ah, 24°C / 4°C / 43°C)
├── Source B: University of Maryland CALCE Benchmark (LCO / NMC, 1.1 Ah - 1.35 Ah)
└── Synchronized Sensor Streams: Voltage V(t), Current I(t), Temperature T(t), EIS Impedance
                         │
                         ▼
[ Stage 2: Phase 1 - Correlation & Factor Importance Ranking ]
├── Candidate Health Indicators (HIs): Discharge duration (Δt), Voltage drop time (3.8V → 3.4V), Peak temp rise (ΔT_max), Re, Rct
├── Mathematical Evaluation:
│   ├── Pearson Correlation (r): Quantifies linear dependency
│   ├── Spearman Rank Correlation (ρ): Quantifies monotonic degradation
│   └── Grey Relational Analysis (GRA): Measures geometric trajectory similarity
└── Feature Screening: Filter top monotonic factors (|ρ| > 0.85) and eliminate multicollinearity
                         │
                         ▼
[ Stage 3: Phase 2 - Dimensionless Integration & Predictive Modeling ]
├── Dimensionless Unified Feature Space:
│   ├── State of Health: SoH = Q_k / Q_nominal (reconciles 2.0 Ah vs 1.1 Ah baseline discrepancy)
│   ├── Normalized SOC decay duration intervals
│   └── Relative temperature rise rates
├── Track A (Conventional ML): StandardScaler ➔ SVR (RBF), Random Forest, XGBoost
└── Track B (Deep Transfer Learning): 1D-CNN (Waveform dynamics) ➔ BiLSTM (Sequence memory)
    └── Transfer Protocol: Pre-train on NASA dataset ➔ Few-shot fine-tune on CALCE dataset
                         │
                         ▼
[ Stage 4: Generalized Forecasting & BMS Decision Support ]
├── Validation Protocol:
│   ├── Exp 1: In-domain baseline (NASA-to-NASA, CALCE-to-CALCE)
│   ├── Exp 2: Zero-shot cross-dataset generalization
│   └── Exp 3: Few-shot fine-tuning on unseen cell (B0018 / CS2_35)
├── Performance Metrics: RMSE, MAE, R² Score, NASA Asymmetric PHM Penalty Score
└── BMS Asset Management: 80% EOL Threshold Alarms & BESS Predictive Maintenance Scheduling
```

---

## 2. Accompanying Explanation (Word Count: 391 Words)
## 2. Accompanying Explanation (Word Count: 395 Words)

### 1. Project Problem, Objective, and Research Question
Residential Battery Energy Storage Systems (BESS) are critical for renewable integration and domestic energy security. However, repeated cycling causes non-linear electrochemical degradation, leading to capacity fade and internal resistance growth. Existing prognostic models are typically overfitted to single-source datasets and specific battery chemistries, restricting their practical utility across diverse real-world installations. The primary objective of this project is to develop a generalized, cross-dataset forecasting model to predict the Remaining Useful Life (RUL) of household BESS. This project addresses the central research question: *How can a two-phase framework combining operational factor ranking and universal feature space mapping achieve accurate, cross-chemistry RUL forecasting for household BESS?*
This project focuses on developing a universal forecasting model to predict the Remaining Useful Life (RUL) of residential Battery Energy Storage Systems (BESS). In household applications, battery reliability and safety are paramount across diverse commercial chemistries such as Lithium Iron Phosphate (LFP) and Nickel Manganese Cobalt (NMC). However, conventional prognostic approaches are typically tailored to single-source datasets and specific cell types, failing to generalize across varied real-world operating environments. To overcome this limitation, the proposed research establishes a cohesive two-phase workflow designed to bridge disparate data sources and achieve robust, cross-chemistry lifetime forecasting.

### 2. Purpose and Scope of the Design Artefact
The design artefact establishes a structured, two-phase data-to-decision workflow:
- **Phase 1 (Feature Importance Analysis):** Extract candidate operational degradation factors (e.g., discharge duration, voltage drop intervals, temperature dynamics, and internal resistance) across multi-source benchmark repositories (NASA PCoE and CALCE) and mathematically evaluate their correlation and ranking.
- **Phase 2 (Cross-Dataset Integration & Generalization):** Eliminate the scale and chemistry barriers separating disparate datasets by mapping heterogeneous factors into a unified universal feature space, followed by deploying machine learning (ML) and neural network (NN) architectures for generalized RUL forecasting.
The first phase is dedicated to **Feature Importance Analysis**. In this stage, candidate operational degradation factors—including discharge duration, partial voltage drop intervals, temperature dynamics, and internal resistance—are extracted from multi-source benchmark repositories such as NASA and CALCE. By mathematically evaluating and ranking the impact of these factors, the project builds a rigorous empirical understanding of how different operating parameters govern battery degradation across varied chemistries. Building upon these insights, the second phase focuses on **Cross-Dataset Integration and Generalization**. This phase aims to eliminate the inherent barriers separating disparate battery datasets by projecting heterogeneous degradation features into a unified, universal feature space, ultimately enabling accurate and chemistry-agnostic RUL predictions.

### 3. Justification of Major Design Decisions and Methodological Choices
- **Sequential Phased Approach:** A fundamental design decision is that the specific mathematical formulation of the universal space and the selection of ML/NN architectures depend directly on Phase 1 findings. Rigorously ranking factor sensitivities across chemistries (LFP, NMC, LCO) prevents premature model commitment and ensures that the universal space is grounded in empirically verified degradation drivers.
- **Open-Source & Reproducible Infrastructure:** To ensure transparency and practical deployability, models are developed using established open-source libraries (PyTorch and Scikit-Learn). All data pipelines, scripts, and baseline results are hosted in a public GitHub repository (https://github.com/Rrydicy/Thesis-project-), which is already initialized and version-controlled.
A key methodological decision in this design is that the specific machine learning (ML) and neural network (NN) architectures, as well as the exact mathematical formulations used to construct the universal feature space, will be determined based on the empirical findings of Phase 1. Evaluating factor rankings across chemistries beforehand prevents premature model commitment and ensures that subsequent predictive modeling is grounded in verified degradation mechanisms. To ensure transparency, reproducibility, and collaborative development, all pipelines and experimental workflows are built using open-source tools (PyTorch and Scikit-Learn) and managed within an already initialized GitHub repository (https://github.com/Rrydicy/Thesis-project-).

### 4. Key Risks, Constraints, and Dependencies
The principal technical risk is the mathematical feasibility of mapping heterogeneous dataset characteristics into a robust universal feature space without losing predictive fidelity. This risk is constrained by varying nominal capacities and voltage plateaus across cell types. To mitigate this risk, the framework employs dimensionless State of Health (SoH) normalization, relative voltage decay intervals, and a multi-tier validation protocol (in-domain testing, zero-shot evaluation, and few-shot fine-tuning using initial cycling data).

### 5. Guidance for Next Stages
This artefact serves as the implementation roadmap guiding: (1) multi-metric correlation ranking across candidate degradation factors (Weeks 4–5); (2) mathematical construction of the universal feature space and baseline ML benchmarking (Weeks 6–7); (3) deep transfer learning and few-shot calibration (Weeks 8–10); and (4) formulation of household BESS predictive maintenance recommendations (Weeks 11–12).
The primary technical risk facing this project lies in whether a mathematically robust mapping can be successfully constructed to project disparate dataset characteristics into a shared universal space without sacrificing prediction accuracy. This risk will be addressed through dimensionless health indicator normalization (such as State of Health) and systematically validated through a multi-tier testing protocol spanning in-domain testing, zero-shot cross-dataset evaluation, and few-shot fine-tuning. This design directly guides the forthcoming project milestones, starting with correlation-based factor ranking, followed by universal feature space formulation, model calibration, and the derivation of actionable predictive maintenance recommendations for domestic BESS.
