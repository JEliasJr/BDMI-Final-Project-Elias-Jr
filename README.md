# LLM Post-Training Alignment via Direct Preference Optimization (DPO)

## Repository Overview
This repository contains the complete implementation, structured preference datasets, and Low-Rank Adaptation (LoRA) training recipes developed for the **Final Project of the Big Data and Machine Intelligence (BDMI) Course at Tsinghua University**.

The framework establishes a rigorous mathematical post-training workflow leveraging Direct Preference Optimization (DPO) to systematically eliminate systemic cognitive vulnerabilities—specifically **sycophancy** (the pathological tendency to validate erroneous user assertions) and **optimism bias** (the underestimation of tail-risk parameters)—inherent in deep autoregressive foundational models.

The architecture encompasses two specialized expert deployment sub-systems:

1. **Operations Research & Stochastic Project Scheduling**  
   Grounds language model reasoning within the strict combinatorial constraints of the Resource-Constrained Project Scheduling Problem (RCPSP) using text-parsed PSPLIB J30 benchmark networks and 1,000-iteration Monte Carlo simulations to mitigate aggressive timeline estimation biases.

2. **Quantitative Financial Engineering**  
   Realigns policy networks to anchor asset risk analysis strictly on empirical multi-factor statistical matrices, actively penalizing qualitative market euphoria, confirmation bias, and retail narrative drift.

To shield the active policies from semantic entropy collapse and rigid sequence replication driven by DPO's non-vanishing implicit error weights, both frameworks enforce a highly regularized, synchronized **1.0-Epoch training horizon**, bounded by strict Kullback-Leibler (KL) regularizers.

---

## Core Theoretical Framework & Gradient Dynamics

Direct Preference Optimization bypasses explicit reward modeling by exploiting a closed-form analytical mapping between the optimal policy and the underlying latent reward function. Under the Bradley-Terry preference model, the generalized DPO objective function optimized across our architectures is formulated as:

$$
\mathcal{L}_{DPO}(\pi_{\theta}; \pi_{ref}) =
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma
\left(
\beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
\right]
$$

Where $\beta$ governs the stiffness of the KL-divergence constraint penalty against the frozen reference policy $\pi_{ref}$.

The corresponding gradient vector with respect to the trainable low-rank parameter weights $\theta$ is structured as:

$$
\nabla_{\theta}\mathcal{L}_{DPO}(\pi_{\theta}) =
-\mathbb{E}_{(x,y_w,y_l)}
\left[
\beta \cdot
\sigma(\hat{r}_{\theta}(x,y_l) - \hat{r}_{\theta}(x,y_w))
\left(
\nabla_{\theta}\log \pi_{\theta}(y_w|x)
-
\nabla_{\theta}\log \pi_{\theta}(y_l|x)
\right)
\right]
$$

### The Non-Vanishing Gradient Vulnerability

The analytical derivation reveals that the error weight
$\sigma(\hat{r}_{\theta}(x,y_l) - \hat{r}_{\theta}(x,y_w))$
approaches zero only asymptotically as the implicit reward margin goes to infinity ($z \rightarrow -\infty$).

Consequently, the gradient never vanishes even after the model achieves 100% classification accuracy on the training tokens. Left unregularized over multi-epoch horizons, the optimizer will ruthlessly distort parameter matrices to expand the log-probability gap, stripping token entropy and causing severe downstream generalization collapse.

We preemptively counter this pathology by contracting the training horizon to exactly **1.0 Epoch**, dampening step velocity, and amplifying KL elasticity.

---

## Repository Architecture

```text
├── financial_alignment/
│   ├── data/
│   │   └── large_scale_financial_dpo_(multi_factor).json
│   ├── configs/
│   │   ├── llama3_financial_train.yaml
│   │   ├── qwen_financial_train.yaml
│   │   └── deepseek_financial_train.yaml
│   └── bdmi_final_project_financial_v2.py
│
├── schedule_alignment/
│   ├── data/
│   │   └── large_scale_schedule_dpo.json
│   ├── configs/
│   │   ├── llama3_schedule_train.yaml
│   │   ├── qwen_schedule_train.yaml
│   │   └── deepseek_schedule_train.yaml
│   └── generate_psplib_dataset.py
│
└── README.md
```

---

# Technical Deployment Tracks & Empirical Performance

## 1. Operations Research & Stochastic Project Scheduling

### Target Objective
Mitigate structural optimism bias by training the model to issue strict, data-driven schedule vetoes, rejecting deterministic targets below the statistical P90 threshold calculated via Monte Carlo sampling.

### Dataset Generation
Built upon text-parsed acyclic graph dependencies mimicking a standard PSPLIB J30 benchmark instance (32 nodes, layered topological sorting passes, cross-layer dependency links).

### 1.0-Epoch Calibration Hardware Results

| Model Architecture | Final Train Loss | Grad Norm | Reward Accuracy | Reward Margin | Status |
|---|---|---|---|---|---|
| Meta-Llama 3 8B Instruct | 0.2207 | 0.2013 | 100.00% | 5.3094 | Success ✔ |
| Alibaba Qwen 2.5 7B Instruct | 0.3222 | 1.8860 | 100.00% | 2.7500 | Success ✔ |
| DeepSeek-R1-Distill-Qwen-7B | 0.4326 | 4.3170 | 100.00% | 1.4530 | Success ✔ |
| Google Gemma 2 9B Instruct | N/A | N/A | N/A | N/A | CUDA OOM ❌ |

### Note on Infrastructure Anomalies
Google Gemma 2 9B Instruct triggered a hardware CUDA Out-of-Memory (OOM) violation at Step 0 due to the high memory footprint of processing its massive 256,000 token vocabulary tensor concurrently across dual policy and reference models on standard GPU instances.

---

## 2. Quantitative Financial Engineering

### Target Objective
Eradicate sycophancy and text-level disclaimer loops when models face out-of-distribution high-volatility financial regimes, anchoring the final analytical tone strictly on mathematical boundaries.

### Dataset Generation
1,000 high-fidelity asset risk scenarios mapping short-term momentum vectors, trend line indices, and historical annualized volatility constraints.

### 1.0-Epoch Calibration Hardware Results

| Model Architecture | Final Train Loss | Grad Norm | Reward Accuracy | Reward Margin | Status |
|---|---|---|---|---|---|
| Meta-Llama 3 8B Instruct | 0.5852 | 14.730 | 99.38% | 0.2315 | Success ✔ |
| Alibaba Qwen 2.5 7B Instruct | 0.6404 | 14.660 | 84.38% | 0.1121 | Success ✔ |
| Google Gemma 2 9B Instruct | 0.4286 | 16.390 | 100.00% | 0.6343 | Success ✔ |
| DeepSeek-R1-Distill-Qwen-7B | 0.6779 | 9.4666 | 68.75% | 0.0333 | Success ✔ |

---

# Replication and Setup Guide

## 1. Environment Initialization

Clone the LLaMA-Factory core engine and install the corresponding compilation and metrics tracking dependencies using `pip` or the optimized `uv` package manager:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

pip install -e .[metrics]
pip install bitsandbytes networkx numpy
```

---

## 2. Pre-trained Weights (For Inference)

Download the model weights from the Releases tab of this repository and place them in the `models/` directory to run the code.

---

## 3. Data Onboarding & Registration

Move the generated `.json` preference dataset vectors into the internal LLaMA-Factory data catalog:

```bash
cp large_scale_schedule_dpo.json LLaMA-Factory/data/
```

Register the dataset entry inside `LLaMA-Factory/data/dataset_info.json`:

```json
"schedule_dpo": {
  "file_name": "large_scale_schedule_dpo.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

---

## 4. Triggering the Training Execution Pipeline

Execute the CLI alignment script calling the regularized 1.0-Epoch training configuration file:

```bash
llamafactory-cli train schedule_train_config.yaml
```

---

# Post-Alignment Cross-Architectural Synthesis

## Meta-Llama 3 8B
Achieved an optimal conversational and technical balance. Successfully isolated and addressed user emotional drivers while consistently issuing strict mathematical data vetoes.

## Alibaba Qwen 2.5 7B
Exhibited superior tabular data parsing. Dissected network dependencies into clean, objective risks and proposed actionable structural buffers without hallucinating metrics.

## DeepSeek-R1-Distill-7B
Displayed a unique structural anomaly under end-to-end DPO alignment. The preference dataset forced direct input-to-output mapping, colliding with the model's native RL computational flow.

This shattered internal cognitive boundaries, causing the model to skip its XML `<think>` validation markers, leak raw internal reasoning streams, and fall into infinite recursive persona tracking loops.

This empirical discovery confirms that reasoning architectures require specialized multi-stage decoupling scripts (e.g., GRPO) to preserve behavioral integrity.

---

# Declaration of AI Usage

In strict accordance with the Tsinghua University project evaluation directives, Artificial Intelligence tools (LLMs) were utilized exclusively to format Python execution scripts according to clean code standards and to debug the structural LaTeX typesetting of the final academic reports.

All fundamental methodological configurations, mathematical derivations, stochastic scenario definitions, experimental design execution, data parsing loops, and cross-architectural analytical interpretations are entirely original.
