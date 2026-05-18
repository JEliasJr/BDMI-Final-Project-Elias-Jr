# LLM Post-Training Alignment via Direct Preference Optimization (DPO)

## Repository Overview
This repository contains the complete implementation, structured preference datasets, and Low-Rank Adaptation (LoRA) training recipes developed for the **Final Project of the Big Data and Machine Intelligence (BDMI) Course at Tsinghua University**. 

The framework establishes a rigorous mathematical post-training workflow leveraging Direct Preference Optimization (DPO) to systematically eliminate systemic cognitive vulnerabilities—specifically **sycophancy** (the pathological tendency to validate erroneous user assertions) and **optimism bias** (the underestimation of tail-risk parameters)—inherent in deep autoregressive foundational models. 

The architecture encompasses two specialized expert deployment sub-systems:
1. **Operations Research & Stochastic Project Scheduling**: Grounds language model reasoning within the strict combinatorial constraints of the Resource-Constrained Project Scheduling Problem (RCPSP) using text-parsed PSPLIB J30 benchmark networks and 1,000-iteration Monte Carlo simulations to mitigate aggressive timeline estimation biases.
2. **Quantitative Financial Engineering**: Realigns policy networks to anchor asset risk analysis strictly on empirical multi-factor statistical matrices, actively penalizing qualitative market euphoria, confirmation bias, and retail narrative drift.

To shield the active policies from semantic entropy collapse and rigid sequence replication driven by DPO's non-vanishing implicit error weights, both frameworks enforce a highly regularized, synchronized **1.0-Epoch training horizon**, bounded by strict Kullback-Leibler (KL) regularizers.

---

## Core Theoretical Framework & Gradient Dynamics
Direct Preference Optimization bypasses explicit reward modeling by exploiting a closed-form analytical mapping between the optimal policy and the underlying latent reward function. Under the Bradley-Terry preference model, the generalized DPO objective function optimized across our architectures is formulated as:

$$\mathcal{L}_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Where $\beta$ governs the stiffness of the KL-divergence constraint penalty against the frozen reference policy $\pi_{ref}$. The corresponding gradient vector with respect to the trainable low-rank parameter weights $\theta$ is structured as:

$$\nabla_{\theta}\mathcal{L}_{DPO}(\pi_{\theta}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\beta \cdot \sigma(\hat{r}_{\theta}(x,y_l) - \hat{r}_{\theta}(x,y_w)) \left(\nabla_{\theta}\log \pi_{\theta}(y_w|x) - \nabla_{\theta}\log \pi_{\theta}(y_l|x)\right)\right]$$

### The Non-Vanishing Gradient Vulnerability
The analytical derivation reveals that the error weight $\sigma(\hat{r}_{\theta}(x,y_l) - \hat{r}_{\theta}(x,y_w))$ approaches zero only asymptotically as the implicit reward margin goes to infinity ($z \rightarrow -\infty$). Consequently, the gradient never vanishes even after the model achieves 100% classification accuracy on the training tokens. Left unregularized over multi-epoch horizons, the optimizer will ruthlessly distort parameter matrices to expand the log-probability gap, stripping token entropy and causing severe downstream generalization collapse. We preemptively counter this pathology by contracting the training horizon to exactly **1.0 Epoch**, dampening step velocity, and amplifying KL elasticity.
