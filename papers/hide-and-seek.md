# Hide and seek in transaction networks: a multi-agent framework for simulating and detecting money laundering activities

- **Authors:** Qianyu Wang¹ (corresponding, `qianyu.w@outlook.com`), Wei-Tek Tsai³˒⁴, Tianyu Shi⁵, Wang Tang⁶, Bowen Du¹˒². ¹ State Key Lab of Software Development Environment, Beihang Univ.; ² School of Transportation Science and Engineering, Beihang; ³ School of Insurance and Economics, Univ. of International Business and Economics, Beijing; ⁴ School of CS, Arizona State Univ.; ⁵ School of CS, Univ. of Toronto; ⁶ School of CS, Univ. of Leeds.
- **Venue:** *Complex & Intelligent Systems* (2025) 11:271, Springer. DOI 10.1007/s40747-025-01913-w. Received 2 Apr 2025, accepted 9 Apr 2025, published 6 May 2025. CC BY-NC-ND 4.0.
- **Code:** none linked. **Data:** "available from the corresponding author upon reasonable request".
- **Names used:** "Multiverse Simulation" (umbrella name). **SISTS** = *Strategic Interaction and Synthetic Transaction Simulation* (written "SISTIS" once, p. 2).
- **Task:** Make synthetic AML transaction data with an adversarial multi-agent RL simulator (launderers vs innocents vs a detector), then transfer the detector trained there to real data (Elliptic, Elliptic++).

## Core idea (TL;DR)

- **Hide-and-seek co-evolution, inspired by OpenAI's hide-and-seek (Baker et al. 2019, ref [11]).** There are three agent roles on a transaction graph, "each with local observations and objectives":
  - **Launderers** ("learning to cheat") "aim to conceal their trails within the network".
  - **Innocents** ("learning to simulate transaction") "conduct regular transactions".
  - **Inspectors / Detector** ("learning to detect") "utilize advanced detection algorithms to uncover illicit activities".
- The abstract groups these as two types: (1) the Detector and (2) Transaction agents, which split into laundering and legal. Launderers and innocents "continuously adapt their strategies to either evade or follow detection, respectively". The detector evolves alongside them, and the paper claims that this co-evolution yields realistic, varied laundering patterns and a stronger detector.
- **"Multiverse":** the framework "creates diverse virtual worlds, each with unique parameters to represent varying levels of illicit activity". The paper gives no further specification of how these worlds are parameterized.
- **Learner:** each agent's policy is a GAT over a **hierarchical relationship graph (HRG)**. Training is "designed based on MADDPG [35]" (Lowe et al. 2017), but Algorithm 1 is written as a per-agent Q-learning TD loss (see gotchas). Two terms are added to that TD loss:
  - a **DGDA-style domain-adaptation loss**: a variational graph auto-encoder with disentangled latents (semantic `Z_s`, domain `Z_d`, noise `Z_n`);
  - an **adversarial-perturbation loss** on the features, "also inspired to mimic the real-world dataset's missing values".
- **Sim-to-real:** the detector is pre-trained on synthetic data from a simulator "built upon AML sim", then fine-tuned on and tested against Elliptic and Elliptic++.
- **Headline result (Elliptic, Table 4):** Acc 95.2, Prec 94.5, Rec 93.8, F1 94.1, AUC 0.981, JS 0.06. It beats DANN, DGDA, GDA-SpecReg, GRDA and ISDAPGN on every metric, though DGDA's accuracy is 95.1.

## Framework (Figs. 1–3)

- **Fig. 1 loop:** the three agents send an *Action* to the *Synthetic environment* and get back an *Obs*. Then "Transfer the detector performance" goes to the *Real dataset*.
- **Fig. 2 (problem formulation):** source domain (synthetic) → *Transfer* → target domain (real), with a *Measure Similarity* step between them. Red edges mean "Ground truth data flow" and dashed blue edges mean "Noisy data flow". Caption: "The main problem is learning the behavior in synthetic data framework and trying to transfer the knowledge into real data".
- **Fig. 3 (SISTS architecture):**
  - Graph nodes are Launderers (blue) and Innocents (red). Inspectors (green) observe at two scales: a large "Macro level obs" circle and a small "Micro level obs" circle.
  - Source data flows into the graph, and the graph flows out to target data.
  - A side pipeline runs **Skill learning → Task mapping → Inference**.
  - The caption describes these steps:
    1. "the source task is learned using a skill learning method through multi-agent graph reinforcement learning, and source data is gathered for initial behavior learning".
    2. "a domain adaptation approach is adapted to find a latent space where the source and target data can be compared". "The inspector is used to detect money laundering behavior in this simulator. Both two types of agents evolve together."
    3. Finally, "the agent uses this function to learn raw target skills offline or to find inter-task mapping to leverage the source learned skills. Then, the target task is learned using pre-learned skills for money laundering activity detection".
- **The three challenges the paper says it addresses:**
  1. Getting large-scale simulations to converge.
  2. Simulating complex patterns that reflect real-world behaviour.
  3. Gradient updating and relationship modelling in multi-agent settings.
- **Components:**
  1. The HRG, a multi-layer representation of interaction levels.
  2. A GAT that weights neighbours by context.
  3. MARL integrated with the HRG and GAT.
- **Agent design:** "Each agent is characterized by its type (e.g., collaborator or competitor)… Collaborators seek to maximize a shared objective, while competitors aim to optimize their individual goals, often at the expense of others." The paper never maps the three roles onto collaborator and competitor.

## Preliminaries (equations verbatim)

### MARL as a POMDP

The POMDP tuple is `(S, A, P, R, O, N)`: state space, action set, transition probability, reward function, observation space, and number of agents. Each agent learns a policy `π^i : O^i → A^i` that solves

```math
\max_{\pi^i} \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R^i(s_t, a_t^i, a_t^{-i}) \mid \pi^i \right], \tag{1}
```

with `γ ∈ [0, 1)`, `s_t ∈ S`, `a_t^i` the action of agent `i`, and `a_t^{-i}` the actions of all other agents.

### GAT attention

```math
\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\mathrm{LeakyReLU}(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_k]))}, \tag{2}
```

Here `‖` is concatenation, **a** is a learnable weight vector, **W** is a weight matrix applied to each node's features `h`, and `𝒩(i)` is the set of neighbours of node `i`.

### Hierarchical relationship modeling (HRG)

```math
H = (V, E, L) \tag{3}
```

`V` is the set of nodes (agents), `E ⊆ V × V` is the set of edges (interactions), and `L: V → ℕ` assigns a level to each node. The interaction between `v_i, v_j ∈ V` is an edge `e_ij ∈ E`, which "can be weighted to represent the strength or capacity of interaction".

```math
l_{ij} = L(v_i) + L(v_j). \tag{4}
```

```math
s_{ij} = f(L(v_i), L(v_j)), \quad \text{where } f : \mathbb{N} \times \mathbb{N} \to \mathbb{R}_{\geq 0}. \tag{5}
```

```math
O_{global} = \sum_{l=1}^{max(L)} O_l, \tag{6}
```

Here `O_l` is "the objective contribution from level `l`". The local objective for an agent at level `l` is

```math
O^l_{local}(v_i) = \sum_{v_j : L(v_j) = l} g(s_{ij}, e_{ij}), \tag{7}
```

with `g: ℝ_{≥0} × E → ℝ`. Agents develop strategies based on their hierarchical level:

```math
\pi^l : O^l_{local} \times O_{global} \to A, \tag{8}
```

where `A` is the set of possible actions. Influence propagates as

```math
I_{ij} = \alpha^{L(v_i) - L(v_j)} s_{ij}, \tag{9}
```

with `α ∈ (0, 1)`, "a decay factor representing the diminution of influence with distance in the hierarchy".

## State, reward, action

### State (GNN-based)

For `G = (V, E)`:

```math
\mathbf{s}_i = \phi(\mathbf{a}_i, \mathbf{f}_i, \mathbf{h}_i), \tag{10}
```

Here **a**_i is the amounts of transactions involving `v_i`, **f**_i their frequencies, and **h**_i the historical patterns. `φ` aggregates these into a state a GNN can process.

```math
\mathbf{e}_{ij} = \psi(\mathbf{t}_{ij}, \mathbf{r}_{ij}), \tag{11}
```

Here **t**_ij is the transaction attributes ("such as transaction types or categories"), and **r**_ij is the relationship strength or capacity between `v_i` and `v_j`.

```math
\mathbf{S} = \{\mathbf{s}_i\}_{i=1}^{N}, \quad \mathbf{E} = \{\mathbf{e}_{ij}\}_{i,j=1}^{N}, \tag{12}
```

where `N` is the number of agents.

### Reward

Regulatory agent `i`, at time `t`:

```math
R^i_{reg}(s_t, a_t) =
\begin{cases}
+r_{detect}, & \text{if suspicious activity is correctly identified,} \\
-r_{miss}, & \text{if suspicious activity is missed,} \\
-r_{false}, & \text{if normal activity is wrongly flagged.}
\end{cases} \tag{13}
```

Transaction agent `j` (legitimate or laundering):

```math
R^j_{trans}(s_t, a_t) =
\begin{cases}
+r_{success}, & \text{if a transaction is successfully executed without detection,} \\
+r_{legit}, & \text{if a legitimate transaction is successfully executed,} \\
-r_{detected}, & \text{if a laundering transaction is detected.}
\end{cases} \tag{14}
```

`r_success` is the reward "for successfully executing a laundering transaction without being detected". `r_legit` is for a legitimate transaction, and `r_detected` is the penalty when a laundering transaction is detected. No reward magnitudes are reported.

### Action

```math
\mathbf{a}_i = [\alpha_{i1}, \alpha_{i2}, \ldots, \alpha_{ik}], \tag{15}
```

Each component `α_ij` can be a decision such as initiating a transaction, choosing its parameters, or investigating suspicious activity.

Transactional agent:

```math
\mathbf{a}_i = [\text{initiate}, \text{amount}, \text{recipient}, \text{strategy}], \tag{16}
```

- "initiate" is a binary decision to start a transaction or not.
- "amount" quantifies the funds being transacted.
- "recipient" identifies the receiving party.
- "strategy" represents "any tactic employed to conduct the transaction securely".

Regulatory agent:

```math
\mathbf{a}_i = [\text{select\_tx}, \text{analysis\_depth}, \text{update\_strategy}], \tag{17}
```

- "select_tx" is the transactions selected for investigation.
- "analysis_depth" is the level of scrutiny applied.
- "update_strategy" is an adjustment to the agent's approach to detection.

The joint action space is

```math
\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_N, \tag{18}
```

where `𝒜_i` is the set of all possible actions for agent `v_i`.

## Multi-agent graph learning with domain adaptation

Training is "designed based on MADDPG [35]", extended with a DA mechanism "inspired by disentanglement-based graph domain adaptation models (DGDA)".

**GAT update.** Nodes are agents and edges are relationships and interactions in a dynamic graph:

```math
\mathbf{h}'_i = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j \right), \tag{19}
```

**Domain adaptation component.** Variational graph auto-encoders "reconstruct and disentangle latent variables into semantic information, domain characteristics, and noise":
- **Z**_s is domain-invariant (semantic).
- **Z**_d is domain-specific attributes.
- **Z**_n is data-generation uncertainties (noise).

```math
\mathcal{L}_{DA} = \mathbb{E}_{\mathbf{Z}_s \sim q(\mathbf{Z}_s | \mathbf{X}, \mathbf{A})}[\log p(\mathbf{Y}_t | \mathbf{Z}_s)] - \beta D_{KL}(q(\mathbf{Z}_d | \mathbf{X}, \mathbf{A}) \,||\, p(\mathbf{Z}_d)), \tag{20}
```

- **X** and **A** are the feature and adjacency matrices.
- **Y**_t is "the target domain label".
- `q` is the encoded distributions of the latent variables, and `p` is the prior over the domain latents.
- `β` balances the two terms.

**Adversarial training.** Perturbations "maximize the prediction error" but are "constrained to ensure they do not alter the semantic meaning of the input data significantly". The loss is

```math
\mathcal{L}_{adv} = \max_{\delta \in \Delta} \mathbb{E}_{(\mathbf{X}+\delta, \mathbf{A})}[\log p(\mathbf{Y}_t | \mathbf{Z}_s(\mathbf{X} + \delta, \mathbf{A}))], \tag{21}
```

Here `δ` is a small perturbation in the permissible set `Δ`, and **Z**_s(·) denotes "the semantic latent variables affected by the perturbed inputs". The two losses combine as

```math
\mathcal{L} = \mathcal{L}_{DA} + \lambda \mathcal{L}_{adv}, \tag{22}
```

with `λ` a tuning parameter. No value of `λ` is reported.

### Algorithm 1 (verbatim)

The paper's title for it is "Multi-agent graph reinforcement learning with GAT and adversarial training".

```
Require: The number of agents N, maximum episodes M, learning rate α, discount factor γ.
Ensure: Optimized policies for all agents.
 1: Initialize graph attention network parameters for all agents θ_{1:N}.
 2: Initialize environment and get initial state s_0.
 3: for episode = 1, M do
 4:   Reset environment and get initial state s.
 5:   Generate adversarial examples by perturbing s within limits Δ.
 6:   while not done do
 7:     for each agent i = 1, N do
 8:       Observe local state s_i from both s and adversarial s.
 9:       Determine action a_i using policy derived from GAT with parameters θ_i.
10:     end for
11:     Execute actions a_{1:N} in the environment.
12:     Observe next state s' and reward r_{1:N}.
13:     Update the graph based on s' and a_{1:N}.
14:     for each agent i = 1, N do
15:       Calculate expected reward Q(s_i, a_i | θ_i).
16:       Calculate target y_i = r_i + γ max_{a'} Q(s'_i, a' | θ_i).
17:       Update θ_i by minimizing combined loss
          L(θ_i) = (y_i − Q(s_i, a_i | θ_i))^2 + L_DA + λ L_adv.
18:     end for
19:     s ← s'.
20:   end while
21: end for
```

## Implementation (Table 1, verbatim)

Runs used one NVIDIA Tesla V100 GPU with Adam at a learning rate of 1×10⁻³ and batch size 128. Dropout 0.5 is applied in the GAT layers. The latent dimensions are 256 (semantic), 64 (domain) and 128 (noise). `β` for the DA loss was "empirically determined to be 0.01".

| Parameter | Description | Value |
|---|---|---|
| MAX_EPISODES | Maximum number of episodes for training | 1000 |
| LEARNING_RATE | Learning rate for the Adam optimizer | 0.001 |
| DISCOUNT_FACTOR | Discount factor for future rewards | 0.99 |
| BATCH_SIZE | Number of samples per gradient update | 128 |
| DROPOUT_RATE | Dropout rate for regularization | 0.5 |
| LATENT_DIM | Dimensionality of the latent variables | 256 (semantic), 64 (domain), 128 (noise) |
| BETA_DA | Balancing parameter for domain adaptation loss | 0.01 |
| ENTROPY_COEF | Coefficient for entropy bonus in policy learning | 0.01 |
| GRADIENT_CLIP | Maximum norm for gradient clipping | 1.0 |

## Data

- **AML sim:** "Altman et al. [2] proposed the multi-agent simulator… Our agent is trained on the simulator built upon AML sim." It is used to initialize the simulation framework.
- **Elliptic:** Weber et al. [36].
- **Elliptic++:** Elmougy & Liu [8], described in the text as "more than 822 k Bitcoin wallet addresses (nodes), each with 56 features, and 1.27 m temporal interactions". The trained model is applied "to test on both Elliptic and Elliptic++ 's test datasets".
- The two sources' "features have discrepancies, which makes sim-to-real transfer [2] difficult".

**Table 2: statistics of the synthetic (AML Sim) and real Elliptic++ datasets.**

| Dataset | AML sim | Elliptic++ |
|---|---|---|
| # of nodes | 19,773 | 859,254 |
| # of edges | 96,531 | 995,508 |
| # of graphs | 1000 | 2000 |
| Avg degree | 4.88 | 1.16 |

**Table 3: example features of the synthetic (AML Sim) and real (Elliptic++) datasets.**

| AML Sim | Elliptic++ |
|---|---|
| CUSTOMER_ID | num_txs_as_sender |
| INIT_BALANCE | num_txs_as_receiver |
| COUNTRY | first_block_appeared_in |
| SENDER_ACCOUNT_ID | last_block_appeared_in |

## Baselines and ablations

**Graph domain adaptation baselines:**
- **DANN [37]** (Ganin et al. 2016): adversarial domain classifier with a gradient-reversal layer.
- **DGDA [38]** (Cai et al. 2024, "Graph domain adaptation: a generative view"): VGAE disentanglement into latent variables, trained with three supervised modules.
- **GDA-SpecReg [39]** (You et al. 2023): theory-grounded spectral regularization based on spectral smoothness and maximum frequency response.
- **GRDA [40]** (Xu et al. 2022): graph-relational DA that aligns domains using a domain graph and graph-structured discriminators.

**Multi-agent baseline:**
- **ISDAPGN [41]** (Shi et al. 2024): a distributed adaptive policy gradient with momentum, using Adam-type updates and importance sampling, with a convergence analysis.

**Ablations (each removes one component):**
- **NMAI:** no multi-agent interaction.
- **NHRG:** no hierarchical relationship graph.
- **NGAT:** no GAT.
- **NDA:** no domain-adaptation loss.
- **SRF:** static reward function, which replaces the role rewards with a non-adaptive one. For any agent `i` at time `t` (unnumbered equation, p. 10):

```math
R^i_{simple}(s_t, a_t) =
\begin{cases}
+r_{uniform}, & \text{for any executed action,} \\
-r_{uniform}, & \text{if any action leads to an undesired outcome.}
\end{cases}
```

`r_uniform` is "a flat reward for taking any action, which does not distinguish between the types of transactions conducted, whether they are legitimate or suspicious".

## Evaluation protocol

- **Dataset fidelity:**
  - A statistical comparison of synthetic and real data on AML-relevant features.
  - Feature-distribution matching with distance metrics and similarity scores.
- **Model performance:**
  - Train the baselines on synthetic data and on real data separately, and report accuracy, precision, recall and F1.
  - **Transfer learning:** pre-train on synthetic data, fine-tune on real data, and compare against models trained only on real data. The paper reports no numbers for the "real-only" arm.
- **Metrics:**
  - AUC-ROC.
  - **Jensen–Shannon divergence**, "to measure the similarity between the feature distributions of synthetic and real datasets".
  - Accuracy, precision, recall and F1:

```math
F1 = \frac{2 \cdot TP}{2 \cdot TP + FN + FP} \tag{23}
```
```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \tag{24}
```
```math
Precision = \frac{TP}{TP + FP} \tag{25}
```
```math
Recall = \frac{TP}{TP + FN} \tag{26}
```

**Research questions:**
- **RQ1:** how the model ensures realistic transaction feature distributions in the generated data.
- **RQ2:** how well a detector trained on simulated data transfers to real-world datasets.
- **RQ3:** how improvements in feature simulation enhance AML detection.
- **RQ4:** how the model adapts to varying complexity and subtlety of laundering behaviour.

## Results (tables verbatim; figure values approximate, read off the plots)

**Table 4: Elliptic dataset.**

| Metric | Our method | DANN [37] | DGDA [38] | GDA-SpecRe [39] | GRDA [40] | ISDAPGN [41] |
|---|---|---|---|---|---|---|
| Accuracy (%) | **95.2** | 92.0 | 95.1 | 94.0 | 93.2 | 94.5 |
| Precision (%) | **94.5** | 91.0 | 92.7 | 93.8 | 92.9 | 93.2 |
| Recall (%) | **93.8** | 90.5 | 92.1 | 93.5 | 92.0 | 92.8 |
| F1 score | **94.1** | 90.7 | 92.4 | 93.6 | 92.3 | 93.0 |
| AUC-ROC | **0.981** | 0.956 | 0.969 | 0.975 | 0.960 | 0.970 |
| Jensen–Shannon divergence | **0.06** | 0.12 | 0.09 | 0.07 | 0.11 | 0.08 |

**Table 5: Elliptic++ dataset.**

| Metric | Our method | DANN [37] | DGDA [38] | GDA-SpecRe [39] | GRDA [40] | ISDAPGN [41] |
|---|---|---|---|---|---|---|
| Accuracy (%) | **94.0** | 91.0 | 93.5 | 92.0 | 91.8 | 93.0 |
| Precision (%) | **93.2** | 90.2 | 91.3 | 92.5 | 91.5 | 92.0 |
| Recall (%) | **92.8** | 89.7 | 90.8 | 92.2 | 90.8 | 91.5 |
| F1 score | **93.0** | 89.9 | 91.0 | 92.3 | 91.0 | 91.8 |
| AUC-ROC | **0.970** | 0.950 | 0.960 | 0.965 | 0.955 | 0.965 |
| Jensen–Shannon divergence | **0.08** | 0.14 | 0.11 | 0.09 | 0.13 | 0.10 |

**Table 6: ablation.**

| Metric | Full model | NMAI | NHRG | NGAT | NDA | SRF |
|---|---|---|---|---|---|---|
| Accuracy (%) | **95.2** | 90.1 | 91.3 | 92.6 | 93.8 | 89.5 |
| Precision (%) | **94.5** | 89.2 | 90.4 | 91.7 | 93.0 | 88.7 |
| Recall (%) | **93.8** | 88.9 | 90.1 | 91.4 | 92.7 | 88.3 |
| F1 score | **94.1** | 89.0 | 90.2 | 91.5 | 92.8 | 88.5 |
| AUC-ROC | **0.981** | 0.948 | 0.957 | 0.965 | 0.976 | 0.942 |
| Jensen–Shannon divergence | **0.06** | 0.12 | 0.11 | 0.09 | 0.07 | 0.13 |

By F1, swapping in the static reward (SRF) hurts most, then removing multi-agent interaction (NMAI), then NHRG, then NGAT. Removing DA (NDA) hurts least.

**Table 7: sensitivity to scale.** The caption reads "relative to full scale". Note that the column order differs from Tables 4–5.

| Number of nodes/edges | Metric | Our method | DGDA | DANN | GDA-SpecRe | GRDA | ISDAPGN |
|---|---|---|---|---|---|---|---|
| 10,000/25,000 | Accuracy (%) | 91.5 | 90.8 | 89.7 | 90.2 | 89.5 | 90.0 |
| | Precision (%) | 90.9 | 90.2 | 89.2 | 90.0 | 89.0 | 89.8 |
| | Recall (%) | 90.3 | 89.7 | 88.5 | 89.8 | 88.9 | 89.3 |
| | F1 score | 90.6 | 90.0 | 88.9 | 89.9 | 88.8 | 89.6 |
| 100,000/250,000 | Accuracy (%) | 93.0 | 92.5 | 91.5 | 92.0 | 91.3 | 92.2 |
| | Precision (%) | 92.7 | 92.1 | 91.0 | 91.8 | 90.9 | 91.5 |
| | Recall (%) | 92.4 | 91.9 | 90.8 | 91.5 | 90.7 | 91.2 |
| | F1 score | 92.5 | 92.0 | 90.9 | 91.6 | 90.8 | 91.3 |
| 500,000/1,250,000 | Accuracy (%) | 94.0 | 93.8 | 92.8 | 93.5 | 92.9 | 93.3 |
| | Precision (%) | 93.7 | 93.6 | 92.5 | 93.3 | 92.6 | 93.0 |
| | Recall (%) | 93.5 | 93.4 | 92.3 | 93.1 | 92.5 | 92.9 |
| | F1 score | 93.6 | 93.5 | 92.4 | 93.2 | 92.6 | 93.1 |

The text names DGDA as the second-best baseline. It says DANN and GRDA "showed more significant drops in F1 scores as the data complexity increased". In the table, though, every model's F1 *rises* with size. DANN's and GRDA's F1 gaps to the proposed model also *narrow*, from 1.7 to 1.2 pts and from 1.8 to 1.0 pts.

**Figures (values read off the plots, approximate):**
- **Fig. 4: per-role training reward over 0–100 "Epochs", with shaded bands.** All roles start strongly negative: Detector ≈ −55k, Innocents ≈ −44k, Launders ≈ −27k. They reach about 0 around epochs 15–27, then oscillate and end near −14k, −11k and −7k respectively. The detector is lowest throughout. Caption: the fluctuations "represent adaptive strategy shifts rather than strict convergence", and "by extending the training to more epochs… fluctuations decreased".
- **Fig. 5: validation every 10 epochs on saved checkpoints.**
  - "Illicit/Licit Agent Ratio" falls from 0.50 to about 0.12 by epoch 100.
  - "Detection Rate" rises from about 0.61 to about 0.82, with a dip at epoch 20, and plateaus. The caption calls the plateau "a realistic balance" caused by adapting launderers.
- **Fig. 6:** histograms of 10 unnamed, normalized features ("Feature 1–10") for real vs synthetic data, which overlap closely. Per the caption, the "Selected features reflect a combination of high importance to the model's decision-making and strong alignment between datasets".
- **Fig. 7: F1 vs "Complexity Level (Amount of Noise)" from 0.2 to 0.7,** with the random seed fixed for all models.
  - Proposed model: about 0.85, 0.82, 0.69, 0.92, 0.79, 0.52.
  - Baselines: about 0.73–0.76, 0.76–0.81, 0.60–0.68, 0.84–0.88, 0.66–0.70, 0.27–0.30.
  - All curves dip at 0.4 and peak at 0.5. The proposed model's margin is largest at high noise (0.6–0.7).

## Claims, limitations, future work

- **Claimed findings:**
  1. The framework generates synthetic data that realistically mimics real transactions, including complex laundering patterns.
  2. Detectors trained on it work well when applied to real data, thanks to the DA techniques.
  3. Adversarial plus DA losses, driven by transaction–detector interaction, co-evolve more sophisticated laundering and detection strategies.
- **Stated limitations:**
  - Agents are assumed rational and to follow defined strategies.
  - The system's complexity makes its decisions hard to explain.
  - It has only been applied to money laundering. Credit-card fraud is suggested next.
- **Future work:** other financial systems and regulatory environments, scalability and complexity management, and blockchain integration.
- **Related-work context the paper uses to motivate itself:**
  - Starnini et al.: an Italian bank, 180M transactions over 6 months, smurf/motif detection, but few patterns and restricted access.
  - Harris et al.: 4469 Scotia Bank customers.
  - AMLworld is described as an improvement over MLDP and AMLsim.
  - FlowScope models transactions as a multipartite graph.

## Inconsistencies and gotchas (read before replicating)

1. **Typos and names:**
   - "SISTIS" (p. 2) is SISTS.
   - The baseline is "GDA-SpecReg" in the text (p. 9) but "GDA-SpecRe" in Tables 4–7 and Fig. 7.
   - The equation after Eq. 22 (SRF reward) is unnumbered.
2. **Overloaded symbols:**
   - `α`: attention coefficients (Eqs. 2, 19), action components (Eq. 15), hierarchy decay (Eq. 9), and learning rate (Alg. 1).
   - Bold **a**: the attention vector (Eq. 2), the transaction *amounts* inside the state (Eq. 10), and the action vector (Eqs. 15–17).
   - `s`: relationship strength `s_ij` (Eq. 5) vs states `s_t` and `s_i` (Eq. 1, Alg. 1) vs the node-state vector **s**_i (Eq. 10).
   - `E` / **E** / `e_ij`: edge set vs edge-feature set vs edge.
   - `A` / **A** / `𝒜`: action set vs adjacency matrix vs joint action space.
   - `h`: GAT features (Eqs. 2, 19) vs historical patterns (Eq. 10).
   - `f`: level-strength function (Eq. 5) vs frequencies **f**_i (Eq. 10).
   - `L`: level map vs losses `𝓛`.
   - `l`: `l_ij` (Eq. 4) vs the level index `l` (Eqs. 6–8).
3. **Dangling HRG pieces:**
   - `l_ij` (Eq. 4) and `I_ij` (Eq. 9) are defined but never used in the losses or in Algorithm 1.
   - How `O_l` (Eq. 6) relates to `O^l_local` (Eq. 7) is not given.
   - Eq. 8 makes *objectives* the policy's input domain.
   - `f`, `g`, `φ`, `ψ`, how levels `L(v)` are assigned, and the number of levels are all unspecified.
   - `s_ij` (Eq. 5) depends only on the two levels, so every pair with the same levels gets the same strength.
4. **Eq. 9's exponent is signed.** With `α ∈ (0,1)`, `L(v_i) < L(v_j)` gives `I_ij > s_ij` (for `s_ij > 0`), which amplifies influence. The stated intent, "diminution of influence with distance in the hierarchy", would need `|L(v_i) − L(v_j)|`. This is our observation; the paper writes it as shown.
5. **Signs in the losses.**
   - Eq. 20 is a log-likelihood minus β·KL, an ELBO-like quantity one would *maximize*. Yet Alg. 1 line 17 *minimizes* `TD² + 𝓛_DA + λ𝓛_adv`.
   - Eq. 21 takes the *max* over δ of the log-likelihood, while the text says the perturbations "maximize the prediction error".
   - A working implementation presumably minimizes `−E[log p(Y_t|Z_s)] + β KL` and uses the δ that maximizes the NLL. This is our interpretation, not stated in the paper.
6. **Eq. 20 is not a full VGAE objective.** There is a KL only on `Z_d`, no KL on `Z_s` or `Z_n`, and no reconstruction term, even though the text says "variational graph auto-encoders to reconstruct". It uses target labels **Y**_t, so the DA is not unsupervised. That fits the paper's fine-tuning on real data, but which target labels are used, and when, is not said.
7. **"Based on MADDPG", but Algorithm 1 is independent DQN-style learning:**
   - the target is `y_i = r_i + γ max_{a'} Q(s'_i, a'|θ_i)`;
   - the critic conditions only on the local state and the agent's own action, with no centralized critic and no separate actor;
   - there is no target network or replay buffer, although batch size is 128.
   - The action domain of "amount" (Eq. 16) is unspecified, and so is how `max_{a'}` is computed over the structured action vector.
   - Table 1's ENTROPY_COEF (0.01, "entropy bonus in policy learning") and GRADIENT_CLIP (1.0) match no equation.
   - Adversarial examples are generated once per episode (line 5, outside the `while` loop) by perturbing the *state* `s`, whereas Eq. 21 perturbs the features **X**.
8. **Episodes vs epochs:** MAX_EPISODES = 1000, but Figs. 4–5 plot 0–100 "Epochs". How the two relate, the episode length and the `done` condition are not given.
9. **Not reported:**
   - The number of agents `N` and the role ratios.
   - Any reward magnitude (`r_detect`, `r_miss`, `r_false`, `r_success`, `r_legit`, `r_detected`, `r_uniform`).
   - `λ`, `Δ`, the decay `α`.
   - GAT layers, heads and hidden sizes.
   - How the "multiverse" worlds are parameterized.
   - The classification target on Elliptic (node or edge, and which labels), the splits, the fine-tuning protocol, the number of seeds, and any variance or error bars.
   - Whether the baselines were trained on the same synthetic data.
10. **Dataset statistics:**
    - Elliptic++ in the text (822k nodes, 1.27M interactions) does not match Table 2 (859,254 nodes, 995,508 edges).
    - Table 2's "Avg degree" is edges/nodes (96,531/19,773 = 4.88 and 995,508/859,254 = 1.16), not 2E/N.
    - How the data is split into 1000 and 2000 "graphs" is not explained.
    - No statistics are given for plain Elliptic, even though Table 4 is on Elliptic.
11. **Which simulator:** the "AML sim" citation [2] is Altman et al. 2024, "Realistic synthetic financial transactions for anti-money laundering models". The paper's related work cites the same [2] when it describes AMLworld as an improvement over "MLDP and AMLsim". So the exact base simulator (AMLSim or AMLworld) is ambiguous.
12. **Feature alignment:** Table 3 lists different example features for the synthetic and real datasets, yet Fig. 6 overlays 10 features and JS divergence is computed. How features were mapped across domains, and what JS measures for baselines that generate no data, is not explained.
13. **Text vs evidence:**
    - "The lowest… JS divergence… indicates greater diversity in our simulations". A lower JS means *closer to real*; it does not imply diversity.
    - Fig. 7 is claimed "superior… across all levels", but at noise 0.3 the proposed model roughly ties DGDA (≈0.82 vs ≈0.81), and at 0.4 it ties GDA-SpecRe (≈0.69 vs ≈0.68). The dip at 0.4 and peak at 0.5 are unexplained, and "amount of noise" is undefined.
    - Table 7's text says baselines "drop" with complexity, but F1 rises with scale for every model.
14. **Inferences from the numbers:**
    - Table 6's Full-model column equals Table 4's "Our method" exactly, so the ablation was evaluated on Elliptic. This is inferred; it is not stated.
    - Table 7 has no full-scale row despite its caption, and its dataset is not stated.
15. **Fig. 5's "Illicit/Licit Agent Ratio"** is undefined. It is unclear how an agent ratio changes during training: roles switching, the fraction of illicit transactions, or something else.
16. **No explicit false-positive penalty for innocents.** Eq. 14 has no term for an innocent being wrongly flagged; only the regulator pays `−r_false`. `+r_legit` requires that the legitimate transaction is "successfully executed", but the paper never says whether a false flag blocks execution, so any indirect cost to innocents is unspecified. This is our observation, relevant to reward redesign.

## Takeaways for replication and synthesis

**Reusable design patterns:**
- A three-role adversarial co-evolution: the detector vs launderers, with innocents as an adaptive background. Only the detector transfers to real data.
- Role-specific outcome reward tables (Eqs. 13–14), with a flat-reward ablation (SRF) as the control. SRF was the most damaging ablation.
- Structured action vectors:
  - transaction agent: `[initiate, amount, recipient, strategy]`;
  - regulator: `[select_tx, analysis_depth, update_strategy]`. Transaction selection and analysis depth act as detector actions.
- Node state = φ(amounts, frequencies, history), and edge feature = ψ(transaction attributes, relationship strength). A GAT is the policy backbone.
- Hierarchical levels on agents, with level-based influence decay (fix the sign; see gotcha 4) and macro/micro observation scopes for inspectors (Fig. 3).
- DGDA-style disentangled latents for sim-to-real: semantic 256, domain 64, noise 128, β = 0.01. Only the semantic latent feeds the label head.
- Adversarial feature perturbation used as a stand-in for missing values in real data.
- Evaluation: JS divergence of feature distributions for fidelity, plus Acc/P/R/F1/AUC for transfer. Also a scale sweep (10k/25k to 500k/1.25M nodes/edges) and a noise sweep (0.2–0.7).

**Decisions a replicator must make, because the paper leaves them open (see gotcha 9):**
- the reward magnitudes and the agent counts per role;
- the level-assignment rule and `f`, `g`, `φ`, `ψ`;
- whether to use real MADDPG (a centralized critic with continuous actors) or the per-agent Q-learning in Algorithm 1;
- the loss signs, `λ` and `Δ`;
- how to align the synthetic and Elliptic feature spaces, and the Elliptic task, labels and splits;
- what the "multiverse" world parameters are.

**Numbers to beat or compare against (Elliptic):** F1 94.1, AUC 0.981. The strongest baseline is GDA-SpecRe at F1 93.6 and AUC 0.975. The paper reports no variance, so differences under about 1 pt are not demonstrably significant.
