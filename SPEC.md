# SPEC: Replicating "Hide and seek in transaction networks" (SISTS) on Elliptic++

Source digest: [`papers/hide-and-seek.md`](papers/hide-and-seek.md) (Wang et al., *Complex & Intelligent Systems* 2025). In this spec, "Gn" means gotcha *n* in the digest's "Inconsistencies and gotchas" section, and "En" means a new gotcha found while writing this spec (see §14).

Status: design spec, not yet implemented. Every value marked **(ours)** is a decision this replication makes because the paper does not report it. Every value marked **(paper)** is copied from the digest.

## 0. TL;DR

- **The data is Elliptic++** (Elmougy & Liu, KDD 2023), and it contains *two* tasks. We run both:
  - **Task T, transaction classification:** illicit vs licit, on the transaction (tx) graph. Elliptic++'s transaction file is a superset of Elliptic v1, so T is our analogue of the paper's Table 4 ("Elliptic").
  - **Task W, wallet classification:** illicit vs licit, on the wallet (address) graph. This is our analogue of Table 5 ("Elliptic++"). The paper's Table 3 lists Elliptic++ wallet features, which is why we read Table 5 as the wallet task.
- We replicate the paper's *method*: three co-evolving agent roles (launderers, innocents, detector) in a simulator. Policies are GATs on a hierarchical relationship graph (HRG). The detector is trained in the simulator with a DGDA-style DA loss and an adversarial loss, then fine-tuned on and evaluated against Elliptic++.
- Our HRG is the hierarchy that Bitcoin data already has: **transactions are level 1 and wallets are level 2**. It is built the same way in the simulator and in the real data. The simulator is UTXO-lite, so it emits exactly the graphs and named features that Elliptic++ has.
- **The two tasks are isolated.** Each MARL run trains two detectors with no shared parameters:
  - detector-T, the acting inspector, which sees transactions causally;
  - detector-W, a passive wallet model.

  Nothing that builds or trains a T model may contain data from after T's fitting window, ts 1–29 (E5, E6).
- We do **not** replicate the paper's baseline zoo (DANN, DGDA, GDA-SpecReg, GRDA, ISDAPGN) or most of its ablations. We keep only the controls needed to tell whether the method helps (§12):
  - target-only (no simulator);
  - static simulator (NMAI);
  - no DA (NDA).
- The paper's headline numbers (Acc 95.2 / 94.0, F1 94.1 / 93.0) are **not comparable** with ours (E10). Our sanity anchors are the official Elliptic++ notebook random forests (RF) on the official splits: illicit F1 0.828 (T) and 0.845 (W).
- Compute: local GTX 1650 Super (4 GB) for data prep, tests and small runs. A rented JarvisLabs A30 (24 GB) for MARL and the fine-tuning grid. The plan fits in about 25 A30-hours (§15).

## 1. Goal and scope

**Questions this replication answers** (adapted from the paper's RQ1–RQ4):

- **Q1 (fidelity, RQ1):** Do transactions and wallets from the co-evolved simulator match Elliptic++'s distributions on the shared named features better than those from a static, rule-based simulator? Measured by Jensen–Shannon (JS) divergence, which is our analogue of Fig. 6 and the JS rows.
- **Q2 (transfer, RQ2):** Does a detector pre-trained in the simulator and then fine-tuned on Elliptic++ beat the same detector trained on Elliptic++ alone? We test this at 5 %, 25 % and 100 % of the training labels, separately for T and W.
- **Q3 (mechanism, RQ3):** Does DA add anything? This is the paper's NDA ablation: the DA-specific terms `L_aux` are removed from both the MARL loop and fine-tuning, while source-label replay `L_y^src` is kept, so it measures the whole DA pipeline. Separately, does co-evolution (MARL) add anything over a static simulator?
- **Q4 (robustness, RQ4, optional, cheap):** How does test-time feature masking from 0.2 to 0.7 affect each arm? This mirrors Fig. 7, as an evaluation-only sweep.

**In scope:**
- the Elliptic++ data pipeline and its audit;
- the shared feature space;
- the HRG-GAT detector with DGDA-lite heads;
- the SISTS-lite UTXO simulator with three learning roles;
- Algorithm 1 as an implementable learner;
- transfer and evaluation on T and W;
- the reduced experiment matrix;
- a written report.

**Out of scope:**
- the DANN, DGDA, GDA-SpecReg, GRDA and ISDAPGN baselines;
- the NHRG and NGAT ablations;
- the Table 7 scale sweep;
- a separate Elliptic v1 run. T already contains v1's 165 features plus 17 more. A v1-only variant is optional extension X2;
- multi-task training on real tx and wallet labels together (forbidden by E6).

**Non-goals:**
- Matching the paper's numbers.
- A positive result. A clean negative result is a valid outcome.

## 2. Dataset reality check (verified 2026-11-08 on `dataset/`)

`dataset/` holds the nine official Elliptic++ CSVs (2.2 GB in total; guide repo `git-disl/EllipticPlusPlus`). Every count below was recomputed locally and matches the repo README.

| File | Rows | Content |
|---|---|---|
| `txs_features.csv` | 203,769 | `txId`, `Time step` (1–49), `Local_feature_1..93`, `Aggregate_feature_1..72` (the Elliptic v1 features, anonymized), and 17 named augmented features: `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in_BTC_{min,max,mean,median,total}`, `out_BTC_{min,max,mean,median,total}` |
| `txs_classes.csv` | 203,769 | class 1 illicit 4,545 / 2 licit 42,019 / 3 unknown 157,205 |
| `txs_edgelist.csv` | 234,355 | tx→tx money flow. **No edge crosses a time step.** No self-loops. |
| `wallets_features.csv` | 1,268,260 | `address`, `Time step`, and 55 named wallet features (counts, block heights, BTC sent/received/transacted, fees, blocks between txs, counterparties) |
| `wallets_classes.csv` | 822,942 | class 1 illicit 14,266 / 2 licit 251,088 / 3 unknown 557,588 |
| `wallets_features_classes_combined.csv` | 1,268,260 | the two files above, joined. **Sorted by address string.** |
| `AddrAddr_edgelist.csv` | 2,868,964 | input address → output address. 2,784,344 distinct pairs, 45,981 self-loops. Max total degree 22,996, median 2. |
| `AddrTx_edgelist.csv` / `TxAddr_edgelist.csv` | 477,117 / 837,124 | address → tx (inputs) and tx → address (outputs). All distinct. |

**Structural facts:**
- Each time step's tx graph is **exactly one weakly connected component**, with 1,089–7,880 txs (median 4,291).
- `AddrAddr` is exactly `AddrTx ⋈ TxAddr` on `txId`. Every input × output pair of a tx is one row.
- Every wallet appears in the address–tx links.
- 965 txs (0.5 %) have **all 17 augmented features null** and no address links (E7).

**Wallet-file facts (E4, E5):**
- `wallets_features.csv` has only 920,691 distinct `(address, Time step)` pairs. 246,545 pairs are duplicated, one of them 568 times.
- **Every feature of an address is identical across all of its rows.** These are lifetime aggregates over all 49 time steps, not per-step snapshots. `num_timesteps_appeared_in` equals the number of distinct time steps of the address's txs in 100 % of cases.
- After dropping `Time step` and de-duplicating, there is exactly one row per address (822,942 rows).

**Label facts (E6):**
- A wallet is illicit **if and only if** it is linked to at least one illicit tx. This holds for 14,266 of 14,266 illicit wallets, and for 0 of the licit and unknown wallets.
- Every licit wallet is linked to a licit tx. So are 14.3 % of unknown wallets.
- **Wallet labels are therefore a function of tx labels.**

**Feature-definition audit, on the subset we can recompute from the edge lists:**

| Feature | Recomputed as | Match |
|---|---|---|
| `in_txs_degree` / `out_txs_degree` | tx-edge in-degree / out-degree | 99.7 % / 99.7 % |
| `num_input_addresses` / `num_output_addresses` | `AddrTx` / `TxAddr` count | 94.6 % / 99.8 % (the feature is never smaller) |
| `total_BTC` | `out_BTC_total` | 100 % |
| `fees` | `in_BTC_total − out_BTC_total` | 100 % |
| `num_txs_as_sender` / `num_txs_as receiver` | distinct txs in `AddrTx` / `TxAddr` | 99.97 % / 98.96 % |
| `total_txs` | distinct linked txs | 100 % |
| `num_timesteps_appeared_in` | distinct time steps of linked txs | 100 % |

**Official splits (E3), reproduced from the guide repo's notebooks:**

| Task | Rule | Train (illicit / licit) | Test (illicit / licit) |
|---|---|---|---|
| T | Drop rows with nulls (the 965), drop unknown. Train `Time step < 35`, test `≥ 35`. | 3,462 / 26,237 | 1,083 / 15,263 (6.6 % illicit) |
| W | Drop `Time step` and de-duplicate, drop unknown, then `train_test_split(test_size=0.30, shuffle=False)` in file order, which is **address-string order**. | 185,747 rows: 11,377 / 174,370 | 79,607 rows: 2,889 / 76,718 (3.6 % illicit) |

The W split is not temporal. Train contains only `1…` (P2PKH) addresses. Test contains 43,011 `1…` addresses and **all 36,596 `3…` (P2SH) addresses**, so it combines a random split with an address-type shift. The first-appearance time distributions of train and test are identical: the 10th, 50th and 90th percentiles are ts 2, 22 and 45 on both sides. **We use both official splits as is**, so our numbers line up with published Elliptic++ baselines.

**Licence and confidentiality (E14):** neither the raw data nor anything derived from it is ever committed. That includes parquet files, id maps and features. `.gitignore` covers `dataset/`, `data/` and `runs/`. Reports may contain aggregate statistics and metrics only, never per-address or per-tx rows.

## 3. Hardware, environment and budget

| | Local | JarvisLabs |
|---|---|---|
| GPU | GTX 1650 Super, 4 GB (≈3.3 GB free with the desktop running), TU116 / sm_75. No tensor cores, no bf16. | A30, 24 GB, sm_80, bf16/TF32. Containers only, region IN2. |
| CPU / RAM | Ryzen 5 3600 (12 threads), **16 GB RAM, no swap** | 16 vCPU, 64 GB RAM |
| Disk | 1.3 TB free on `/home`. `/tmp` is a 7.8 GB RAM-backed tmpfs (E11). | 100 GB default. **Only `/home` survives a pause** (JarvisLabs containers). |
| Price | free | $0.41/h on-demand, $0.29/h spot (preemptible), as of 2026-11-08 (`jl gpus`) |

The whole dataset fits in RAM on both machines. As float32 it is about 150 MB of tx features, 180 MB of wallet features, and about 4.4M edges. So no out-of-core tooling is needed. Still, write scratch files under `data/`, not `/tmp`.

**Environment (E12):**
- `uv`-managed Python **3.12**. The system Python is 3.14, which has thin wheel coverage.
- One `pyproject.toml`, with a lockfile. The dependencies are `torch`, `torch_geometric` (pure Python, **no** `torch_scatter`, `torch_sparse` or `pyg-lib`), `polars`, `pyarrow`, `numpy`, `numba`, `scikit-learn`, `pyyaml` and `pytest`.
- Pin the torch build at implementation time. On both machines, assert that `torch.cuda.get_arch_list()` contains `sm_75` and `sm_80`, and that a one-batch GAT forward/backward runs. On JarvisLabs, check `nvidia-smi` first, and pick a CUDA wheel that the driver supports.
- Neighbour sampling is done **by our own numba CSR sampler** (§5 D2). Training never needs `NeighborLoader` or a compiled PyG extension.

**Precision:**
- Local: fp32, with AMP fp16 allowed only for the detector forward pass.
- A30: bf16 autocast for GAT layers, fp32 for the loss and the optimizer.

**Budget:**
- Plan for **≤ 40 A30-hours**, about $16 on-demand. The estimate in §15 is about 25 h.
- Use spot only for jobs that checkpoint at least every 15 minutes and resume automatically.
- One container (100 GB), paused between sessions, destroyed when the project ends.
- Before ending any session, run `jl list --json`, then pause or destroy what we created.

**Data movement (E13):** upload the processed `data/parquet/` and `data/graph/` directories with `jl upload`. They are well under 1 GB. Never upload the raw CSVs, and never put data in the repo.

## 4. Repository layout

```
SPEC.md
pyproject.toml, uv.lock
configs/            # one YAML per stage, arm and task. Every run records its full config and git SHA.
src/hns/
  data/   load.py audit.py ids.py graph.py causal.py split.py norm.py
  feats/  tx_feats.py wallet_feats.py   # the named-feature definitions, shared by real-data audit and simulator
  sim/    utxo.py world.py entities.py typologies.py env.py labels.py multiverse.py calibrate.py export.py
  models/ hrg.py hrgat.py detector.py qnet.py sampler.py
  marl/   buffer.py learner.py train.py
  transfer/ finetune.py adv.py
  eval/   metrics.py js.py bootstrap.py report.py rf_anchor.py
tests/              # unit tests, plus tiny synthetic fixtures. Never real data.
scripts/            # jl_*.sh helpers (create, upload, run, pull results, pause)
data/   (git-ignored)  parquet/ graph/ splits/ norm/
runs/   (git-ignored)  <stage>/<task>/<arm>/<seed>/{config.yaml, metrics.jsonl, ckpt/, preds_*.parquet}
reports/            # committed: aggregate tables and figures only
```

## 5. Data pipeline (local, fast)

**D0, load.**
- Read every CSV with polars, using explicit dtypes: float32 features, int64 `txId`, int16 `Time step`, int8 `class`.
- Map address strings to dense int32 `wid`. Use sorted-unique order, so `wid` order equals file order; the W split relies on this.
- Map `txId` to dense `tid`.
- Write parquet to `data/parquet/`. De-duplicate wallets to one row per address (E4), and drop the `Time step` column from the wallet table.

**D1, audit.** Write `data/audit.json`, and fail loudly if any of these is false:
- every count, null count and structural fact in §2;
- the label rule "illicit wallet ⟺ linked to an illicit tx" (E6);
- wallet features are constant per address;
- the feature-definition match rates are at least the §2 values minus 0.5 points.

**D2, graphs.**
- **tx→tx flow:** CSR in both directions.
- **Address–tx links:** typed `in` (address is an input of the tx) or `out` (address receives an output), with the tx's time step. CSR in both directions.
- **AddrAddr:** collapsed to distinct directed pairs, with `log1p(multiplicity)` and the first time step of the pair. Self-loops are removed from the edge list and turned into a per-wallet `self_loop_frac` feature (E20).
- The numba sampler handles hubs: the maximum wallet degree is 22,996, and the sampler draws uniformly over distinct neighbours.

**D3, causal wallet state for T (E5).** For every `(wid, t)`, compute the counts of linked txs **with time step ≤ t**: as input, as output, total, and the number of distinct time steps. Store them as a sorted `(wid, t)` table, queried with `searchsorted`. These are the only wallet features T may use.

**D4, splits and label fractions.**
- Reproduce the §2 official splits exactly.
- **Val:**
  - T: the last 5 train time steps, ts 30–34, are val, so T trains on ts 1–29. This is a temporal val, because T's test is temporal.
  - W: a stratified, seeded random 10 % of official-train.

  In both tasks, val is used for early stopping, thresholds and λ, and test is scored once per final run.
- The **fitting split** is train minus val. For W, it also includes the unknown wallets, which are unlabelled and so not in either official split. All "fitting split" references below mean this.
- **Label fractions** `f` ∈ {0.05, 0.25, 1.0} of the fitting set: stratified, seeded, and nested (the 5 % subset sits inside the 25 % subset). For T, they are stratified per time step.
- Unknown-class nodes stay in every graph as unlabelled nodes. They are never targets.

**D5, frozen normalization (E16).**
- Transform:
  - each feature follows its transform in the feature manifest (§7). By default, counts, BTC amounts, fees and `fees_as_share_total` (which reaches 3.64) get `log1p`;
  - block heights are converted to "time steps since the dataset's first block, 391,200", with 1,975 blocks per time step (E19);
  - true fractions stay in [0, 1].
- Then standardize with a mean and sd per feature, fitted **once per task** on that task's real fitting split, label-blind. The moments are frozen for both domains and every arm.
  - **T:** labelled and unlabelled txs in ts 1–29, plus causal wallet counts at t ≤ 29.
  - **W:** official-train wallets minus the val wallets, plus all unknown wallets.
  - T never uses W's moments, and W never uses T's.
- JS histograms use 50 equal-width bins over ranges fixed now, on the pre-standardization scale. The range is the real fitting split's [0.5th, 99.5th] percentile, widened by 10 %, and the outermost bins collect everything beyond it.

**Leakage rules (E5, E6, E15):**
- Labels are used only as targets, **never as features**. No feature may encode a neighbour's class.
- **Never use one task's real labels in the other task**: not as input, not as an auxiliary loss, not for calibration.
  - The W label is literally the OR of the linked tx labels (E6).
  - The official W test wallets are linked to many T-train txs.
- **T is causal.** The HRG for a tx at time step t contains only txs with time step ≤ t, and wallet nodes with D3's causal counts. It never uses the lifetime wallet features.
- **T isolation.** Everything that builds or trains a T model uses real data from **ts 1–29 only**: calibration (§9.6), normalization (D5), in-loop DA batches (§10) and fine-tuning. Val (ts 30–34) is used only for selection.
  - W-view data never reaches a T model: no lifetime wallet features, and no W-split wallets.
  - Why it matters: unknown wallets alone link to 60,639 txs in ts 35–49, so W-view data carries T-test information.
- W models may use W-view data from the W fitting split.
- Calibration (§9.6) is **label-blind**. It uses real features and structure from the fitting splits, plus the public class totals in the README. Hyperparameters and gates use **val**. Test is scored once per final run.

## 6. Tasks, metrics and evaluation protocol

**Tasks:**
- T classifies each labelled tx (illicit = 1).
- W classifies each labelled wallet (illicit = 1).

The model outputs a probability per node.

**Metrics, reported on test for each task:**
- **Primary:** illicit-class F1, at the threshold that maximizes F1 on **val**. Also report illicit **PR-AUC**, since base rates are 6.6 % (T) and 3.6 % (W).
- ROC-AUC, plus illicit precision and recall.
- Recall@k with k = 1 % and 5 % of test nodes.
- For comparison only: accuracy, macro-F1, weighted-F1 and micro-F1, with the confusion matrix. The paper's "F1" is undefined, so we report every reading of it (E10).
- **T only:** illicit F1 per test time step, ts 35–49. The dark-market shutdown at ts 43 is known to break every Elliptic model (E18).

**Seeds and uncertainty:**
- 3 seeds per configuration (seeds 0, 1, 2). Each seed controls model init, batch order, the label subsample and the W val split.
- Report mean ± sd over seeds.
- For each key comparison and seed, report a paired bootstrap 95 % CI of Δ illicit-F1 and Δ PR-AUC over test nodes (1,000 resamples). For T, resample whole time steps. Call a difference "supported" only if the CI excludes 0 in all three seeds. The paper reports no variance (G13), and we must not repeat that.

**Training protocol (all GNN arms):**
- Adam, lr 1e-3, batch 128 target nodes, gradient clip 1.0, dropout 0.5 (paper, Table 1).
- At most 50 epochs, with early stopping on val illicit-F1 (patience 5), restoring the best checkpoint.
- BCE with `pos_weight = sqrt(n_neg / n_pos)` (ours).
- On the 1650S, use micro-batches of 32 with 4-step gradient accumulation if 128 does not fit.

**Pipeline anchor (S0), cheap and always run.** Reproduce the official notebook RF (50 trees) on the official splits, with the notebook's feature set:
- T: all 182 tx features, nulls dropped.
- W: all 55 wallet features.
- The notebook min-max-scales with statistics fitted on train *and* test. We fit on train only, which is a small difference in itself, so we report both versions.

**Acceptance:** illicit F1 within ±0.03 of 0.828 (T) and 0.845 (W). If S0 misses, the data pipeline or split is wrong, and nothing downstream runs. S0 is a pipeline check, not a model comparison.

## 7. Shared feature space (sim ↔ real alignment; G12, E8, E9, E16)

Elliptic++ gives us **named** features, so most of the space is shared by definition. `hns.feats` implements each named feature once. That code is used both by the D1 audit (on the recomputable subset) and by the simulator. Where the audit shows a definition differs from the obvious reading, the code follows the audited behaviour. For example, `num_input_addresses` is never smaller than the link count.

| Node type | Shared channel (computed identically in sim and real) | Target-only adapter (real only, zero-init) |
|---|---|---|
| tx | the 17 augmented features; tx-graph in/out degree within the view | the 165 anonymized `Local_*` and `Aggregate_*` features |
| wallet, in the **T view** (causal) | D3 causal counts up to t: as input, as output, total, distinct time steps | none |
| wallet, in the **W view** (lifetime) | all 55 named wallet features; `self_loop_frac`; AddrAddr in/out degree | none |

**Feature manifest (E9).** `hns/feats/manifest.yaml` lists every shared feature: the 17 tx features, the 55 wallet features, the causal counts and the derived features. Each entry gives:
- its formula over the ledger;
- its transform;
- its empty-history sentinel;
- its status: **audited**, meaning it matches recomputation on real data (§2), or **assumed**, meaning it cannot be recomputed from Elliptic++, which has no per-address amounts or block heights.

Facts the manifest must encode, all checked on the real data:
- Empty histories are `0`. For example, `btc_sent_*` is 0 for wallets that never send, `btc_received_*` is 0 for wallets that never receive, and `blocks_btwn_*` is 0 for wallets with a single tx.
- `fees_*` are **non-zero for wallets that only receive**. So a tx's fee counts toward every participating address, not only the senders. The exact per-address share is an *assumption*: we use the full tx fee.
- `fees_as_share_total` reaches 3.64, so it sums per-tx shares. It gets `log1p`, not the [0, 1] rule. The other `fees_as_share_*` columns are ≤ 0.27.
- `first_sent_block` is non-zero for some wallets that never send. This is an unexplained quirk of the real data, and it is not reproduced. The assumed definition is "first block in which the address is an input".
- `self_loop_frac` = the number of self-loop AddrAddr rows ÷ the number of AddrAddr rows where the wallet is the input address, and 0 if there are none.

The JS report lists assumed features separately.

**Rules:**
- **No missing-value indicator (E7).** The 965 null-augmented txs are 519 licit, 446 unknown and **0 illicit**, so "missing" would be a licit shortcut.
  - T follows the notebook and drops them from the graph and the labels.
  - W never sees them, because they have no address links.
- The adapter's linear layer starts at zero and is added to the shared projection *before* the first GAT layer. So a simulator-pretrained detector, moved to Elliptic++, starts out as exactly its shared-only self. **Zero-shot** evaluation uses the shared channel only.
- The simulator emits no simulator-only features. Anything the real data lacks is not modelled, so there is no source adapter.

**What JS divergence measures here.**
- Compute it per feature over the shared channel, separately for tx nodes and W-view wallet nodes.
- Report it for all nodes, and per class against the simulator's labels and the real fitting split's labels. The per-class JS is a reported statistic only; it never feeds calibration.
- Use the §5 D5 bins, with log base 2, so JS ∈ [0, 1].
- Report the mean over features, the per-feature values, and Fig. 6's analogue: the 10 features with the highest S0-RF importance, overlaid.
- "Real" means the T-safe data (§9.6) for calibration, and each task's test split for the final report.

JS is defined only for the simulator arms. The paper's JS for methods that generate no data (G12, G13) has no counterpart here.

**Shared names are not shared distributions.** A shared feature name and formula do not make the simulator and the real data equivalent in distribution or in meaning. This holds especially for the features marked *assumed*. The domain shift remains, but it is a shift in distribution, not in schema. The paper's source was account-based, with a schema gap (§9.1). We do **not** manufacture a banking-style feature mismatch to make DA look necessary.

## 8. Detector model: HRG-GAT encoder with DGDA-lite heads

The architecture below is instantiated **twice, with no shared parameters**:
- **detector-T** is the acting inspector in the simulator, and works on the T view. It is fine-tuned for T.
- **detector-W** is a passive wallet model, and works on the W view. It is fine-tuned for W.

Each one is used in the simulator, for zero-shot transfer and for fine-tuning on its own task only.

### 8.1 Hierarchical relationship graph (HRG; G3, G4)

- **Levels:**
  - `L = 1` for **tx nodes**, the micro view.
  - `L = 2` for **wallet nodes**, the macro view: actors that aggregate many transactions.

  The paper's levels are unspecified (G3). This hierarchy exists natively in Elliptic++, and it gives Fig. 3's micro and macro observation scopes.
- **Edges:**
  - tx→tx flow, level pair (1,1);
  - wallet–tx `in` and `out` links, level pair (1,2)/(2,1), with the link type as an edge feature;
  - AddrAddr, level pair (2,2), only in the W view. The T view leaves it out, because T's causal constraint would need a per-edge time filter, and wallet–tx links already connect the wallets.
- **Level-pair strength** (Eq. 5): `s_ij = softplus(F[L_i, L_j])`, where `F` is a learnable 2×2 table. This is our `f`.
- **Influence** (Eq. 9), with the sign fixed (G4): `I_ij = α^{|L_i − L_j|} · s_ij`, where `α = 0.5` (ours).
- `l_ij` (Eq. 4) is not used. The paper never uses it either (G3).
- **Edge state** (Eq. 11): `e_ij = ψ(t_ij, r_ij) = MLP([edge type; log1p(multiplicity)])`. Here `r_ij` is the multiplicity term.
- **Node state** (Eq. 10): `s_i = φ(a_i, f_i, h_i) = Linear_type(shared features) + adapter`, with one projection per node type. Amounts are `a_i` (BTC features), frequencies are `f_i` (count features), and history is `h_i` (lifetime or causal aggregates).
- **Views:**
  - The **T view** of a target tx at time step t is its sampled 2-hop HRG neighbourhood, restricted to txs with time step ≤ t, with causal wallet nodes (§5).
  - The **W view** of a target wallet is its sampled 2-hop neighbourhood over all links, with lifetime wallet features.
  - Sampling caps neighbours per node at 10 per edge type for hop 1 and 5 for hop 2, and the total at 512 nodes. It is seeded per target node.

### 8.2 HRGATConv (Eqs. 2, 19)

- This is a custom PyG `MessagePassing` layer.
- **Logits:** `e_ij = LeakyReLU(aᵀ[W h_i ‖ W h_j ‖ W_e e_ij]) + log I_ij`.
- **Weights:** `α_ij = softmax_j(e_ij)`, so the attention is proportional to `exp(...) · I_ij`.
- **Update:** `h'_i = σ(Σ_j α_ij W h_j)`, with ELU as the activation.
- **Size:** 2 layers, 4 heads × 64 = 256 (ours), with dropout 0.5 on features and attention (paper).
- **Target marker:** a learned embedding of `{target, hop1, hop2}` is added to the input.
- **Readout:** `O_local` (Eq. 7) is the target node's output embedding. `O_global` (Eq. 6) is the mean-pool over all nodes in the sampled view. The heads see `[O_local ‖ O_global]` (512 dims). This is our concrete reading of Eqs. 6–8 (G3).

### 8.3 DGDA-lite heads, i.e. what L_DA means (G5, G6)

From the 512-d readout, three Gaussian heads output `(μ, log σ²)` for `Z_s` (256), `Z_d` (64) and `Z_n` (128) (paper, Table 1), with reparameterized samples during training. Each detector has its own heads, and one label head on `Z_s`: an MLP 256→128→1. The losses, all **minimized**, are:

| Term | Definition | In Eq. 20? |
|---|---|---|
| `L_y` | BCE(`p(y \| Z_s)`, y) | yes (sign fixed: minimize NLL) |
| `L_kl_d` | `β · KL(q(Z_d) ‖ N(μ_dom, I))`, with `β = 0.01` and learnable domain means `μ_src`, `μ_tgt` | yes (paper has the prior `p(Z_d)`; domain-conditioned prior is ours) |
| `L_dom` | BCE(domain head(`Z_d`), domain), with the domain head an MLP 64→1 | **no, added (ours)** |
| `L_kl_sn` | `β · [KL(q(Z_s) ‖ N(0,I)) + KL(q(Z_n) ‖ N(0,I))]` | **no, added (ours)** |
| `L_rec` | MSE(decoder(`[Z_s ‖ Z_d ‖ Z_n]`), the target's standardized shared features) | **no, added (ours)**. The paper's text says "reconstruct" (G6). |

Define `L_aux = L_kl_d + L_dom + L_kl_sn + L_rec`, and `L_DA = L_y^src + L_aux`.
- `L_y^src` is `L_y` on simulator nodes, using simulator labels.
- `L_y^tgt` is `L_y` on labelled real nodes of the task being fine-tuned. It is **not** part of `L_DA`. It appears once, in the fine-tuning objective (§11).
- `L_aux` uses simulator nodes and unlabelled real nodes from the **fitting split** of the detector's own task (T: ts 1–29). Val and test nodes are never used.

Why we add three terms:
- As written, Eq. 20 gives `Z_d` and `Z_n` no learning signal except a KL toward the prior. They would collapse onto the prior, and the "disentanglement" would do nothing.
- `L_dom` and `L_rec` are the minimum needed to make `Z_d` carry domain information and to keep the latents informative, as in DGDA.

### 8.4 Adversarial term (Eq. 21; G5, G7)

`L_adv` is the label NLL on perturbed inputs. Fixing the sign, δ **maximizes** the NLL, and θ **minimizes** the NLL at the perturbed input. The perturbation set Δ (ours) has two parts:
1. A one-step FGSM on the standardized shared and adapter inputs, with ε = 0.1 under L∞.
2. Random feature masking: each node feature is zeroed with probability `p_miss` = 0.1. This is the paper's "mimic missing values".

The graph structure is never perturbed.

**λ settings (ours):**
- **In the MARL loop:** `λ_MARL = 0.1`, preset for both detectors and chosen without any real labels. Averaging validation-selected values across tasks would carry W's labels, which encode T-test labels (E6), into detector-T.
- **For fine-tuning:** `λ` is chosen from {0.1, 0.5} by val illicit-F1 of the T0 arm, at the 25 % fraction with seed 0, **separately for each task**. It is then frozen for every arm of that task.

**Timing:**
- In the simulator, the **mask** is drawn once per episode at reset, as in Algorithm 1 line 5. The mask needs no labels. Bit `(node, feature)` is set when `hash(episode_seed, node_id, feature) < p_miss`, so nodes created later in the episode are covered by the same draw.
- **FGSM** needs labelled nodes, and labels only exist once txs execute. So FGSM is computed per detector update batch, on the masked inputs. This is a **deviation from Algorithm 1 line 5** (ours).
- The detector acts on clean features. `L_adv` is the NLL on the masked-plus-FGSM copy, which is how we read line 8's "both s and adversarial s".
- During fine-tuning, the mask and FGSM are both drawn per batch.

## 9. Simulator: SISTS-lite (UTXO)

### 9.1 Base simulator (G11)

**What the paper used (most likely).** The paper says its simulator is "built upon AML sim" and cites Altman et al. (AMLworld). But its Table 3 lists the source features `CUSTOMER_ID`, `INIT_BALANCE`, `COUNTRY` and `SENDER_ACCOUNT_ID`. Those are IBM AMLSim's output columns: they are in a historical revision of the AMLSim wiki's data schema, though not in the current one. None of them is an AMLworld Kaggle column. So the base was most likely IBM AMLSim, an account-based simulator. How the authors adapted it is not reported, and is unverifiable.

**What we build.** We implement an **independent, step-able UTXO-lite simulator** in Python. It borrows AMLSim-inspired transaction templates, and uses no AMLSim engine code. This is a **method replication**, not a recovered implementation. The templates come from two places:
- the 8 laundering typologies: `fan_in`, `fan_out`, `cycle`, `bipartite`, `stack`, `random`, `scatter_gather` and `gather_scatter`. AMLSim introduced them, and AMLworld (Altman et al., §3.2) generates the same 8;
- the normal models: `single`, `fan_out`, `fan_in`, `forward`, `mutual` and `periodical`.

We also add `peel` (a peel chain, the common Bitcoin pattern) and `direct`. AMLSim's `fan_in` is merged into `scatter_gather` for launderers (§9.3).

**Why not run AMLSim or AMLworld.**
- AMLSim (Java 8 and MASON, Python 3.7, networkx 1.11; last commit 2022-02) does not natively expose the UTXO representation or the policy-control interface we need. Its MASON loop does step internally, so adapting it is possible, but costly. Its account-transfer logs also cannot uniquely reconstruct UTXO spends, change outputs, fees or multi-input txs, so converting them would need a second generator.
- AMLworld publishes datasets, not generator code (its §7). Its banking world of banks, currencies and payment formats has no Bitcoin counterpart. We use it as a design reference only.

### 9.2 Ledger, time and worlds (G8)

**UTXO-lite ledger:**
- Each address holds a list of unspent outputs `(funding tid, amount)`.
- A tx picks one or more input addresses and spends **all** of their outputs. It creates payment outputs, plus an optional change output.
- The change goes to a fresh address of the sender with probability `p_fresh`, or else back to an input address. Going back creates the AddrAddr self-loops. The real rate is 2.0 % of rows in ts 1–29 (1.6 % over all 49 steps).
- **Fee** = `feerate × size`, with `size = 10 + 148·n_in + 34·n_out` bytes and `feerate` lognormal per world. `fees = in_BTC_total − out_BTC_total` and `total_BTC = out_BTC_total` hold by construction, matching the §2 audit.
- **tx→tx flow edge** `a → b` if `b` spends an output of `a` **and both are in the same time step**, as in Elliptic++, where no edge crosses a time step.

**Time:**
- An **episode is one time step**, matching Elliptic++'s 49 per-step graphs. It has `T = 50` sub-steps. At 1,975 blocks per time step, that is about 39.5 blocks per sub-step.
- A **world is 10 consecutive episodes**. Entities, addresses and balances persist across a world's episodes, so wallet lifetimes span time steps as in the real data.
- An **epoch is one world** (ours). So 1,000 episodes (paper, MAX_EPISODES) is 100 worlds, which is 100 epochs, matching Figs. 4–5.
- `done` is the end of the time step. A world ends after its 10th episode.
- The world's 10-step window sits at a random offset in [0, 39] time steps, so that block-height features cover the real range (E19).

**Export per world.**
1. **Prune every time step first.** Keep only the largest weakly connected component of each time step's tx graph, with its address links. That mirrors Elliptic++'s one-component-per-step construction (§2).
2. **Then recompute, over the retained txs only:**
   - wallet labels (the E6 rule);
   - every wallet feature and causal count;
   - every graph-derived feature (degrees, `self_loop_frac`, AddrAddr pairs).

   So no exported value describes a dropped tx. In the real data, wallet features also count only the txs in the dataset (§2: `total_txs` matches 100 %).

The detector acts on all txs in the live simulation. Exports feed detector-W's training (§10), transfer replay, JS and reporting.

### 9.3 Entities, roles, actions and rewards (Eqs. 13–17; G7, G9, G16)

Parameters are **shared within each role** (ours). Algorithm 1's per-agent `θ_i` would mean thousands of networks.

Each agent observes its 2-hop neighbourhood in an **agent-view HRG**: the T-view construction, plus one controller node per agent (a level-2 node linked to the addresses it controls). The observation also includes the world's global readout. A controller node is visible only to its own agent's policy, **never to the detector**.

**Services** are exchanges and other licit hubs, 20–60 per world. They are **scripted**: they consolidate deposits (fan-in) and pay batched withdrawals (fan-out), which creates the heavy-tailed degrees. In ts 1–29, wallet link-degree has median 1, 99th percentile 6 and max 1,025.

| Role | Count per world | Action `[...]` (Eqs. 16–17) | Reward (ours, magnitudes) |
|---|---|---|---|
| **Launderer** | `n_L` (multiverse). Each has 1 illicit source address holding the proceeds, plus a mule pool that grows on demand. | `initiate` ∈ {0, 1}, where 0 means wait. `amount` ∈ {10, 25, 50, 100} % of the source balance. `recipient` = a pointer over K = 16 **service** deposit addresses: the tranche's cash-out target. `strategy` ∈ 9 route templates (below), each ≤ 6 hops, executed one hop per sub-step and ending at the recipient. `mules` ∈ {fresh, reuse} for the route's intermediate addresses. `change` ∈ {fresh, reuse}. | Per sub-step, `(BTC cashed out at a service undetected − BTC newly frozen) / D_w`: `+r_success` and `−r_detected` measured in BTC, over the world's mean proceeds per launderer and episode `D_w` (below). `−0.001` per sub-step while funds remain. |
| **Innocent** | 3k–6k individuals per world, each with 1–5 addresses. A scripted demand process makes about 2 % active per sub-step, each with a fixed recipient and amount. | `initiate` ∈ {now, defer}. `amount split` ∈ {1, 2, 3}. `route` ∈ {direct, forward via own second address, via service}. `change` ∈ {fresh, reuse}. The recipient is fixed by demand. | `+r_legit = 0.1` per executed tx. A **flagged (false-positive) tx is blocked for that sub-step** and earns no `r_legit`. This is the indirect false-positive cost that the paper leaves unspecified (G16). |
| **Detector** | 1 | `select_tx`: flag the top-k of this sub-step's new txs by `Q_flag − Q_skip`. `analysis_depth` ∈ {1, 2} GAT hops (deeper messages are masked). `update_strategy` ∈ {−, 0, +}, which multiplies the budget fraction `b` by {0.8, 1, 1.25} within [1 %, 20 %]. | `+r_detect = 1.0` for each flagged illicit tx. `−r_miss = 1.0` for each unflagged illicit tx that pays a service (cash-out), and 0 for intermediate hops, which can still be caught later. `−r_false = 0.1` for each flagged licit tx. `−0.01 × depth` per investigated tx. |

**Launderer proceeds and reward (ours; revised after M5's first runs).**
- At world creation, each launderer's proceeds for the world's 10 episodes, `p_{i,e} ∼ lognormal(μ, σ)` (§9.7), are drawn on their own RNG stream. Each is credited to the source at the start of its episode. Future proceeds are in no observation.
- `P_w = Σ_{i,e} p_{i,e}`, and `D_w = P_w / (10 n_L)` is the world's mean proceeds per launderer and episode. `D_w` is fixed for the world and shared by its launderers.
- Frozen BTC counts every output of a flagged launderer tx, a setup's change to the source included: a frozen tx freezes all its outputs (step 6 below).
- Without discounting and the holding cost, the launderers' mean episode reward is then exactly the world's **BTC utility**, `(cashed − frozen) / P_w`. A fixed amount earns the same whether it moves in one tranche or in ten.
- M5's first runs (2026-11-10) used `(cashed − frozen tranche) / tranche` per hop instead. That pays up to +1 per tranche whatever its size, which rewards splitting, and it ignores frozen change. It stays logged as a diagnostic.
- Neither form is Eq. 14 verbatim: the paper reports no magnitudes.

**Action execution (feasibility masks `M_b` and templates):**
- **Branch sets depend only on state.** For each role, the set of active branches `B(s)` and the per-option masks `M_b(s)` are functions of the state, never of the current action. This is what §10's target needs.
  - **Launderer with no tranche in flight, and funds at the source:** `B(s) = {initiate, amount, recipient, strategy, mules, change}`. All six are chosen together in one sub-step.
  - **Launderer with a tranche in flight:** `B(s) = {initiate}`, meaning proceed with the next hop now, or wait. The env executes the template's next hop.
  - **Launderer with no funds left:** no transition is stored.
  - `M_b(s)` masks options that are infeasible in `s`: empty recipient slots, amounts whose payment would fall below 1e-5 BTC, and `mules = reuse` when the pool has fewer than k mules.
- **Realized branches.** `R(s, a) ⊆ B(s)` is the set of branches whose choice took effect, and only these enter §10's loss.
  - If `initiate = 0`, then `R = {initiate}`.
  - At `amount = 100 %`, `change` is dropped, because there is no change output.
  - For `direct`, `mules` is dropped.
- **A tranche** is funded only from the illicit source. `amount` applies to `source balance − fee`, with the fee taken first.
- **Recipient** = one of K = 16 **service deposit addresses**, sampled uniformly from the world's services every sub-step. It is the tranche's cash-out target, and every route ends there. Intermediate addresses are never chosen by the pointer. They come from the `mules` branch: `fresh` creates new mule addresses, and `reuse` draws k from the pool uniformly.
- **Hops.** One hop is one level of txs, all executed in the same sub-step. A hop spends all outputs held by the addresses it uses. `k ∼ U{2..5}` is drawn by the env, and is not an action.
- **Terminal hop.** Every address still holding tranche funds pays the recipient in its own tx: parallel cash-outs.
- **Templates.** Templates that need k funded inputs start with a **setup scatter** from the source.

  | Strategy | Hops (setup + body + terminal), ≤ 6 |
  |---|---|
  | `direct` | source → recipient (1) |
  | `fan_out` | source → k mules (1 tx, k outputs); terminal: k cash-outs (2) |
  | `scatter_gather` | source → k mules; the k mules → 1 mule (1 tx, k inputs); terminal (3). With a single source, this also covers AMLSim's `fan_in`, which is therefore not a separate strategy. |
  | `gather_scatter` | setup scatter → k mules; gather → 1 mule; scatter → k fresh outputs; terminal (4) |
  | `stack` | source → k mules; each mule → 1 further mule (k parallel 1-to-1 txs); terminal (3) |
  | `bipartite` | setup scatter → k mules; 1 tx with k inputs and k outputs; terminal (3) |
  | `cycle` | source → m₁ → m₂ → m₁ (revisits m₁, reusing the address); terminal (4) |
  | `peel` | a chain of 2–5 hops, each paying 5–20 % to the recipient and the rest to a fresh mule; terminal remainder (≤ 6) |
  | `random` | a random walk of 2–5 hops over mules from the `mules` branch; terminal (≤ 6) |

  This narrows AMLSim's list: `fan_in` is merged into `scatter_gather`, and waiting is `initiate = 0`.
- **Innocents.** With a pending demand, `B(s) = {initiate, split, route, change}`. Otherwise no transition is stored. `split = k` gives k outputs to the fixed recipient. `forward` is 2 txs through the innocent's own second address. `via service` is a deposit, then a scripted service payout to the recipient. `change` is dropped from `R` when the tx has no change output.
- **Detector.** The global branches `B(s) = {analysis_depth, update_strategy}` are always active. `select_tx` flags are separate terminal transitions (§10).

**Sub-step order (atomic):**
1. **Snapshot.** Every agent observes the committed ledger state `s_t`.
2. **Propose.** Every agent proposes its actions.
3. **Validate and reserve.** Proposals are processed in a seeded random order. Each valid tx reserves its input outputs. A proposal whose inputs are already reserved, or are insufficient, becomes a canonical **no-op**: it is stored with the realized action and gets reward 0.
4. **Score.** The tentative txs form the candidate set. The detector scores them on `s_t` plus the tentative txs, with each tx's features computed as if committed.
5. **Flag.** The detector flags the top-k.
6. **Commit or revert.**
   - Unflagged txs commit.
   - Flagged illicit txs commit with **all** their outputs **frozen**, change included, so the funds are lost to the launderer.
   - Flagged licit txs are **reverted**: their inputs are released, and the demand stays pending and retries later.
7. **Reward**, then `s_{t+1}` = the committed ledger.

Reverted txs never enter the ledger, the graphs, the exports or the labels. They exist only in the detector's flag transition (`−r_false`), and they are excluded from `L_y^src`.

The SRF ablation (optional) replaces every role's reward with `+r_uniform = 0.1` for any executed action and `−0.1` for an undesired outcome (paper, unnumbered equation).

### 9.4 Simulator labels (mirroring Elliptic++'s labelling; E2, E6)

- **Ground truth:** a tx is illicit if a launderer initiates it. Otherwise it is licit-by-truth.
- **Observed labels**, mimicking Elliptic++'s partial labelling:
  - Every illicit tx is labelled illicit with probability `p_ill`.
  - Licit txs are labelled licit when a service initiates them, and otherwise with probability `p_lic`.
  - Everything else is `unknown`.

  `p_ill` and `p_lic` are calibrated to the public README totals: 22.9 % of txs labelled, of which 9.8 % are illicit. Those are public statistics, not our labels.
- **Wallet labels** follow the audited real rule (E6):
  - illicit if linked to an illicit-labelled tx;
  - licit if linked to a licit-labelled tx and not illicit, except that a fraction `p_wunk` of the wallets matching this rule is set to unknown. `p_wunk` is **solved from the README totals**, after `p_ill` and `p_lic`: it is the value that brings the simulator's labelled-wallet fraction to the 32.2 % target below, clipped to [0, 0.5]. If the target cannot be reached, the shortfall goes in the calibration report. We do **not** use the audited real figure (24.1 %), because it is computed from per-wallet labels over the whole dataset, test split included;
  - otherwise unknown.

  Targets: 32.2 % of wallets labelled, of which 5.4 % are illicit.
- Detector training in the simulator uses **observed** labels only, as on real data. Unknown nodes stay in the graph. Ground truth is used only for rewards and reporting.
- **Shortcut guard:** the simulator's label masking must not depend on any feature except through the rules above. A test fits a logistic regression on the shared features to predict "labelled vs unknown" among licit-by-truth txs that innocents initiate. Its ROC-AUC must be ≤ 0.6.

### 9.5 Multiverse (G1, the paper's "diverse virtual worlds")

Each world draws its parameters from:
- launderer count `n_L` ∈ {10, 25, 50};
- service count ∈ [20, 60];
- initial detector budget `b` ∈ {2 %, 5 %, 10 %};
- innocent demand rate ∈ [1 %, 3 %];
- `p_fresh` ∈ [0.5, 0.95];
- feerate lognormal parameters;
- feature mask rate ∈ [0, 0.3];
- the window offset ∈ [0, 39];
- ±10 % jitter on every calibrated generator parameter.

This is domain randomization over plausible worlds. The default world is the calibrated centre.

### 9.6 Calibration to Elliptic++ (label-blind, fitting splits only)

**Objective.** Run a random search of about 50 configurations over the §9.7 schema, on CPU, with no learning. It minimizes the mean JS (§7) between the static-mode simulator and **T-safe real data only** (ts 1–29, pooled over classes and unknown, label-blind):
- the tx shared features;
- causal wallet counts at t = 29;
- per-step tx counts;
- the wallet–tx link-degree distribution, using only links to txs in ts 1–29.

Lifetime wallet features are **not** calibrated against: they would carry T-test information into the simulator that detector-T trains in (§5, T isolation). W-view fidelity is therefore a measured outcome (Q1), not a fitted one.

The labelled fractions use only the README totals. The result is frozen as the multiverse centre, and the calibration report is saved. The centre and the §9.4 masking rules are **the same for every arm** (Full, NMAI and NDA). They are never re-tuned on per-class JS or on any downstream validation score. Launderer behaviour is **not** calibrated. It comes from what the launderers learn, and it is what Q1 measures.

### 9.7 Generator parameter schema and initialization

**Initialization (pre-world funding).** At world start, each address's starting balance is a single output of a synthetic **genesis funding tx**. Genesis txs sit outside the world: they are never exported, never a tx node or a flow edge, and never counted in any feature. Real Elliptic++ likewise counts only in-dataset parents (`in_txs_degree`).

Calibration (§9.6) searches within these bounds, all ours. Every real anchor below, and every data-informed bound, is computed on **ts 1–29 only** (T-safe; `audit11.py`).

| Parameter | Search bounds | Real anchor |
|---|---|---|
| Innocents per world | 3,000–6,000 | per-step tx count: median 4,291, range 1,089–7,880 |
| Addresses per innocent | geometric, mean 1–5 | — |
| Innocent starting balance | lognormal BTC, μ ∈ [−3, 1], σ ∈ [1, 3] | `in_BTC_total`: median 0.702, 5th–95th percentile 0.0112–163 |
| Demand amount | fraction of balance ∼ Beta(a, b), a, b ∈ [0.5, 5] | `in_BTC_total`, `out_BTC_*` |
| Demand recipient | innocent with probability `p_ii` ∈ [0.5, 0.9], chosen by preferential attachment on degree with exponent ∈ [0.5, 1.5]; otherwise a service | link-degree distribution |
| Demand rate | 1–3 % of innocents per sub-step | per-step tx count |
| Services per world | 20–60 | wallet link-degree: p99 6, max 1,025 |
| Service payout batch size | 2–100 outputs | `num_output_addresses`: median 2, 95th percentile 9, max 13,107 |
| Service consolidation interval | 1–10 sub-steps | `num_input_addresses`: median 1, 95th percentile 6, max 656 |
| `p_fresh` | 0.5–0.95 | AddrAddr self-loop rate 2.0 % |
| Feerate | lognormal BTC/byte, μ ∈ [−15.0, −14.0], σ ∈ [0.4, 1.0] | `fees / size`: median 5.24e-7 (ln = −14.46), 5th–95th percentile 1.71e-7 to 1.56e-6 |
| Launderer proceeds | lognormal BTC, μ ∈ [−1, 3], σ ∈ [0.5, 1.5] | not calibrated; launderers are learned |

The `size = 10 + 148·n_in + 34·n_out` formula matches the real median `size` within 0.2 %.

## 10. MARL learner: Algorithm 1 made implementable (G7)

**Learner choice.** We implement Algorithm 1's **per-role Q-learning with a TD loss plus `L_DA + λ L_adv`**, and not full MADDPG. The paper's equations and pseudocode describe Q-learning. MADDPG needs continuous or Gumbel-softmax actors, and nothing in the paper specifies them. The additions below fill gaps Algorithm 1 leaves open (G7), and each is labelled:

| Gap in Algorithm 1 | Our fill |
|---|---|
| `max_{a'}` over a structured action | **BDQ, the branching dueling Q-network** (Tavakoli et al. 2018), with per-branch values: `Q_b(s, a_b) = V(s) + A_b(s, a_b) − mean_{a ∈ M_b(s)} A_b(s, a)`, for each branch `b ∈ B(s)`. `B(s)` and `M_b(s)` depend on the state only (§9.3). The pointer branch's `A_b` scores candidates by attention. |
| ENTROPY_COEF 0.01 matches no equation | It becomes the **soft-Q temperature** `τ = 0.01`. One shared target per transition: `y = r + γ^Δ (1 − d) · F(s')`, where `s'` is the agent's next decision point and `Δ` the number of sub-steps to it, so γ is per sub-step. For **launderers and innocents**, `F(s') = τ · logsumexp_{a ∈ M_initiate(s')} (Q_initiate^targ(s', a) / τ)`, the **root target** (ours). For the inspector's **global branches**, `F(s') = (1/|B(s')|) Σ_{b ∈ B(s')} τ · logsumexp_{a ∈ M_b(s')} (Q_b^targ(s', a) / τ)`. Both depend on `s'` alone, so no next-step action is needed. The loss is `(1/|R(s,a)|) Σ_{b ∈ R(s,a)} (y − Q_b(s, a_b))²`, the BDQ form with mean aggregation over the branches whose choice took effect (§9.3). The behaviour policy is softmax(Q_b / τ) per branch over `M_b`, combined with ε-greedy that decays from 1.0 to 0.05 over the first 300 episodes. As τ → 0 this is Algorithm 1's max. |

**Why the root target (ours; revised after M5's first runs).** `initiate` is in `B(s)` and realized under every action, so it is the only branch that every transition trains. The other branches rank a launch's options and stay out of the bootstrap. The first version of this SPEC bootstrapped launderers and innocents through the mean over all of `B(s')`. But when an agent waits only `initiate` is realized, so five branches trained only on launches carried most of every idle state's target. In M5's first runs the launderer values inflated: `Q(wait)` reached 6.5–357, where no return above a few units was attainable (reports/m5_marl.md). This conditional BDQ variant is ours and has no general convergence guarantee. Those runs also discounted by γ per decision, so a 50-sub-step gap cost one γ; γ is now per sub-step.
| `done` | `d = 1` only at **world end**. Time-step (episode) boundaries are bootstrapped through, because funds and addresses persist within a world. A detector `select_tx` flag transition is always terminal: `d = 1`, so `y = r`. |
| Batch 128 but no replay buffer | A replay buffer per role holds 100k transitions of agent-view observations (capped at 256 nodes), plus a 1,000-transition warm-up. |
| No target network | A Polyak-averaged target network with rate 0.005. |
| GRADIENT_CLIP 1.0 | Global-norm clip of 1.0 on each role's update. |

**Which losses each role uses:**
- Launderers and innocents use the TD loss only. They have no labels and nothing transfers from them.
- **detector-T** (it acts) uses `TD + L_DA + λ_MARL L_adv`, with `L_DA = L_y^src + L_aux` (§8.3).
  - `L_y^src` uses committed simulator txs with observed labels, in the T view.
  - `L_aux` also uses *unlabelled* real T-view txs from **ts 1–29 only**.
  - Each update draws 128 simulator txs and 128 real txs.
- **detector-W** (passive, no actions, no TD) uses `L_DA + λ_MARL L_adv`.
  - It trains after each world ends, on that world's pruned and recomputed export (§9.2), in the W view, with observed wallet labels.
  - `L_aux` uses unlabelled real W-fitting-split wallets.
  - It runs the same number of updates as detector-T.
- `L_adv` is computed on simulator nodes for both detectors. No real batch crosses tasks.
- **Real labels are never used inside the MARL loop.** So one MARL run serves both tasks and every label fraction (ours). It also matches the paper's text, "pre-trained on synthetic, fine-tuned on real". Eq. 20's `Y_t` is read as "the labels of the domain being classified".

**Transitions:**
- Each per-tx `select_tx` flag decision is a one-step, terminal transition with `y = r`. A flag's outcome is known in the same sub-step.
- The detector's global branches (`analysis_depth`, `update_strategy`) use the sub-step's summed detector reward, and bootstrap from the next sub-step's global readout.
- Launderer transitions are per launderer per sub-step, observed from its controller node. Innocent transitions exist only on sub-steps when the innocent has a demand.

**Update schedule:** one gradient step per role per sub-step, on a batch of 128. Adam with lr 1e-3, γ = 0.99. Checkpoint every 25 episodes, including the full RNG state.

**Health checks, logged every epoch (they replace Figs. 4–5):**
- per-role mean episode reward;
- detection rate (recall on illicit cash-out txs);
- the **illicit/licit tx ratio** among executed txs per episode. This is our definition of Fig. 5's undefined ratio (G15);
- where the launderers' proceeds went, as fractions of them: cashed out undetected (the **success fraction**), frozen (change included), paid in fees, and still held. They close, `proceeds = cashed + frozen + fees + held`, and the gap is logged every epoch. **BTC utility** is the success fraction minus the frozen fraction;
- each strategy's usage share.

A run is **degenerate** if, over the last 20 epochs, either the detection rate or the launderer success fraction stays above 0.95 or below 0.05. A degenerate run is reported as such, as a warning. It no longer decides acceptance on its own.

**Acceptance of a MARL run (ours; replaces the degenerate rule as the M5 gate).** M5's first runs passed the degenerate rule by the letter while their launderers had not learned: against their final inspectors, a scripted `direct` policy cashed out most of its proceeds, while the learned launderers chose templates that were almost always caught (reports/m5_marl.md, reports/m5_capability.md). A run passes if:
1. **Complete:** 1,000 episodes, every loss finite, and the BTC accounting closes every epoch (gap ≤ 1e-6 of proceeds).
2. **Calibrated values:** on held-out worlds at ε = 0, the mean of `Q_initiate(s, a) − G` over launderer decisions is within `max(0.5, 0.25 · mean |G|)`. `G` is the realized return from the decision, with γ per sub-step, to world end.
3. **Adapted launderers:** the final launderers play 8 held-out multiverse worlds, the same for every policy compared, against the run's final inspector and innocents frozen at ε = 0.05. At **matched launch opportunities**, each launderer launches once per episode, at a sub-step drawn uniformly from [0, 43], and the network chooses every option. There the final launderers must beat (a) uniform option choices and (b) the run's own launderer snapshot at world 30, where ε reaches its floor, each by at least 0.02 in BTC utility. The paired 95 % t-interval over the worlds of all 3 seeds must lie above 0, and the mean difference must be ≥ 0 in at least 2 of the 3 seeds.

Reported, not gated:
- the natural policy (the network also decides when to launch) against wait-only and the static (NMAI) launderers;
- launch, proceed and detection rates;
- the health curves.

**Calling it co-evolution.** Adapting to a frozen inspector does not show feedback. A passing run is called co-evolved only if two checks on the same held-out worlds both give paired 95 % intervals above 0, pooled over seeds:
- (a) the inspector responded: the final inspector freezes a larger share of the final launderers' proceeds than the world-30 inspector does;
- (b) the launderers responded: against the final inspector, the final launderers earn more BTC utility than a **no-feedback control**, which is launderers trained from the world-30 snapshot for the same number of episodes against the world-30 inspector, frozen.

Otherwise the report says "adaptation shown, co-evolution not identified".

## 11. Transfer protocol

1. **Pretrain.**
   - **Full:** the MARL run's final detector-T checkpoint (for T) and detector-W checkpoint (for W).
   - **NMAI:** the same detector, trained for the same number of detector updates. It uses the same losses, including in-loop DA, but on data from the **static** simulator. That simulator has the same worlds, but launderers draw typologies uniformly and innocents use scripted routes, so nobody learns.
   - **NDA:** a MARL run identical to Full, but with `L_DA` reduced to `L_y^src` (so `L_aux` is removed) and no real batches.
2. **Export.** Run the final policies (or the static rules, for NMAI) on 100 multiverse worlds, which is 1,000 time-step graphs, matching the paper's 1,000 "graphs". Store the per-step components (§9.2). Then compute Q1's JS and our version of Table 2: nodes, edges, graphs, average degree reported as both E/N and 2E/N (G10), labelled fractions and class rates.
3. **Fine-tune, separately for T and W.**
   - Start from that task's detector (detector-T or detector-W), and keep its encoder and label head.
   - Add the zero-init target adapter (T only).
   - Train on the label fraction `f` of the task's fitting set.
   - Loss:
     - Full and NMAI: `L_ft = L_y^tgt + L_y^src + L_aux + λ L_adv`, with every term at weight 1 (ours). The target label loss is counted once. `L_y^src` and the source half of `L_aux` replay that arm's exported simulator nodes of the matching node type. `L_adv` is computed on the target batch.
     - NDA: `L_ft = L_y^tgt + L_y^src + λ L_adv`. This is the same source-label replay as Full, with only `L_aux` removed.
     - Target-only: `L_ft = L_y^tgt + λ L_adv`.
   - Follow the §6 protocol.
4. **Zero-shot.** Evaluate the pretrained Full and NMAI detectors, with the shared channel only and no adapter, on each task's test, before any fine-tuning.

## 12. Experiment matrix (reduced; every row is run for both T and W)

| Arm | Pretrain | Fine-tune loss | Fractions | Seeds | Answers |
|---|---|---|---|---|---|
| S0 RF anchor | none | — | 1.0 | 3 | pipeline check against published notebook RF (§6) |
| **T0 target-only** | none (random init) | `L_y^tgt + λL_adv` | 0.05 / 0.25 / 1.0 | 3 | the paper's missing "real-only" arm (digest, Evaluation protocol) |
| T0-shared (optional, T only) | none | `L_y^tgt + λL_adv`, without the 165 anonymized features | 1.0 | 3 | how much the anonymized features are worth |
| **Full** | SISTS MARL + in-loop DA | `L_y^tgt + L_DA + λL_adv` | 0.05 / 0.25 / 1.0 | 3 | Q2 (vs T0) |
| **NMAI** | static simulator + in-loop DA | `L_y^tgt + L_DA + λL_adv` | 0.05 / 0.25 / 1.0 | 3 | Q3 co-evolution (vs Full), Q1 |
| **NDA** | SISTS MARL, no `L_aux` | `L_y^tgt + L_y^src + λL_adv` | 0.05 / 0.25 / 1.0 | 3 | Q3 whole-pipeline DA effect (vs Full) |
| Zero-shot | Full, NMAI | — | 0 | 3 | sim realism |
| Q4 noise (optional) | the final models above | evaluation only: mask a fraction ∈ {0.2, …, 0.7} of test features | — | 3 | Fig. 7 analogue |
| SRF (optional) | MARL with SRF rewards | as Full | 0.25 | 1 | the paper's strongest ablation |
| W-temporal (optional) | T0, Full | as above, on a W split by first appearance (train: first ts ≤ 34) | 1.0 | 3 | robustness to the lexicographic W split. Caveat: lifetime features still span the boundary for 10,471 wallets, 2,499 of them labelled (E5). |

The pretraining runs are 3 Full, 3 NDA and 3 NMAI, one per seed. Each run produces both detector-T and detector-W, which share no parameters (§8).

**What Full − NDA means here.** NDA drops `L_aux` and the in-loop target batches. It keeps the target-informed calibration, the shared normalization and source replay. So Full − NDA measures the **coupled pipeline effect**: feature alignment, the regularization that `L_aux` adds, and the simulator trajectories that change because the detector changed. It does not isolate feature alignment. And because sim and real share one schema (§7), it does not test the heterogeneous-schema alignment that the paper's AMLSim source needed. Report it with this caveat.

**What Full − NMAI means here (ours; added after M5's first runs).**
- Full, NDA and NMAI share every simulator mechanic: the generator, the proceeds schedules, the reward and accounting definitions, the label rules, the world draws and the number of detector updates.
- In NMAI the inspector acts and trains exactly as in Full. Only the transaction agents do not learn: launderers draw typologies uniformly, and innocents use scripted routes.
- Full also learns its innocents, so Full − NMAI measures the effect of the learned transaction agents as a whole. Attribute it to launderer–inspector feedback only if a control with scripted innocents in Full supports that, and only if the run passes §10's co-evolution check.

**Deliverable:** `reports/results.md`, containing:
- the main tables, one per task (illicit F1, PR-AUC, ROC-AUC, P, R, R@1 %, and the comparison-only metrics; mean ± sd);
- the Δ-vs-T0 bootstrap CIs;
- label-efficiency plots;
- T's per-time-step F1;
- the JS table and Fig. 6 analogue (Full vs NMAI);
- MARL health curves;
- our Table 2 analogue;
- the S0 anchor results;
- an explicit list of every **(ours)** decision that the results depend on.

## 13. Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| MAX_EPISODES | 1000 | paper |
| Episode / world / epoch | 1 time step of 50 sub-steps / 10 episodes / 1 world | ours (G8) |
| Learning rate, optimizer | 1e-3, Adam | paper |
| Discount γ | 0.99 | paper |
| Batch size | 128 | paper |
| Dropout | 0.5 (GAT features and attention) | paper |
| Launderer / innocent Q-network dropout | PENDING: set by the M5 capability test | ours |
| Launderer / innocent target | root: bootstrap through `initiate` only, γ per sub-step (§10) | ours |
| Latent dims Z_s / Z_d / Z_n | 256 / 64 / 128 | paper |
| β (KL weight) | 0.01 | paper |
| ENTROPY_COEF | 0.01, used as the soft-Q temperature τ | paper value, our use (G7) |
| GRADIENT_CLIP | 1.0 | paper |
| λ (adversarial) | MARL: preset 0.1. Fine-tuning: {0.1, 0.5}, picked on each task's val. | ours |
| Δ | FGSM ε = 0.1 (L∞, standardized), `p_miss` = 0.1 | ours |
| HRG decay α | 0.5, exponent `|ΔL|` | ours (G4) |
| GAT | 2 layers, 4 heads × 64, ELU | ours |
| Replay buffer, warm-up, target-network rate | 100k, 1k, Polyak 0.005 | ours |
| ε-greedy | 1.0 → 0.05 over 300 episodes | ours |
| Neighbourhood sampling | 10 / 5 per edge type for hops 1 / 2, cap 512 nodes | ours |
| Rewards | §9.3 table | ours (G9) |
| Fine-tune | ≤ 50 epochs, patience 5 on val illicit-F1, `pos_weight` = √(neg/pos) | ours |

## 14. Gotcha ledger

The paper's gotchas G1–G16 are from the digest. E1–E20 are new.

| Id | Gotcha | Resolution | § |
|---|---|---|---|
| G1 | Name and typo noise | Use SISTS and GDA-SpecReg. Neither affects the implementation. | — |
| G2 | Overloaded symbols | The code uses distinct names: `att_a`, `amount`, `hrg_alpha`, `lr`, `level`, `loss_*`. | 8 |
| G3 | Dangling HRG pieces | Defined: tx/wallet levels, `f`, `φ`, `ψ`, and `O_local`/`O_global` as readouts. `l_ij` is dropped. | 8.1–8.2 |
| G4 | Eq. 9's exponent is signed | Use `|L_i − L_j|`. | 8.1 |
| G5 | Signs in Eqs. 20–21 | Minimize NLL + β·KL. δ maximizes the NLL. | 8.3–8.4 |
| G6 | Eq. 20 is not a VGAE objective | Add recon, a domain head, and KL on Z_s and Z_n (each labelled as added). | 8.3 |
| G7 | "MADDPG" vs Algorithm 1 | BDQ-style branching soft-Q per role (per-branch Q, one shared target per transition: the root branch for launderers and innocents, the branch mean for the inspector's global branches; feasibility masks, world-end terminal, γ per sub-step), with replay, a target network and parameter sharing. L_DA and L_adv apply to the detectors only. | 10 |
| G8 | Episodes vs epochs | Episode = one time step, world = 10 episodes = 1 epoch. T = 50. `done` defined. | 9.2 |
| G9 | Unreported values | All set in §13 and §9.3 and marked (ours). | 13 |
| G10 | Table 2 inconsistencies | The text's "822k wallets, 1.27M temporal interactions" matches the README: 822,942 wallets, 1,268,260 wallet rows. Table 2's 859,254 nodes and 995,508 edges match nothing we can derive. Our Table 2 analogue reports E/N *and* 2E/N. | 2, 11 |
| G11 | Which simulator | The paper most likely ran IBM AMLSim: its Table 3 columns are AMLSim's historical output schema, although it cites AMLworld. We build an independent step-able UTXO-lite simulator with AMLSim-inspired templates and no AMLSim code: a method replication. AMLworld (no public code) is a design reference only. | 9.1 |
| G12 | Feature alignment undefined | Elliptic++'s named features *are* the shared channel, with definitions audited. A zero-init adapter carries the 165 anonymized tx features. Shared names do not imply shared distributions. Full − NDA is a coupled pipeline effect, not schema alignment. | 7, 12 |
| G13 | Claims beyond the evidence | 3 seeds, bootstrap CIs, and a "supported" rule. Negative results are reported. | 6 |
| G14 | Which dataset the ablations used | Every arm is evaluated on both T and W. | 12 |
| G15 | Fig. 5 ratio undefined | Defined as executed illicit txs / licit txs per episode. | 10 |
| G16 | No false-positive cost for innocents | A flagged innocent tx is blocked for that sub-step and loses `r_legit`. | 9.3 |
| E1 | Elliptic++ holds two tasks, and Elliptic v1 is inside it | T (tx, Table 4 analogue) and W (wallet, Table 5 analogue). v1 is the 165 anonymized columns. | 0, 2 |
| E2 | Imbalance and mostly-unknown labels | Illicit F1 and PR-AUC are primary. Unknown nodes stay in the graph, unlabelled. The simulator mimics partial labelling. | 6, 9.4 |
| E3 | The official splits differ by task, and W's is lexicographic | Reproduced exactly, with counts in §2. The W address-type shift is documented. W-temporal is an optional check. | 2, 5, 12 |
| E4 | Duplicate wallet rows | De-duplicate to one row per address. `Time step` is not a per-row snapshot. | 2, 5 |
| E5 | Lifetime wallet features contain future information | T uses causal wallet counts only, and is isolated end to end: calibration, normalization, in-loop DA and fine-tuning use ts 1–29 only. detector-T and detector-W share no parameters. W uses lifetime features, since its split is not temporal. W-temporal is caveated. | 5, 8, 9.6, 10 |
| E6 | Wallet label ⟺ linked to an illicit tx | Never use one task's real labels in the other, including through hyperparameters (`λ_MARL` is preset). No real multi-task training. The simulator copies the rule, with `p_wunk` solved from the README wallet totals, not from per-wallet labels. | 2, 5, 8.4, 9.4 |
| E7 | 965 txs with null augmented features, 0 of them illicit | Drop them, as the notebook does. No missing indicator, because it would be a licit shortcut. | 2, 7 |
| E8 | 165 anonymized tx features | Target adapter only. The optional leave-out arm T0-shared measures their value. | 7, 12 |
| E9 | Named-feature definitions are undocumented | A definition audit against recomputable features, plus a feature manifest that marks every unauditable definition as an assumption, with sentinels and transforms. | 2, 7 |
| E10 | The paper's metrics are not comparable | The paper's accuracy (95.2 / 94.0) is *below* the notebook RF's micro-F1 (0.980 / 0.990), while RF's illicit F1 is 0.828 / 0.845. So the paper's "F1" is probably macro or weighted, on another protocol. Report every F1 variant. Compare against the S0 anchor, never against the paper. | 6 |
| E11 | 4 GB, no-bf16 local GPU; RAM-backed `/tmp`; spot preemption | Micro-batching and fp32 locally. Scratch under `data/`. Checkpoint plus resume, and `/home`-only persistence, on JarvisLabs. | 3 |
| E12 | Python 3.14, PyG extension wheels | uv with Python 3.12, pure PyG, our own sampler, arch-list assert. | 3 |
| E13 | Data movement | Upload processed parquet only (< 1 GB). | 3 |
| E14 | Data licence | Git-ignore `dataset/`, `data/` and `runs/`. Aggregates only in reports. | 2 |
| E15 | Target labels inside DA, calibration or hyperparameters | Labels from the task's own fitting split only, only in fine-tuning, counted once. Calibration is label-blind (README totals) and T-safe. `λ_MARL` is preset. Val is used for per-task selection, test once. | 5, 8.4, 9.6, 10, 11 |
| E16 | Per-domain normalization hides the shift | Moments fitted once on the real fitting split, shared by both domains. No per-domain quantiles. | 5 |
| E17 | Non-deterministic CUDA scatter | Seed everything. Report variance across seeds instead of claiming bitwise reproducibility. | 6 |
| E18 | Dark-market shutdown at ts 43 | Report T's per-time-step F1. Do not tune on test time steps. | 6 |
| E19 | Absolute block heights in wallet features | Convert to time steps since the first dataset block. Place simulator worlds at random offsets. | 5, 9.2 |
| E20 | AddrAddr hubs, multi-edges and self-loops | Uniform sampling over distinct neighbours. Multiplicity as an edge feature. Self-loops become `self_loop_frac`. | 5, 8.1 |

## 15. Milestones, acceptance checks and compute

| M | Deliverable | Acceptance check | Where | Est. |
|---|---|---|---|---|
| M0 | uv env, `pytest` suite, arch asserts | Tests pass on both machines | local + A30 | 0.3 A30-h |
| M1 | D0–D5 + audit | `audit.json` matches §2. Split counts match §2 exactly. | local | < 1 h wall |
| M2 | S0 RF anchor | Illicit F1 within ±0.03 of 0.828 (T) and 0.845 (W). Otherwise stop and fix the pipeline. | local | — |
| M3 | T0 | Gate on **val**: T0's val illicit-F1 ≥ that of an RF refit on the *fitting split only*, minus 0.05, on both tasks. (S0 itself trains on official-train, which includes val.) Test is not looked at here. | local (1650S) or A30 | 2 A30-h |
| M4 | Simulator, static mode, calibration | Unit tests on a hand-built ledger: conservation of BTC (`in = out + fee`), the flow edge rule, the label rules (E6), the shortcut guard (§9.4), every manifest definition and sentinel, feasibility masks, the sub-step order (reservation conflicts become no-ops; reverted txs leave no trace), and the export recompute. Calibration report. | local | — |
| M5 | MARL Full, NDA (3 seeds each) | First, the frozen-opponent capability test passes (`hns.marl.capability`, reports/m5_capability.md). Then: health curves present, and the §10 acceptance (complete with closed accounting, calibrated values, adapted launderers). The degenerate rule and the co-evolution check are reported. | A30 | ≈ 6 runs × 2.5 h = 15 h (first runs: 3.5–3.8 h each, six at once on one A30) |
| M6 | NMAI pretrain, exports, JS | JS table, Fig. 6 analogue, Table 2 analogue. | A30 | 2 h |
| M7 | Fine-tune grid (4 arms × 3 fractions × 3 seeds × 2 tasks, plus zero-shot) | All runs logged. `reports/results.md` generated by `hns.eval.report`. | A30, 4 processes in parallel | ≈ 4 h |
| M8 | Optional: Q4 noise sweep, SRF, W-temporal, T0-shared, X2 (v1-only T) | — | A30 | ≤ 4 h |

The total is about 25 A30-hours. M5's 2.5 h per run is an estimate: measure it on a 20-episode benchmark first, and re-plan from that. If M5 takes more than 4 h per run, cut it in this order: drop the NDA seeds to 1, then cut the sub-steps to T = 30.

## 16. Risks and fallbacks

- **MARL is unstable, degenerate, or fails the §10 acceptance.**
  - **Test the learner before the game** (ours; added after M5's first runs). Check the target, the BTC accounting and the role networks' train/eval dropout gap, then run the frozen-opponent capability test (`hns.marl.capability`). If a legal scripted policy is profitable and the learned one misses it, repair the learner or its exploration; do not make the game easier.
  - Only if well-covered policies find no escape, change the game, one change at a time and in every simulator arm alike:
    1. Freeze innocents to their scripted routes, so only two roles learn.
    2. Then shrink the strategy set.
    3. Then raise or lower the detector budget range.
  - Each repair cycle is declared in advance. It ends with the capability test and a short live pilot before any full re-run.
  - **Stop rule.** After two repair cycles without passing the §10 acceptance, stop tuning. Run M6/M7 with the static (NMAI) pretraining against T0, report the co-trained detectors as an exploratory arm, and report Q3 as "co-evolution not identified". Never choose an M5 variant or checkpoint after seeing an M7 target-test result.

  Report which fallbacks were used.
- **The simulator detector wins trivially** (detection > 0.95 from early on). The launderers are too weak. Widen `k` or the hop limit, or lower `b`.
- **Launderer txs are separable by template shape alone.** Detect this by a static-mode probe on shared features (template-only criminals, spend-all inputs, pruning artefacts), or by the bullet above. Bounded fix: give launderers ordinary spending that reuses the innocents' demand machinery. AMLworld motivates this: 81,143 of its 100,604 LI-Large laundering txs fall outside the 8 patterns (its Table 8). Import the mechanism only, not AMLworld's banking proportions. Full placement models, shell-company economies and transitive-taint labels stay out of scope. Taint is not how Elliptic++ labels txs.
- **Aggregate rates are unstable.** In an agent-based generator, small parameter changes can swing aggregate rates sharply (AMLworld §5). Report per-world tx counts and the true illicit share for every multiverse draw, and look at them before M5.
- **The simulator cannot match the real feature distributions** (calibrated mean JS > 0.3). Report it. Q2 is then expected to be negative, and that is a finding.
- **No A30 is free.** Check `jl gpus` for an equivalent with ≥ 24 GB at ≤ $0.6/h. Do not switch silently: note it in the run config.
- **T0 already beats S0 by a wide margin.** Q2 then rests on the 5 % and 25 % fractions, where pretraining can still help.

## Appendix: numbers for context

| Source | Task | Metric | Value |
|---|---|---|---|
| Elliptic++ guide repo notebook, RF (50 trees) | T, official split | illicit P / R / F1, micro-F1 | 0.975 / 0.719 / 0.828, 0.980 |
| Elliptic++ guide repo notebook, RF (50 trees) | W, official split | illicit P / R / F1, micro-F1 | 0.911 / 0.789 / 0.845, 0.990 |
| Elliptic++ guide repo notebook, LR / MLP / XGB | T | illicit F1 | 0.448 / 0.612 / 0.754 |
| Elliptic++ guide repo notebook, LR / MLP / XGB | W | illicit F1 | 0.083 / 0.532 / 0.662 |
| Paper (digest, Table 4) | "Elliptic" | Acc / F1 / AUC | 95.2 / 94.1 / 0.981 (not comparable, E10) |
| Paper (digest, Table 5) | "Elliptic++" | Acc / F1 / AUC | 94.0 / 93.0 / 0.970 (not comparable, E10) |
