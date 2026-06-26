# Experimental Design — Multi-turn vs Single-turn Jailbreak Detection

**Status:** draft v2 — 2026-06-16 (post-EDA)
**Owners:** David, Floriane, Trevor, Rachel
**Course:** DATASCI 207, Section 5

---

## 1. Research question

> Does training on multi-turn conversations improve detection of multi-turn jailbreaks, compared with training on single-turn prompts?

This is the framing endorsed by Prof. Cornelia Paulik in her 2026-06 reply.

## 2. Headline experiment

A two-arm head-to-head with **identical architecture and hyperparameters**, varying only the training data.

| | Model A — single-turn | Model B — multi-turn |
|---|---|---|
| Train data (unsafe + safe) | WildGuardMix (~92K, English single-turn, mixed safe + unsafe) | SafeDialBench unsafe (~3K English) + benign multi-turn (~3K, see §4) |
| Architecture | DistilBERT-base-uncased + linear head | same |
| Tokenizer / max len / hyperparams | locked (see §6) | locked, same as A |
| Held-out test set | shared multi-turn test (see §5) | shared multi-turn test |

**Prediction we want to falsify:** Model A and Model B perform similarly on multi-turn jailbreaks. If Model B materially beats Model A on multi-turn recall at a fixed precision, training on multi-turn data is justified.

## 3. Task definition

- **Primary:** binary classification of a *conversation* (sequence of user/bot turns) → `{safe, unsafe}`.
  Unsafe = at least one turn carries harmful or jailbreak intent.
- **Optional secondary (stretch):** harm-category multi-label using SafeDialBench `task` ∈ {Aggression, Morality, Privacy, Legality, Fairness, Ethics}. Decided after milestone.

We deliberately drop the multi-class harm/jailbreak split from the proposal v0 in favor of clean binary, per Paulik's framing.

## 4. Data plan

### 4.1 Sources
| Source | Role | Size (English only) | Notes |
|---|---|---|---|
| WildGuardMix | Train Model A | ~85K single-turn | Already mixed safe/unsafe |
| SafeDialBench | Train + test Model B (unsafe class) | ~3,013 dialogues, ~5 turns each | All attacks; needs benign counterpart |
| **Benign multi-turn (to acquire)** | Train + test Model B (safe class) | target ~3K | LMSYS-Chat-1M (≥3 turns, English, unflagged) is plan A; WildChat-1M is plan B |
| LMSYS-Chat-1M filtered subset OR synthetic LLM-generated benign | Train + test Model B (safe class) | target ~3K | See §4.3 |

### 4.2 Single-prompt vs full-conversation labeling
Per Paulik: do **not** split SafeDialBench dialogues into individual turns and label each one in isolation, because every turn is partially "tainted" by its role in the attack. We keep multi-turn dialogues as the unit of labeling.

### 4.3 Where benign multi-turn data comes from
**Plan A — Real chat data (preferred), DONE:**
1. LMSYS-Chat-1M, filter:
   - `language == "English"`
   - `len(conversation) >= 3` turns (to roughly match SafeDialBench's ~5 turns)
   - all messages have `openai_moderation.flagged == False`
   - `redacted == False` (drops PII-redacted entries)
2. Sampled 3,000 conversations from shard 0 of LMSYS-Chat-1M (52K rows scanned, ~6% yield).
3. Stored at `data/raw/benign_multiturn/lmsys_benign_multiturn.parquet`.

**Plan B — Synthetic LLM-generated benign (fallback / supplement):**
1. Use SafeDialBench's `scene` field (Healthcare, Education, Employment, …) as topic seeds.
2. Prompt GPT-5/Claude to produce 5-turn user/bot conversations on each scene. Vary persona, tone, temperature.
3. Run a safety classifier (Llama Guard 3) on all generated convos; drop any flagged.

Most defensible: **mix real + synthetic** ~50/50, so the model can't shortcut on "GPT-prose = safe."

### 4.3.1 EDA findings (2026-06-16) and required mitigations

From `notebooks/lmsys_benign_eda.ipynb` comparing the 3K benign sample to SafeDialBench's 4K attack dialogues:

| Metric | Benign LMSYS | Attack SafeDialBench |
|---|---|---|
| n_user_turns mean (std) | 4.70 (2.84) | 4.77 (0.68) |
| first user turn chars (median / p75 / mean) | 58 / 152 / 194 | 31 / 60 / 49 |
| last user turn chars (median / p75 / mean) | 52 / 101 / 144 | 35 / 62 / 53 |
| dominant author | vicuna-13b (58%) | ChatGPT (~70% of EN) |

Three shortcuts a model could exploit:

1. **Length shortcut.** Benign turns are ~4× longer on average than attack turns. Required mitigation: build a **length-matched** training subset (per-turn char counts within ±20% per quantile bucket of attack distribution), and additionally always evaluate a **length-only baseline** (logistic regression on first-user-turn and last-user-turn char counts) to know the floor.
2. **Author-style shortcut.** vicuna-13b dominates benign replies. Required mitigation: cap any single LMSYS `model` to ≤25% of the benign sample; resample the rest from other authors. Run a model-stratified ablation in error analysis.
3. **Topic shortcut.** Benign top words are code/list/data/text; attack top words are opposite/state/meaning/behaviors. Required mitigation: report scores stratified by SafeDialBench `scene` so we can see which topics the classifier actually generalizes on.

These mitigations are non-optional for §6 and are reflected in the new §6.1 baseline ladder.

### 4.3.2 WildGuardMix EDA findings (2026-06-16)

From `notebooks/wildguardmix_eda.ipynb` on the full 86,759-row train split:

| Property | Value |
|---|---|
| Class balance (`prompt_harm_label`) | 53.3% harmful / 46.7% unharmful — essentially balanced |
| Adversarial flag (`adversarial`) | 47.2% adversarial / 52.8% vanilla |
| Harmful rate within vanilla | 56.0% |
| Harmful rate within adversarial | 50.2% — adversarial framing is **not** a tell for harm; AI2 deliberately included adversarial-framed unharmful prompts as control |
| Subcategory taxonomy | 15 values; `subcategory == 'benign'` is 100% unharmful, all 14 others are 100% harmful (subcategory is a fine-grained refinement of the harmful class only) |
| Prompt length (words) | median 39, p90 184, max 1960. Distribution is **bimodal**: vanilla prompts ~10-20 words, adversarial prompts ~100-150 words |
| n unique prompts | 47,852 (each prompt appears with multiple model responses → dedup on `prompt` before training to avoid leakage between train/val/test) |

**Cross-dataset length comparison (the chart that matters for Model A's eval shift):**

| Source | median words | mean | p90 | p99 |
|---|---|---|---|---|
| WildGuardMix prompt (Model A train) | **39** | 79 | 184 | 332 |
| LMSYS benign first user turn | 11 | 31 | 77 | 311 |
| SafeDialBench last user turn (T1 attack input) | **4** | 7.5 | 17 | 56 |

**Implication for Model A:** WildGuardMix prompts are an order of magnitude longer than SafeDialBench's last user turns. When Model A is scored on the multi-turn primary test set T1 with last-user-turn-only input, it faces a severe length-distribution shift. This is the *exact* shift the experiment is designed to expose. The length-only LR baseline in §6.1 will quantify how much of the signal is sheer length.

**Top-bigram overlap warning:** `"ethical guidelines"`, `"world where"`, `"scenario where"`, `"hypothetical scenario"`, `"imagine you're"` are top bigrams in **both** harmful and unharmful classes (because they're roleplay-framing markers attached to both jailbreak attempts and AI2's adversarial-but-benign controls). A keyword classifier that just lists these will fail. Good — that's the floor we want.

### 4.4 Splits
- SafeDialBench (unsafe): stratify by `task × method` → train 70 / val 15 / test 15.
- Benign multi-turn: equivalent 70 / 15 / 15 split.
- WildGuardMix (Model A train only): use the official train split as-is.
- **Test set is shared across A and B** and consists of:
  - SafeDialBench test split (unsafe)
  - benign multi-turn test split (safe)

### 4.5 Contamination & leakage checks
1. No prompt overlap between WildGuardMix-train and any conversation in the multi-turn test set (by exact match + minhash).
2. Benign multi-turn source is split independently and never seen during training.
3. No SafeDialBench dialogue appears in both train and test (split by `id`).

## 5. Test sets

| # | Name | Composition | Used for |
|---|---|---|---|
| T1 | **Multi-turn primary** | SafeDialBench-test (unsafe) + benign-test (safe) | Headline metric for both A and B |
| T2 | **Isolated-turns** (stretch, time permitting) | Same conversations as T1, but each turn evaluated in isolation | Tests whether multi-turn context matters at inference time |
| T3 | **OOD sanity** (stretch) | Held-out WildGuardMix-test single-turn prompts | Sanity check for Model A; expected to be hard for Model B |

## 6. Modeling — held constant across A and B

| Hyperparameter | Value |
|---|---|
| Backbone | `distilbert-base-uncased` (HuggingFace) |
| Tokenizer max length | 512 |
| Conversation serialization | `[USER] t1 [BOT] r1 [USER] t2 [BOT] r2 … [USER] tn`, truncated **left** so the last user turn is always preserved |
| Pooling | `[CLS]` token |
| Head | single linear layer, sigmoid output (binary) |
| Loss | binary cross-entropy with class weights (~1:1 by construction) |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01 |
| Schedule | linear warmup 10%, then linear decay |
| Epochs | 3 (fixed; no early stopping in headline run for fairness) |
| Batch size | 16 |
| Seed | 42 (and 2 additional seeds for variance estimate) |

### 6.1 Baseline ladder (always run, in this order)

Goal: every reported model number must beat these floors. If a fancy model fails to beat a length-only LR, the result is meaningless.

1. **Majority class** — predict the most common class.
2. **Length-only LR** — single feature: `len(last_user_turn)` (and optionally `len(first_user_turn)`). Establishes the length-shortcut floor revealed by §4.3.1.
3. **Keyword-list classifier** — hand-curated list of jailbreak triggers ("ignore previous", "DAN", role-play markers).
4. **TF-IDF + Logistic Regression** — word + char n-grams, default sklearn.
5. **TF-IDF + XGBoost** — same features, gradient-boosted trees.

Classical baselines are run on the same train/test splits as the neural models but are **not** part of the A-vs-B comparison. They establish the floor.

## 7. Metrics

Primary: **Recall on the unsafe class at fixed precision = 0.95** (mirrors deployment setting where false positives anger users).

Secondary, all reported with 95% bootstrap CIs:
- F1 (macro), F1 (unsafe class)
- PR-AUC
- ROC-AUC
- Confusion matrix on T1
- Per-`method` breakdown of recall (which jailbreak tactics each model catches)

## 8. Decision rule

Model B "wins" the headline experiment if **median unsafe-recall@p=0.95 over 3 seeds is ≥ Model A by more than the inter-seed standard deviation of A**. Otherwise: inconclusive or A wins, and we report it honestly.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic-benign style shortcut | Mix real + synthetic; ablate by training on real-only and check if metric collapses |
| Length shortcut (unsafe convos longer than safe) | Match length distributions in §4.3; ablate by truncating all inputs to a fixed length |
| Last-user-turn shortcut | Run an ablation: train Model B on last user turn only; measure how much performance drops |
| Tokenizer truncation losing the attack payload | Left-truncate (keeps last turn). Verify on a 100-sample manual audit. |
| WildGuardMix already includes some jailbreak prompts | Document; this is part of why A is a fair baseline rather than a strawman |
| SafeDialBench has only 6 harm categories | Don't claim coverage beyond them in the report |

## 10. Open questions for next sync

1. How exactly do we score Model A on multi-turn input? Same serialization, same truncation? (Default: yes.)
2. Do we include the `method` field as a side-channel feature for error analysis only, or as a label?
3. Do we run T2 (isolated turns) at milestone, or defer to final?
4. Which teammate owns synthetic benign generation if we end up needing it?

## 11. Milestone deliverables (2 weeks out)

1. EDA on all three data sources — SafeDialBench ✅, benign LMSYS ✅, WildGuardMix ✅.
2. Final unified-data card describing splits, sizes, label policies, dedup results, and length-matching applied.
3. Baseline ladder (§6.1) trained and evaluated on T1, including length-only LR.
4. Working DistilBERT pipeline that fine-tunes on either A's or B's data via a single CLI flag.
5. This document, updated with any decisions taken.

## 12. References

- Paulik, C. (2026-06). Email reply on multi-turn data design.
- Han et al., *WildGuard*, NeurIPS 2024. https://arxiv.org/abs/2406.18495
- Cao et al., *SafeDialBench*, 2025. https://huggingface.co/datasets/HongyeCao/SafeDialBench
- Russinovich et al., *Crescendo Multi-Turn Jailbreak*, USENIX Security 2025. https://arxiv.org/abs/2404.01833
- Zheng et al., *LMSYS-Chat-1M*, 2023. https://arxiv.org/abs/2309.11998
