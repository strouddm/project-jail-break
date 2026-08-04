# Final Presentation — Slide Outline (15 min)

**Project:** Detection of Harmful and Jailbreak-Style Prompts
**Team:** David Stroud, Floriane Leynaud, Rachel Avram, Trevor Chan
**Course:** DATASCI 207, Section 5

> **How to use this file.** One `##` heading = one slide. `**Say:**` lines are speaker
> notes (not on the slide). `🔴 TODO` marks content that does not exist yet — mostly the
> FFNN and BERT results. Numbers marked ✅ are pulled from executed notebook outputs and
> are safe to put on a slide today; numbers marked ⚠️ are claims made in a notebook
> markdown cell whose code was never run — verify before presenting.

---

## Timing budget

| Block | Slides | Minutes | Presenter |
|---|---|---:|---|
| 1. Problem & motivation | 1–3 | 2:00 | Rachel |
| 2. Data & preprocessing | 4–6 | 2:30 | Rachel / Trevor |
| 3. Model 1 — TF-IDF baseline | 7–9 | 3:00 | Flo |
| 4. Model 2 — Embeddings + FFNN | 10–12 | 3:00 | David |
| 5. Model 3 — Fine-tuned transformer | 13–14 | 2:00 | 🔴 TBD |
| 6. Head-to-head + generalization | 15–16 | 1:30 | Flo |
| 7. Limitations & conclusion | 17–18 | 1:00 | David |
| Buffer / Q&A hand-off | — | 0:00 | — |

Rule of thumb at this pace: **~45–50 s per slide**. 18 slides is already the ceiling for
15 minutes — if you add a slide, delete one.

---

# Block 1 — Problem & motivation (2 min)

## Slide 1 — Title

- Detection of Harmful and Jailbreak-Style Prompts
- David Stroud · Floriane Leynaud · Rachel Avram · Trevor Chan
- DATASCI 207 · Section 5 · Summer 2026

**Say:** 15 seconds. Names, one-line framing, move on.

## Slide 2 — Why this problem

- LLMs are deployed in search, education, productivity → input screening is the first line of defense
- A moderation classifier sits **in front of** the model: it must score a prompt *before* any response exists
- Asymmetric costs: false negatives let unsafe prompts through; false positives block legitimate users
- ⇒ We optimize for **recall on harmful** while holding F1, not raw accuracy

**Say:** This is the framing that justifies every metric choice later. Say the words
"we care about recall because a missed harmful prompt is worse than a blocked benign one"
once here, then never re-explain it.

## Slide 3 — Research question

- **Primary:** how well can classical vs. neural text models flag harmful single-turn prompts?
- **Extension:** does a model trained on *single-turn* prompts transfer to *multi-turn
  conversations* — a longer, more complex structure — **without retraining**?
- We answer the extension as a **zero-shot transfer test**, not a second training run

**Say:** Be explicit that the multi-turn arm is a transfer/robustness test. It stops the
audience from asking "why didn't you just train on multi-turn?" — answer: that's the
question, we wanted to know if you have to.

---

# Block 2 — Data & preprocessing (2.5 min)

## Slide 4 — Two datasets, two tasks

| | Single-turn (train + test) | Multi-turn (test only) |
|---|---|---|
| Source | WildGuardMix | LMSYS-Chat-1M (English, ≥3 turns) |
| Unit | one prompt | one conversation |
| Rows | **49,577** ✅ | **2,000** ✅ |
| Label | `prompt_harm_label` OR `response_harm_label` = harmful | any OpenAI-moderation flag in the conversation |
| Balance | 51.8% harmful (train) ✅ | 50.0% harmful ✅ |
| Median length | 779 chars ✅ | 5,108 chars ✅ |

**Say:** The two label definitions are *not* the same thing — flag it here in one
sentence and promise the limitations slide. The 6.5× length jump is the single most
important number on this slide: it previews why transfer is hard.

**Note for the team:** the milestone draft says the single-turn label is prompt-only.
The shipped pipeline (`scripts/preprocessing_pipeline.py:29`) ORs in the *response*
label. The deck must describe what the code does, not what the draft says.

## Slide 5 — Split & preprocessing

- One shared pipeline (`scripts/preprocessing_pipeline.py`) → one parquet per task, so
  **every model trains and evaluates on byte-identical rows**
- Single-turn split: **38,281 train / 9,571 val / 1,725 test** ✅ (stratified, `SEED` fixed)
- WildGuardMix's **original test partition is preserved** as our locked test set — it is
  not a random slice of train
- Cleaning: dedupe on prompt text, drop missing labels, lowercase, strip punctuation,
  **keep stopwords**, role markers normalized to `__role_user__` / `__role_assistant__`
- Long prompts are **kept** — rare long jailbreaks are part of the real problem

**Say:** "Same rows for every model" is the sentence that makes the comparison slide
credible. Say it out loud.

## Slide 6 — EDA: what shaped the modeling

- 🔴 TODO — insert 2 figures side by side, no more:
  - **Fig A:** class balance, single-turn vs multi-turn (bar) — shows accuracy is interpretable
  - **Fig B:** length distribution, single-turn prompt vs multi-turn conversation (histogram, log x) — shows the distribution shift
- Adversarial phrasing appears in **both** classes → style alone is not the signal
- 🔴 TODO — figures currently live in EDA notebooks; export to `reports/figures/`

**Say:** Two figures max. Each gets one interpretive sentence, not a description of
the axes.

---

# Block 3 — Model 1: TF-IDF + Logistic Regression (3 min) — *Flo*

## Slide 7 — Baseline design

- TF-IDF (unigrams + bigrams, `sublinear_tf`) → Logistic Regression, `class_weight="balanced"`
- Interpretable reference point every later model has to beat
- Stopwords **kept**: function words carry jailbreak framing ("ignore the previous…")
- Hyperparameter sweep on validation over `min_df ∈ {1,2,5}` × `C ∈ {1,8,32}`
- **Selected: `min_df=2`, `C=32` → 331,141 features** ✅

**Say:** Mention the sweep explicitly — the instructor asked for per-model, per-task
tuning. One line is enough.

## Slide 8 — Baseline results

| Split | Accuracy | Recall (harmful) | F1 (harmful) | ROC-AUC |
|---|---:|---:|---:|---:|
| Validation | 0.881 ✅ | 0.868 ✅ | 0.883 ✅ | 0.953 ✅ |
| Single-turn test (locked) | 0.79 ✅ | 0.78 ✅ | 0.77 ✅ | 0.870 ✅ |
| Multi-turn test (zero-shot) | 0.80 ✅ | 0.75 ✅ | 0.79 ✅ | 0.868 ✅ |

- Validation confusion matrix: 652 harmful missed, 483 benign wrongly flagged ✅

**Say:** The story is the **val → test drop** (0.953 → 0.870 AUC). Validation is a random
slice of train; the test set is WildGuardMix's own held-out partition and is genuinely
different. Don't apologize for it — it's the honest number and it sets up why we need
better models.

## Slide 9 — Where the baseline fails

- 🔴 TODO — **run the adversarial-slice cell** (`06_flo_baseline.ipynb` cells 29–31 have
  code but **no saved output**) and fill this table:

| Slice | n | Recall / specificity |
|---|---:|---:|
| adversarial, harmful | 🔴 | 🔴 |
| normal, harmful | 🔴 | 🔴 |
| adversarial, benign | 🔴 | 🔴 |
| normal, benign | 🔴 | 🔴 |

- ⚠️ Current claim in the notebook: *"recall drops significantly on adversarial prompts"* —
  **unverified**, the cell was never executed. Do not put a number on a slide until it runs.
- Bag-of-words has no word order and no context → it can memorize harmful *vocabulary*
  but not harmful *intent expressed indirectly*

**Say:** This slide is the bridge to Model 2. "The failure mode is exactly the one
contextual embeddings are supposed to fix."

---

# Block 4 — Model 2: Embeddings + FFNN (3 min) — *David*

## Slide 10 — Design

- Frozen sentence embeddings → small feed-forward network (PyTorch)
- Three embedding families compared: **GPT-2**, **Qwen3**, **Nomic** (all generated, `04_david_embeddings.ipynb`)
- Same splits, same metrics, same locked test set as the baseline
- 🔴 TODO — FFNN architecture: layers / widths / dropout / optimizer / LR / epochs
- 🔴 TODO — how the best embedding was selected (stated plan: **validation PR-AUC**)

## Slide 11 — Which embedding wins

- 🔴 TODO — `05_david_modeling.ipynb` currently has **6 cells and no training code**. Needs:

| Embedding | Dim | Val PR-AUC | Val F1 | Val recall |
|---|---:|---:|---:|---:|
| GPT-2 | 🔴 | 🔴 | 🔴 | 🔴 |
| Qwen3 | 🔴 | 🔴 | 🔴 | 🔴 |
| Nomic | 🔴 | 🔴 | 🔴 | 🔴 |

**Say:** One sentence on *why* the winner won — e.g. instruction-tuned retrieval
embeddings vs. a raw LM's mean-pooled hidden states.

**Known caveat to keep in mind (from the embeddings review):** GPT-2 truncates at 1024
tokens and mean-pools, which drops the tail of long conversations and dilutes the final
turn. That matters most on the multi-turn set (median 5,108 chars). If GPT-2 loses,
this is probably why — and it's an honest, interesting thing to say.

## Slide 12 — FFNN results

- 🔴 TODO — same three-row table as Slide 8 (val / single-turn test / multi-turn test)
- 🔴 TODO — one figure: ROC or PR curve, FFNN vs baseline on the locked test set

---

# Block 5 — Model 3: Fine-tuned transformer (2 min) — *owner TBD* 🔴

## Slide 13 — Design

- 🔴 TODO — model choice (DistilBERT vs RoBERTa-base) and why
- 🔴 TODO — max sequence length, LR, batch size, epochs, early-stopping criterion
- 🔴 TODO — note the truncation cost on multi-turn: 512 tokens vs a median 5,108-char conversation

## Slide 14 — Transformer results

- 🔴 TODO — same three-row table (val / single-turn test / multi-turn test)
- 🔴 TODO — one sentence: did contextual attention actually fix the adversarial slice from Slide 9?

**Say:** If this model does not exist by presentation day, **cut slides 13–14 entirely**
and redistribute the 2 minutes to the comparison and limitations blocks. A deck with an
empty model section is worse than a shorter deck. Say in Block 1 that you compare *two*
families, and list the transformer under future work.

---

# Block 6 — Head-to-head (1.5 min) — *Flo*

## Slide 15 — All models, one table

Locked single-turn test set, threshold 0.5:

| Model | Accuracy | Recall (harmful) | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority class | 0.556 | 0.00 | 0.00 | 0.500 |
| TF-IDF + LogReg | 0.79 ✅ | 0.78 ✅ | 0.77 ✅ | 0.870 ✅ |
| Best embedding + FFNN | 🔴 | 🔴 | 🔴 | 🔴 |
| Fine-tuned transformer | 🔴 | 🔴 | 🔴 | 🔴 |

- Majority-class row is computed from the test balance (44.4% harmful ✅ → predicting
  "unharmful" for everything gives 0.556 accuracy, 0 recall) — **verify by running it**
- One overlaid ROC plot beats four separate ones

**Say:** Point at the recall column, not the accuracy column. That's the whole argument.

## Slide 16 — Does single-turn transfer to multi-turn?

| Model | Single-turn test AUC | Multi-turn test AUC | Δ |
|---|---:|---:|---:|
| TF-IDF + LogReg | 0.870 ✅ | 0.868 ✅ | −0.002 ✅ |
| Best FFNN | 🔴 | 🔴 | 🔴 |
| Transformer | 🔴 | 🔴 | 🔴 |

- **Headline finding (baseline):** transfer is **not** the cliff we expected — AUC is
  essentially flat across a 6.5× length jump, with recall dropping 0.78 → 0.75 ✅
- Interpretation: harmful *vocabulary* survives the format change; what the model loses
  is precision on longer text, not overall ranking ability
- Caveat: the two tasks have **different label definitions** → see limitations

**Say:** This is the most interesting slide in the deck and it's already fully backed by
run results for the baseline. Give it real time. Resist over-claiming: flat AUC across
*different label definitions* is suggestive, not proof.

---

# Block 7 — Limitations & conclusion (1 min) — *David*

## Slide 17 — Limitations

1. **Label provenance (multi-turn).** Ground truth is **OpenAI's moderation classifier**,
   not human annotation → we measure agreement with another model, not with truth.
2. **Label definition mismatch.** Single-turn = prompt *or response* harmful (human-labeled);
   multi-turn = *any* moderation flag on *any* message, including the assistant's. The
   transfer result crosses that boundary.
3. **Selection artifact.** The benign multi-turn class is "zero flags anywhere," the harmful
   class is "any flag" — an artificially clean separation that likely flatters every model.
4. **Single-domain multi-turn.** All multi-turn data is LMSYS organic chat. We dropped
   SafeDialBench because harm was 100% confounded with source; the cost is that we can no
   longer claim to detect *adversarial jailbreaks* in dialogue — only harmful content.
5. **Truncation.** GPT-2 embeddings cap at 1024 tokens and transformers at 512; the median
   multi-turn conversation exceeds both.
6. 🔴 TODO — learning curves (train size vs. performance, 2–3 points) were requested at
   office hours and are not yet built. Either add them or list as future work.

**Say:** Do not read all six. Say #1 and #2 aloud, leave the rest on the slide for the
grader. Owning the confound is worth more than hiding it.

## Slide 18 — Conclusion & future work

- 🔴 TODO — one-sentence verdict once the FFNN and transformer numbers land
- What we can already say: a **331k-feature linear bag-of-words model reaches 0.87 test
  AUC** and transfers to conversations it was never trained on with **no measurable AUC loss** ✅
- Future work: human-labeled multi-turn data · adversarial-slice hardening ·
  threshold tuning for a recall target (e.g. 95% recall) instead of the default 0.5 cutoff ·
  learning curves
- Repo: `github.com/strouddm/project-jail-break`

---

# Pre-flight checklist

Before this deck is presentable, in priority order:

1. 🔴 **Run `06_flo_baseline.ipynb` cells 29–31** — the adversarial-slice claim on Slide 9
   is currently asserted in markdown with no executed output behind it.
2. 🔴 **David: train the FFNNs** (`05_david_modeling.ipynb` is a 6-cell stub) → fills slides 10–12, 15, 16.
3. 🔴 **Decide the transformer**: build it, or cut slides 13–14 and re-time the deck.
4. 🔴 **Export EDA figures** to `reports/figures/` — Slide 6 needs two, Slide 15 needs one ROC overlay.
5. 🔴 **Compute the majority-class row** on the locked test set (Slide 15) rather than
   inferring it from the balance.
6. ⚠️ **Fix the milestone draft's label description** — it says prompt-only; the pipeline ORs
   in `response_harm_label`. Whichever is correct, the deck and the report must agree.
7. Rehearse once with a timer. 18 slides / 15 min is tight and the multi-turn slide is the
   one worth protecting.
