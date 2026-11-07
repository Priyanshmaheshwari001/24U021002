# Evaluation Report: Humour Detection in Social Media Posts

**Task:** Binary Text Classification (Humorous `1` vs. Non-Humorous `0`)  
**Model:** Google Gemini (`gemini-3.5-flash-lite`, Free-Tier API)  
**Dataset:** `data/processed/eval_sample.csv` (Derived from HaHackathon benchmark)  
**Sample Size:** 30 samples (15 Humorous, 15 Non-Humorous — 50/50 balanced)  
**Date:** October 2026  

---

## 1. Executive Summary

This report evaluates three prompting strategies using Google's `gemini-3.5-flash-lite` model for detecting humor in social media text:
1. **Zero-Shot Prompting**: Direct task instructions with explicit schema constraints.
2. **Few-Shot Prompting**: Task instructions augmented with 6 balanced exemplars (3 humorous, 3 non-humorous) with reasoning.
3. **Chain-of-Thought (CoT) Prompting**: Multi-step structured instructions guiding the model to evaluate intent, comedic devices, and tone before assigning a label.

Across a balanced evaluation dataset of 30 social media posts, all three strategies successfully parsed and correctly classified the valid returned outputs. Network and transient API capacity errors accounted for a small number of incomplete requests, which are tracked and reported explicitly.

---

## 2. Experimental Setup & Dataset

### 2.1 Dataset Composition
The dataset is sampled from the public **HaHackathon Humour Detection** benchmark (`hahackathon_train.csv`), consisting of short-form tweets, puns, observational remarks, quotes, and factual trivia.

* **Target Classes**:
  * **`1` (Humorous)**: Puns, jokes, wordplay, sarcasm, and absurdist statements intended to entertain.
  * **`0` (Non-Humorous)**: Informative trivia, news, serious aphorisms, and sincere statements.
* **Cleaning Pipeline** (`src/prepare_data.py`):
  * Stripped URLs.
  * Anonymized user handles (`@mentions` $\rightarrow$ `@user`).
  * Demojized Unicode emojis into descriptive text tags (`:smile:`).
  * Normalized whitespace while preserving capitalization and punctuation to retain comedic timing.

### 2.2 Class Distribution ($N = 30$)
The evaluation sample was generated using `python src/prepare_data.py --n_per_class 15`:

| Class | Label | Count | Percentage |
| :--- | :---: | :---: | :---: |
| **Non-Humour** | `0` | 15 | 50.0% |
| **Humour** | `1` | 15 | 50.0% |
| **Total** | — | **30** | **100.0%** |

All three prompting strategies were evaluated on the **exact same 30 posts** to enable a direct, controlled comparison.

---

## 3. Evaluated Prompt Strategies

All strategies require the model to respond strictly with a JSON object:
```json
{
  "label": 0 or 1,
  "reason": "Brief explanation"
}
```

* **Zero-Shot**: Contains definitions for humorous (label 1) and non-humorous (label 0) text, followed by the target post and JSON schema constraints.
* **Few-Shot**: Supplies the zero-shot definitions plus 6 exemplars from `data/processed/fewshot_pool.csv` (disjoint from the eval sample). Exemplars cover phonetic puns, dialogue subversion, observational hyperbole, linguistic trivia, and sincere quotes.
* **Chain-of-Thought (CoT)**: Instructs the model to internally evaluate:
  1. *Intent*: Comedic vs. informative/literal.
  2. *Wordplay & Devices*: Puns, punchlines, sarcasm, irony, exaggeration.
  3. *Tone*: Playful vs. serious or sincere.
  Synthesizes these steps into the final explanation and label.

---

## 4. Quantitative Results & Metric Breakdown

Metrics are calculated with **Humour (`1`)** designated as the positive class.

### 4.1 Comparative Summary Table

| Metric | Zero-Shot | Few-Shot | CoT-Style |
| :--- | :---: | :---: | :---: |
| **Total Evaluated Rows** | 30 | 30 | 30 |
| **Valid Responses** | 30 | 29 | 29 |
| **Failed / Incomplete Rows** | **0 (0.0%)** | **1 (3.33%)** | **1 (3.33%)** |
| **Accuracy (on Valid Rows)** | **100.0%** (30/30) | **100.0%** (29/29) | **100.0%** (29/29) |
| **Strict Accuracy (All 30 Rows)** | **100.0%** (30/30) | **96.67%** (29/30) | **96.67%** (29/30) |
| **Precision (`1` - Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **Recall (`1` - Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **$F_1$-Score (`1` - Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **Precision (`0` - Non-Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **Recall (`0` - Non-Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **$F_1$-Score (`0` - Non-Humour)** | 1.0000 | 1.0000 | 1.0000 |
| **Macro $F_1$-Score** | 1.0000 | 1.0000 | 1.0000 |

*Note: In the strict accuracy calculation, failed/unreturned rows are treated as incorrect.*

---

### 4.2 Confusion Matrices

#### Zero-Shot ($N = 30$)
```text
                     Predicted: 0 (Non-Humour)    Predicted: 1 (Humour)
Actual: 0 (Non-Humour)          15 (TN)                       0 (FP)
Actual: 1 (Humour)               0 (FN)                      15 (TP)
```
* **True Positives (TP)**: 15 | **True Negatives (TN)**: 15
* **False Positives (FP)**: 0 | **False Negatives (FN)**: 0

#### Few-Shot ($N = 29$ Valid Samples, 1 Incomplete)
```text
                     Predicted: 0 (Non-Humour)    Predicted: 1 (Humour)
Actual: 0 (Non-Humour)          15 (TN)                       0 (FP)
Actual: 1 (Humour)               0 (FN)                      14 (TP)
```
* **True Positives (TP)**: 14 | **True Negatives (TN)**: 15
* **False Positives (FP)**: 0 | **False Negatives (FN)**: 0

#### CoT-Style ($N = 29$ Valid Samples, 1 Incomplete)
```text
                     Predicted: 0 (Non-Humour)    Predicted: 1 (Humour)
Actual: 0 (Non-Humour)          15 (TN)                       0 (FP)
Actual: 1 (Humour)               0 (FN)                      14 (TP)
```
* **True Positives (TP)**: 14 | **True Negatives (TN)**: 15
* **False Positives (FP)**: 0 | **False Negatives (FN)**: 0

---

## 5. Explicit Reporting of Failed Rows & Error Modes

Across the 90 total API requests (30 samples $\times$ 3 strategies), **2 requests failed** due to network or server-side constraints. Neither error stemmed from JSON parsing or model formatting issues:

| Strategy | Row Index | Text Snippet | True Label | Pred Label | Failure Cause |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **`few_shot`** | Row 26 | *"What do you call a lesbian Eskimo? A Klondike."* | 1 | -1 | **Network Drop**: Server disconnected without sending a response during transit. |
| **`cot_style`** | Row 29 | *"Don't you just hate it when the guy sucking you off turns out to be gay..."* | 1 | -1 | **API 503 Spike**: `503 UNAVAILABLE` — Google Gemini model experiencing temporary high demand. |

Both failures were recorded with `pred_label = -1` in `results/predictions/` and are included in the strict metric penalization.

---

## 6. Qualitative Findings: How the Model Analyzes Text

Reviewing the explanations produced by `gemini-3.5-flash-lite` reveals specific linguistic mechanisms identified across the benchmark:

1. **Phonetic Wordplay & Puns**:
   * *Sample*: `"What do you call a Christian who visits shrines? A roamin' Catholic. ROFL"`
   * *Model Reason*: *"The post is a joke featuring a pun on 'Roman Catholic' ('roamin' Catholic') explicitly intended to be funny."*
2. **Contextual Internet Tropes**:
   * *Sample*: `"**Finds old Ouija board** ... W-E-H-A-V-E-U-P-D-A-T-E-D-O-U-R-P-R-I-V-A-C-Y-P-O-L-I-C-Y"`
   * *Model Reason*: *"Uses subversion of expectation by turning a supernatural horror trope into a relatable internet meme about constant privacy policy updates."*
3. **Separating Trivia from Comedic Punchlines**:
   * *Sample*: `"Paul Newman's friends loved his homemade salad dressing so much that he started 'Newman's Own.' Newman never took personal pay, and the company donates 100% of its profits to charities..."`
   * *Model Reason*: *"The text presents factual historical trivia about Paul Newman's charitable business without any comedic intent or humor devices."*
4. **Sincere Reflections**:
   * *Sample*: `"'There are days that make the sacrifices seem worthwhile. And then there are the days where everything feels like a sacrifice.'"`
   * *Model Reason*: *"The post expresses a sincere, reflective observation about life's challenges without comedic intent, irony, or humorous elements."*

---

## 7. Limitations

The findings of this evaluation must be interpreted within the context of several clear constraints:

1. **Small Sample Size ($N = 30$)**:
   * While balanced (15 positive, 15 negative), a sample of 30 examples cannot represent the full distribution of humor styles, dialects, or cultural references present in broader social media. High accuracy on this subset should not be extrapolated as universal performance across larger datasets.
2. **Possible Dataset Contamination**:
   * The HaHackathon dataset was published in 2020 and is widely available on GitHub and in academic repositories. Because Google Gemini's pre-training web corpus includes extensive public web data, the model may have memorized or previously encountered some of these specific joke setups and punchlines.
3. **Prompt Sensitivity**:
   * Results are dependent on specific phrasing within the prompt templates. Subtle modifications to definitions, few-shot demonstration choices, or formatting instructions can alter boundary decisions, particularly for dry sarcasm, dark humor, or ambiguous cultural references.
4. **Free-Tier API Rate Limits & Quotas**:
   * Preview models (such as `gemini-3.8-flash`) enforce strict daily caps (20 requests/day). The stable `gemini-3.5-flash-lite` tier provides 500 requests/day and 15 requests/minute, but remains subject to occasional `503 UNAVAILABLE` spikes under high cloud demand. Reliable large-scale evaluation requires either paid tier provisioning or substantial persistent caching.

---

## 8. Summary & Next Steps

* **Performance**: Across valid responses, all three prompting methods (`zero_shot`, `few_shot`, `cot_style`) achieved 100% agreement with the reference labels on this 30-sample benchmark slice.
* **Reliability**: Incorporating defensive regex and JSON fence stripping eliminated all client-side parsing failures.
* **Next Steps**:
  * Scale the evaluation to a larger subset ($N \ge 100$) using batch processing or overnight scheduled caching.
  * Evaluate on newer, out-of-domain humor datasets created after 2024 to verify generalization and test for pre-training memorization.
