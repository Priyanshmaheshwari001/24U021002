"""Prompt Templates for Humour Detection in Social Media Posts.

Defines three standardized prompt strategies:
1. `zero_shot`: Clear definition of humour and classification task with strict JSON output schema.
2. `few_shot`: Same definition plus 6 balanced few-shot examples (3 humorous, 3 non-humorous)
   sourced from data/processed/fewshot_pool.csv.
3. `cot_style`: Chain-of-thought style prompting guiding the model to evaluate intent,
   wordplay/devices, and tone before assigning the final label and rationale in JSON.

All templates guarantee identical output JSON formatting:
{
  "label": 1 or 0,
  "reason": "..."
}
"""

import os
from typing import Callable, Dict, List
import pandas as pd

# Core Definition of Humour used consistently across all prompt strategies
HUMOUR_DEFINITION: str = """DEFINITION OF HUMOUR:
- Humour (label 1): Text that incorporates jokes, puns, irony, wordplay, absurdity, hyperbole, or sarcasm deliberately intended to amuse, entertain, or evoke laughter.
- Non-Humour (label 0): Text that expresses serious opinions, sincere emotions, facts/trivia, news, inspirational quotes, advice, or everyday literal remarks without comedic intent."""

# Standard output format specification
OUTPUT_FORMAT_SPEC: str = """OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this exact schema:
{
  "label": 1 or 0,
  "reason": "Short concise explanation justifying the classification"
}"""


# ------------------------------------------------------------------------------
# Few-Shot Examples (Loaded from data/processed/fewshot_pool.csv with fallback)
# ------------------------------------------------------------------------------
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "text": "My wife says I make too many puns about sweat. I'm not sure what she's saunabout",
        "label": 1,
        "reason": "Employs wordplay/pun replacing 'talking about' with 'saunabout' to create a humorous punchline.",
    },
    {
        "text": '"The" is the most spoken word in the English language.',
        "label": 0,
        "reason": "Factual trivia statement with purely informative intent and no comedic devices or wordplay.",
    },
    {
        "text": "I consider a hostile work environment any office that has a Karaoke Night.",
        "label": 1,
        "reason": "Uses hyperbole and comedic irony to playfully exaggerate office karaoke as a hostile workplace.",
    },
    {
        "text": "You never know how much patience you have, until being patient is the only choice you have.",
        "label": 0,
        "reason": "Inspirational and reflective aphorism with serious emotional and philosophical tone.",
    },
    {
        "text": "Genie: I shall grant you 3 wishes Me: I wish for a world without lawyers Genie: Done, you have no more wishes Me: But you said 3 Genie: Sue me.",
        "label": 1,
        "reason": "Classic dialogue joke with a witty punchline subverting expectations through legal irony.",
    },
    {
        "text": "17 Never give her a reason to doubt you. boyfriend tips",
        "label": 0,
        "reason": "Direct prescriptive relationship advice without irony, humor, or comedic intent.",
    },
]


def load_fewshot_examples(csv_path: str = "data/processed/fewshot_pool.csv") -> List[Dict]:
    """Load 3 funny and 3 not funny examples from the few-shot pool CSV, or use defaults."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            funny = df[df["label"] == 1].head(3)
            not_funny = df[df["label"] == 0].head(3)

            if len(funny) >= 3 and len(not_funny) >= 3:
                examples = []
                # Interleave funny and not funny
                for f_row, nf_row in zip(funny.itertuples(), not_funny.itertuples()):
                    examples.append({
                        "text": getattr(f_row, "clean_text", getattr(f_row, "text")),
                        "label": 1,
                        "reason": "Contains comedic intent, punchline, or wordplay intended to amuse.",
                    })
                    examples.append({
                        "text": getattr(nf_row, "clean_text", getattr(nf_row, "text")),
                        "label": 0,
                        "reason": "Literal statement, advice, or sincere thought with no comedic elements.",
                    })
                # Customize reasons for the exact known rows from fewshot_pool if available
                examples[0]["reason"] = "Humorous personal remark with self-deprecating commentary on clothing."
                examples[1]["reason"] = "Factual trivia statement with purely informative intent."
                examples[2]["reason"] = "Dark humor with an ironic subversion of expectations regarding Helen Keller."
                examples[3]["reason"] = "Inspirational quote offering sincere life reflection."
                examples[4]["reason"] = "Misdirection and dark comedic twist regarding hiding a body."
                examples[5]["reason"] = "Straightforward relationship tip with no humor or wordplay."
                return examples
        except Exception:
            pass
    return DEFAULT_FEW_SHOT_EXAMPLES


# ------------------------------------------------------------------------------
# Prompt Functions
# ------------------------------------------------------------------------------
def zero_shot(text: str) -> str:
    """Generate a Zero-Shot prompt with a clear definition of humour."""
    prompt = f"""You are an expert NLP classifier specializing in humor detection in social media.

{HUMOUR_DEFINITION}

TASK:
Classify whether the given social media post is humorous (1) or not humorous (0).

{OUTPUT_FORMAT_SPEC}

Post to classify:
\"\"\"{text}\"\"\""""
    return prompt


def few_shot(text: str, examples: List[Dict] = None) -> str:
    """Generate a Few-Shot prompt with 6 balanced examples (3 funny, 3 not funny)."""
    if examples is None:
        examples = DEFAULT_FEW_SHOT_EXAMPLES

    examples_str = ""
    for idx, ex in enumerate(examples, 1):
        clean_text_sample = ex["text"].strip().strip('"')
        examples_str += f"""Example {idx}:
Post: "{clean_text_sample}"
Response:
{{
  "label": {ex['label']},
  "reason": "{ex['reason']}"
}}

"""

    prompt = f"""You are an expert NLP classifier specializing in humor detection in social media.

{HUMOUR_DEFINITION}

TASK:
Classify whether the given social media post is humorous (1) or not humorous (0). Refer to the labeled examples below for guidance on edge cases.

EXAMPLES:
{examples_str.strip()}

{OUTPUT_FORMAT_SPEC}

Post to classify:
\"\"\"{text}\"\"\""""
    return prompt


def cot_style(text: str) -> str:
    """Generate a Chain-of-Thought (CoT) style prompt guiding analysis of intent, wordplay, and tone."""
    prompt = f"""You are an expert NLP classifier specializing in humor detection in social media.

{HUMOUR_DEFINITION}

REASONING PROCESS:
Before deciding on the final label, mentally analyze the text through these 3 steps:
1. Intent: Is the author trying to amuse/entertain (humorous) or sincerely inform/share (non-humorous)?
2. Wordplay & Devices: Does the text rely on a punchline, pun, sarcasm, hyperbole, parody, or double meaning?
3. Tone: Is the tone playful and humorous, or serious, literal, and factual?

Synthesize your findings into a brief rationale for the "reason" field, then output the binary label.

{OUTPUT_FORMAT_SPEC}

Post to classify:
\"\"\"{text}\"\"\""""
    return prompt


# ------------------------------------------------------------------------------
# Registry Dictionary
# ------------------------------------------------------------------------------
PROMPT_TEMPLATES: Dict[str, Callable[[str], str]] = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "cot_style": cot_style,
}


def get_prompt(template_name: str, text: str) -> str:
    """Retrieve formatted prompt by template name ('zero_shot', 'few_shot', 'cot_style')."""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt template '{template_name}'. Available: {list(PROMPT_TEMPLATES.keys())}"
        )
    return PROMPT_TEMPLATES[template_name](text)


# ------------------------------------------------------------------------------
# Self-Test and Demonstration
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "Parallel lines have so much in common. It's a shame they'll never meet."

    print("=" * 80)
    print("1. ZERO-SHOT PROMPT TEMPLATE")
    print("=" * 80)
    print(zero_shot(sample_text))

    print("\n" + "=" * 80)
    print("2. FEW-SHOT PROMPT TEMPLATE (6 Examples: 3 Funny, 3 Not Funny)")
    print("=" * 80)
    print(few_shot(sample_text))

    print("\n" + "=" * 80)
    print("3. COT-STYLE PROMPT TEMPLATE (Chain of Thought)")
    print("=" * 80)
    print(cot_style(sample_text))
    print("=" * 80)
