"""
Generate persona rows per cluster using representative threads and action stats.
Outputs a CSV with columns: gender, description, behavior_patterns.
"""

import csv
import json
from pathlib import Path

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Paths to the inputs produced by extract_thread.py
REP_PATH = Path("custom_persona/representative_threads.json")
ACTION_PATH = Path("custom_persona/action_stats.json")

# Output CSV path
OUT_CSV = Path("custom_persona/personas.csv")

# OpenAI model config (adjust if using a different model/provider)
MODEL_TYPE = ModelType.GPT_4O_MINI


def load_data():
    with REP_PATH.open() as f:
        reps = json.load(f)
    with ACTION_PATH.open() as f:
        actions = json.load(f)
    return reps, actions


def build_prompt(cluster_key: str, texts: list[str], action_stats: dict) -> str:
    pct = action_stats["percentages"]
    rep_snippets = "\n\n".join(texts[:5])  # limit for brevity
    prompt = f"""
You are generating a synthetic social-media persona for a cluster of users.

Cluster: {cluster_key}

Inputs:
- Representative posts (text) for {cluster_key}:
{rep_snippets}

- Action statistics (percentages over all sampled threads in this cluster):
  like: {pct['like']:.2f}%, unlike: {pct['unlike']:.2f}%, repost: {pct['repost']:.2f}%, unrepost: {pct['unrepost']:.2f}%,
  follow: {pct['follow']:.2f}%, unfollow: {pct['unfollow']:.2f}%, block: {pct['block']:.2f}%, unblock: {pct['unblock']:.2f}%,
  post_update: {pct['post_update']:.2f}%, post_delete: {pct['post_delete']:.2f}%, quote: {pct['quote']:.2f}%,
  post: {pct['post']:.2f}%, reply: {pct['reply']:.2f}%.

Task:
- Infer a plausible gender (“male”, “female”, or “unspecified”).
- Write a concise persona description (1–2 sentences) reflecting tone, interests, and style seen in the representative threads.
- Summarize behavior patterns by choosing one or two labels from this fixed set that best reflect the action stats: Post heavily, Comment heavily, Post Comment Mixed, Like Heavy. Only use these exact labels.

Output format:
Return exactly one CSV row with header:
gender,description,behavior_patterns

Example output (header + one row):
gender,description,behavior_patterns
female,"Tech-savvy commentator who shares curated clips and adds brisk opinions on industry news.","Post heavily; Like Heavy"

Now generate the CSV row for this cluster only:
"""
    return prompt.strip()


def generate_personas():
    reps, actions = load_data()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=MODEL_TYPE,
    )

    rows = [("gender", "description", "behavior_patterns")]
    for cluster_key, texts in reps.items():
        prompt = build_prompt(cluster_key, texts, actions[cluster_key])
        response = model.run(messages=[{
            "role": "user",
            "content": prompt
        }])
        # Extract text from response, trying multiple known attributes.
        text = ""
        if hasattr(response, "output_messages"):
            msgs = getattr(response, "output_messages", [])
            if msgs:
                text = getattr(msgs[0], "content", "") or getattr(
                    msgs[0], "text", "")
        # Try OpenAI-style choices
        if not text and hasattr(response, "choices"):
            choices = getattr(response, "choices", [])
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg:
                    text = getattr(msg, "content", "") or getattr(
                        msg, "text", "")
        if not text:
            text = getattr(response, "text", "") or getattr(
                response, "content", "")

        if not text:
            print(f"[warn] Empty model response for {cluster_key}: {response}")

        # Expecting a two-line CSV (header + row); take the last non-empty line.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else ""
        # Basic CSV parsing of the returned line
        reader = csv.reader([last_line])
        for parsed in reader:
            if len(parsed) == 3:
                rows.append(tuple(parsed))
            else:
                # Fallback: stuff the whole line into description if parsing fails
                rows.append(("unspecified", last_line, ""))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Wrote personas to {OUT_CSV}")


if __name__ == "__main__":
    generate_personas()
