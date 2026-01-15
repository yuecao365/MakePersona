"""
Generate a synthetic CSV similar to False_Business_0.csv using the LLM personas.

Inputs:
- custom_persona/personas.csv (gender, description, behavior_patterns)
- custom_persona/representative_threads.json (for follow hints)

Output:
- data/twitter_dataset/anonymous_topic_200_1h/False_Business_llm.csv

Schema:
user_id,name,username,following_agentid_list,previous_tweets,user_char,description
"""

import csv
import json
import random
from pathlib import Path

import pandas as pd

PERSONA_CSV = Path("custom_persona/personas.csv")
REP_THREADS = Path("custom_persona/representative_threads.json")
OUT_CSV = Path("data/twitter_dataset/anonymous_topic_200_1h/False_Business_llm.csv")


def load_representative_user_ids():
    """
    Extract user_ids from representative_threads.json per cluster key.
    Returns: dict cluster_key -> list of user_ids (as strings)
    """
    with REP_THREADS.open() as f:
        data = json.load(f)
    rep_user_ids = {}
    for cluster_key, threads in data.items():
        ids = []
        for t in threads:
            # Each thread is a list of message dicts with user_id
            if isinstance(t, list):
                for msg in t:
                    uid = msg.get("user_id")
                    if uid:
                        ids.append(str(uid))
        # de-duplicate while preserving order
        seen = set()
        uniq_ids = []
        for uid in ids:
            if uid not in seen:
                seen.add(uid)
                uniq_ids.append(uid)
        rep_user_ids[cluster_key] = uniq_ids
    return rep_user_ids


def main():
    random.seed(42)
    personas = pd.read_csv(PERSONA_CSV)
    rep_user_ids = load_representative_user_ids()

    rows = []
    starter_idx = 0  # only this persona will have a seed post

    for idx, row in personas.iterrows():
        cluster_key = f"cluster{idx + 1}"
        user_id = idx  # align user_id with row index / agent_id
        name = f"Persona {idx + 1}"
        username = f"persona_{idx + 1}"

        # Following list: mix rep-based ids (if any) plus random persona ids.
        follows_from_reps = rep_user_ids.get(cluster_key, [])
        # Add up to 3 random other personas (by index) to diversify the graph.
        candidate_idxs = [i for i in range(len(personas)) if i != idx]
        rand_follows = random.sample(candidate_idxs,
                                     k=random.randint(0, min(3, len(candidate_idxs))))
        # Combine and dedupe while preserving order
        combined = follows_from_reps + [str(i) for i in rand_follows]
        # Ensure everyone follows the starter user.
        if idx != starter_idx:
            combined.insert(0, str(starter_idx))
        seen = set()
        following_agentid_list = []
        for fid in combined:
            if fid not in seen:
                seen.add(fid)
                following_agentid_list.append(fid)

        # Only one starter post for the starter_idx persona; others empty.
        if idx == starter_idx:
            seed_post = ("Discussing the 2025 Canadian Federal Election: "
                         "expectations, concerns, and hopes for the outcome.")
            previous_tweets = [seed_post]
        else:
            previous_tweets = []
        user_char = f"{row['gender']}; {row['description']}; Behavior Patterns: {row['behavior_patterns']}"
        display_description = row["description"]

        rows.append({
            "user_id": user_id,
            "name": name,
            "username": username,
            "following_agentid_list": str(following_agentid_list),
            "previous_tweets": str(previous_tweets),
            "user_char": user_char,
            "description": display_description,
        })

    # Write to CSV
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "name",
                "username",
                "following_agentid_list",
                "previous_tweets",
                "user_char",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote synthetic CSV to {OUT_CSV}")


if __name__ == "__main__":
    main()
