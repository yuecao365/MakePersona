from collections import defaultdict
import json
import random
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from datasets import load_dataset
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Total rows in the dataset (update automatically from ds.num_rows).
TOTAL_ROWS = None
CLUSTERS = 25
SAMPLES_PER_CLUSTER = 2500
TOTAL_SAMPLES = CLUSTERS * SAMPLES_PER_CLUSTER


def sample_from_dataset() -> dict[int, list[Mapping]]:
    """Sample rows across the full dataset and bucket by cluster_id."""
    random.seed(42)  # make sampling reproducible

    ds = load_dataset(
        "ComplexDataLab/BluePrint",
        "25_clusters",
        split="full",
        cache_dir="data/hf_cache",
    )

    total_rows = ds.num_rows if TOTAL_ROWS is None else TOTAL_ROWS
    if total_rows < TOTAL_SAMPLES:
        raise ValueError(
            f"Dataset smaller ({total_rows}) than requested samples ({TOTAL_SAMPLES}).")

    target_indices = sorted(random.sample(range(total_rows), TOTAL_SAMPLES))

    samples: dict[int, list[Mapping]] = defaultdict(list)
    for idx in target_indices:
        row = ds[idx]
        cid = row["cluster_id"]  # 0-based
        samples[cid].append(row)

    return samples


def _thread_to_text(row: Mapping) -> str:
    """Concatenate all messages in a thread into a single text blob."""
    parts = []
    for msg in row.get("thread", []):
        txt = msg.get("text", "")
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


def _has_authored_text(row: Mapping) -> bool:
    """Keep threads where the user posted/quoted/replied with text."""
    for msg in row.get("thread", []):
        actions = msg.get("actions") or {}
        if actions.get("post") or actions.get("quote") or actions.get("reply"):
            txt = msg.get("text")
            if txt:
                return True
    return False


def get_representative_threads(
    samples: Mapping[int, Sequence[Mapping]],
    embed_fn: Callable[[Iterable[str]], Sequence[Sequence[float]]],
    k: int = 20,
) -> dict[int, list[Mapping]]:
    """
    For each cluster_id in samples:
    1) Embed all sampled threads.
    2) Run k-means with k clusters.
    3) Pick the thread closest to each centroid.
    """
    representatives: dict[int, list[Mapping]] = {}

    for cid, rows in samples.items():
        if not rows:
            representatives[cid] = []
            continue

        # Filter to authored threads with text; fall back to all if none match.
        filtered_rows = [row for row in rows if _has_authored_text(row)]
        rows_for_kmeans = filtered_rows if filtered_rows else rows

        texts = [_thread_to_text(row) for row in rows_for_kmeans]
        texts = [t for t in texts if t]  # drop empty blobs
        if not texts:
            representatives[cid] = []
            continue

        rows_for_kmeans = [row for row, t in zip(rows_for_kmeans, texts) if t]

        embeddings = np.array(list(embed_fn(texts)), dtype=float)

        # If fewer samples than k, just return all.
        if len(rows_for_kmeans) <= k:
            representatives[cid] = list(rows_for_kmeans)
            continue

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        kmeans.fit(embeddings)
        centers = kmeans.cluster_centers_

        reps: list[Mapping] = []
        for c in centers:
            dists = np.linalg.norm(embeddings - c, axis=1)
            nearest_idx = int(np.argmin(dists))
            reps.append(rows_for_kmeans[nearest_idx])
        representatives[cid] = reps

    return representatives


def main():
    samples = sample_from_dataset()

    # Sentence-Transformers embedder (CPU by default; set device="cuda" if available).
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def embed_fn(texts: Iterable[str]) -> np.ndarray:
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

    reps = get_representative_threads(samples, embed_fn=embed_fn, k=20)
    print({cid: len(rows) for cid, rows in reps.items()})

    # Save representative thread texts and raw rows.
    rep_texts = {
        f"cluster{cid + 1}": [_thread_to_text(row) for row in rows]
        for cid, rows in reps.items()
    }
    rep_rows = {f"cluster{cid + 1}": rows for cid, rows in reps.items()}

    with open("custom_persona/representative_threads.json", "w") as f:
        json.dump(rep_rows, f, indent=2)
    print("Wrote representative threads (raw) to custom_persona/representative_threads.json")

    with open("custom_persona/representative_threads_text.json", "w") as f:
        json.dump(rep_texts, f, indent=2)
    print("Wrote representative thread texts to custom_persona/representative_threads_text.json")

    # Compute action frequencies per cluster across all sampled threads.
    action_keys = [
        "like",
        "unlike",
        "repost",
        "unrepost",
        "follow",
        "unfollow",
        "block",
        "unblock",
        "post_update",
        "post_delete",
        "quote",
        "post",
        "reply",
    ]
    action_stats = {}
    for cid, rows in samples.items():
        counts = {k: 0 for k in action_keys}
        total = 0
        for row in rows:
            for msg in row.get("thread", []):
                actions = msg.get("actions") or {}
                for k in action_keys:
                    if actions.get(k):
                        counts[k] += 1
                        total += 1
        percentages = {
            k: (counts[k] / total * 100.0) if total else 0.0 for k in action_keys
        }
        action_stats[f"cluster{cid + 1}"] = {
            "counts": counts,
            "total_true_actions": total,
            "percentages": percentages,
        }

    stats_path = "custom_persona/action_stats.json"
    with open(stats_path, "w") as f:
        json.dump(action_stats, f, indent=2)
    print(f"Wrote action stats to {stats_path}")


if __name__ == "__main__":
    main()
