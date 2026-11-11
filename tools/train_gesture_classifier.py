#!/usr/bin/env python3
"""Train an inspectable one-vs-rest landmark classifier without dependencies."""

import json
import math
import random
import sys
from pathlib import Path


LABELS = ("pinch", "claw", "open")


def sigmoid(value):
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def train(rows, label, epochs=500, rate=0.08, regularization=0.001):
    width = len(rows[0]["x"])
    weights = [0.0] * (width + 1)
    for _ in range(epochs):
        random.shuffle(rows)
        for row in rows:
            target = 1.0 if row["label"] == label else 0.0
            score = weights[-1] + sum(w * x for w, x in zip(weights, row["x"]))
            error = sigmoid(score) - target
            for index, value in enumerate(row["x"]):
                weights[index] -= rate * (error * value + regularization * weights[index])
            weights[-1] -= rate * error
    return {"weights": weights[:-1], "bias": weights[-1]}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: train_gesture_classifier.py DATASET.json MODEL.json")
    source, target = map(Path, sys.argv[1:])
    dataset = json.loads(source.read_text(encoding="utf-8"))
    if dataset.get("schema") != "barehands-landmarks/v1":
        raise SystemExit("unsupported dataset schema")
    feature_names = dataset.get("featureNames", [])
    frames = [frame for frame in dataset.get("frames", []) if frame.get("features")]
    if not frames:
        raise SystemExit("dataset contains no feature frames")

    means = [sum(float(row["features"].get(name, 0)) for row in frames) / len(frames)
             for name in feature_names]
    deviations = []
    for index, name in enumerate(feature_names):
        variance = sum((float(row["features"].get(name, 0)) - means[index]) ** 2
                       for row in frames) / len(frames)
        deviations.append(max(math.sqrt(variance), 1e-6))
    rows = [{
        "label": frame.get("label", "relaxed"),
        "x": [(float(frame["features"].get(name, 0)) - means[index]) /
              deviations[index] for index, name in enumerate(feature_names)]
    } for frame in frames]

    users = {str(frame.get("userId", dataset.get("metadata", {}).get("userId", "")))
             for frame in frames}
    if len(users - {""}) < 2:
        print("warning: evaluate on recordings from entirely unseen users before deployment",
              file=sys.stderr)
    random.seed(42)
    model = {
        "schema": "barehands-landmark-classifier/v1",
        "trainedAt": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "featureNames": feature_names,
        "normalization": {"mean": means, "std": deviations},
        "classifiers": {label: train(rows, label) for label in LABELS},
        "productionReady": False,
        "note": "Run in shadow mode and validate on held-out users before enabling."
    }
    target.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} from {len(rows)} frames")


if __name__ == "__main__":
    main()

