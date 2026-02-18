# NFL Big Data Bowl 2025: Blitz Prediction — Pre-Snap Disguise Detection with Transformers

Inspired by [SumerSports/SportsTrackingTransformer](https://github.com/SumerSports/SportsTrackingTransformer), which uses a transformer encoder-decoder to embed player and ball tracking into a shared spatial representation, this project adapts that philosophy toward a specific defensive problem: **can we detect a blitz before the snap?** 

---

## Problem

Defenses disguise pressure. A linebacker walks up to the line of scrimmage, freezes, and drops into coverage. A safety rotates late. A nickel corner creeps just enough to threaten the A-gap. Quarterbacks and offensive coordinators try to read these cues in real time — this project tries to do the same thing with pre-snap tracking data.

Given NFL Next Gen Stats 10Hz tracking in a narrow window around the snap (−0.8s to +0.5s), the goal is to produce a play-level probability that the defense is bringing more than four rushers to the quarterbacks.

---

## Approach

Pre-snap, we isolate the top-8 defenders nearest the line of scrimmage and extract features that capture disguise: depth to LOS, creep velocity (are they walking up?), orientation rotation deltas (are they turning to blitz?), and lateral drift among second-level defenders.

These per-defender feature vectors are fed into a **Transformer encoder** — self-attention lets the model reason about collective defensive intent (who's mugging, who's bailing, whether the safety rotation matches the linebacker creak) without assuming any fixed ordering of players.

Frame-level probabilities are aggregated to a play-level decision by taking the max across the evaluation window, reflecting the analyst heuristic that if a defense *shows* pressure at any point pre-snap, it counts.

---

## Data

NFL Big Data Bowl 2025 tracking data (`tracking_week_1..9.csv`), processed per week into cleaned parquets. Offense is normalized left-to-right. Blitz labels are derived from `player_play.wasInitialPassRusher`.

---

## Pipeline

| Notebook | Purpose |
|---|---|
| `01_preprocess_chunked` | Two-pass chunked streaming → parquet |
| `02_features_blitz` | LOS depth, creep, rotation features |
| `03_pack_tensors` | Top-K defenders → [N, K, F] tensors |
| `04_train_transformer` | Training loop, checkpoints |
| `05_infer_frames` | Per-frame probabilities |
| `06_eval_frames_plays` | Play-level aggregation + slices |
| `07_disguise_mining` | Feature-to-error analysis |
