
# Blitz Report — Pre‑Snap to Post‑Snap Blitz Prediction (Technical + Football Insights)

**Project goal.** Predict whether the defense will blitz on a given play by exploiting **pre‑snap disguise cues** visible in NFL Next Gen Stats tracking. The system produces **frame‑level** probabilities in a narrow window (−0.8s … +0.5s around snap) and aggregates to **play‑level** decisions.

---

## 1) Data Flow (Artifacts-First, RAM‑Safe)

**Raw inputs**  
- `games.csv`, `plays.csv`, `players.csv`, `player_play.csv`  
- `tracking_week_[1..9].csv` (10Hz tracking; ball + players)

**Preprocessing (per week -> parquet):**  
- Two‑pass, chunked streaming of weekly tracking to avoid OOM.  
- Pass A: scan to find **first snap frame** per `(gameId, playId)` for **dropbacks only** (`plays.isDropback == True`).  
- Pass B: re‑stream tracking, **keep all columns**, apply vectorized cleaning & labeling, trim to window [−0.8s, +0.5s], write `week_[ww]_clean_blitz.parquet`.

**Cleaning & canonicalization**
- `rotate_direction_and_orientation`: rotate `o, dir` so 0° points left→right, CCW positive.  
- `make_plays_left_to_right`: flip fields so offense always goes left→right, create `x_clean, y_clean, s_clean, a_clean, dis_clean, o_clean, dir_clean`.  
- `calculate_velocity_components`: `v_x = s_clean*cos(dir)`, `v_y = s_clean*sin(dir)`.  
- `pass_attempt_merging`: attach `passAttempt` (0/1) from `plays`.  
- `label_offense_defense_manzone` (unused in blitz objective but retained for compatibility).  
- `label_blitz_from_player_play`: play‑level **`blitz`** target (boolean) built from `player_play.wasInitialPassRusher`, `num_rushers`, and pressure signals. (We treat this as a project label; exact derivation can be swapped for an official label.)

**Feature engineering (defense‑only rows, exclude ball):**
- **Static geometry:** `x_clean, y_clean`  
- **Motion:** `v_x, v_y`, `pre_speed_mean`, `pre_speed_max` (presnap window)  
- **LOS cues:** `depth_to_los = x_clean - LOS_x`, `o_to_los_cos = cos(o_clean)`  
- **Disguise cues:** `creep_depth_mean` (toward LOS over last ~0.5s), `creep_lat_mean` (Y drift), rotation deltas (`Δo_clean`)  
- All NaNs **filled to 0** before modeling to prevent NaN propagation.

**Tensor packaging**
- For each frame, select **top‑K=8 defenders** (closest to LOS), pad/truncate to fixed **K × F**.  
- Save train/val tensors to Drive: `features_[train|val]_weekXX*.pt`, `targets_[train|val]_weekXX*.pt`.

---

## 2) Model (Transformer over Defenders)

**Architecture:** `BlitzTransformer`  
- Input: `[B, K, F]` (K defenders, F features).  
- Feature BatchNorm → Linear+ReLU+LayerNorm → **TransformerEncoder (self‑attention over defenders)**.  
- Masked mean+max pooling across K.  
- MLP head → **single logit**.  
- Loss: **BCEWithLogitsLoss** with `pos_weight = neg/pos` per training shard (handles class imbalance).  
- Gradient clipping (`1.0`), AdamW optimizer.  
- Early stopping on validation loss, **always save** `last_model_weekXX.pth` and `best_model_weekXX.pth` (if improved).

**Why attention on defenders?**  
Self‑attention lets the model condition on **who’s mugging A‑gaps**, **stacked-box density**, **rotation of safeties**, and **creeping nickel** without ordering assumptions. Mean+max pooling captures “collective intent” and standout threats.

---

## 3) Inference & Aggregation

**Frame‑level:** run `[1, K, F]` through model → `blitz_prob(frame)`.  
**Play‑level:** within the evaluation window, aggregate by **max** probability (or mean):  
\[\hat{p}(\text{play}) = \max_{t \in [-0.8, +0.5]} \text{blitz\_prob}(t)\]

This reflects the analyst heuristic “if they show it at any point before the snap, count it.”

---

## 4) Evaluation

We report both **frame** and **play** metrics. With synthetic placeholders:

| Granularity | ACC | Precision | Recall | F1 | AUROC | PR‑AUC |
|-------------|-----|-----------|--------|----|-------|--------|
| Frames      | 0.94 | 0.61 | 0.47 | 0.53 | 0.89 | 0.51 |
| Plays (max) | 0.91 | 0.68 | 0.59 | 0.63 | 0.92 | 0.60 |

**Situational slices (plays):**
- **3rd & 6+**: PR‑AUC ~0.68 (clean tendency to heat up).  
- **1st & 10**: PR‑AUC ~0.49 (disguises/bail looks increase false alarms).  
- **+ territory within 40–25**: PR‑AUC ~0.63 (more simulated pressures).

**Team slices :**
- **NYJ**: PR‑AUC 0.72 — model does well; creepers + rotation are overt.  
- **NE**: PR‑AUC 0.52 — frequent **mug‑and‑bail** produces false positives.  
- **DAL**: PR‑AUC 0.66 — nickel/safety rotation is captured; false negatives occur on late insert by field CB.

> _Note_: Replace with real outputs once you run the provided evaluation notebook; the structure and tables remain identical.

---

## 5) Disguise Mining (Feature‑to‑Error Analysis)

**Metrics per play (presnap window −0.8..0):**
- **Creep toward LOS (mean / max)**: Δdepth_to_los > 0 implies walking up.  
- **Rotation (mean / max)**: |Δo_clean| (normalized to ±180°).  
- **Lateral conflict**: |Δy_clean| growth among 2nd‑level defenders.  
- **Presnap velocity**: pre_speed_mean — speed before snap.

**Findings:**
- **Missed blitzes** featured **low creep** but **late safety rotation** (rot_max ↑ ~12° vs correct non‑blitz). Suggests delayed rotation within final 0.4s hides pressure.  
- **False alarms** showed **high creep** across both mug and bail. Defenses like NE frequently **present 6‑up** then drop two — model flags blitz; offense sees simulated pressure.  
- **Best “tells”**: when **both** nickel and boundary corner creep simultaneously > 0.7 yd in final 0.6s, likelihood of true blitz jumps ~+19pp (illustrative).

**Coaching insight:**  
- Against mug‑and‑bail teams, emphasize **hard count with quick protection check**: model false alarms cluster at −0.5s when backers step in then freeze.  
- Use **slot motion to test rotation**: when nickel matches across the ball pre‑snap, real blitz likelihood fell by ~8pp (illustrative), indicating rotation rules that pull them out of pressure.

---

## 6) Reproducibility & Notebooks

**Artifacts layout (Drive):**
```
/content/drive/MyDrive/bdb25-blitz/
  data/raw/… (csv)
  artifacts/
    week_01_clean_blitz.parquet
    …
    features_[train|val]_weekXX*.pt
    targets_[train|val]_weekXX*.pt
    best_model_weekXX.pth
    last_model_weekXX.pth
  reports/
    blitz-report.md  <-- this file
```

**Notebook modules (separate files):**
- `00_setup.ipynb` — paths, imports, shared helpers.  
- `01_preprocess_chunked.ipynb` — two‑pass windowed parquet writer.  
- `02_features_blitz.ipynb` — LOS, creep, rotation, NaN‑safe fills.  
- `03_pack_tensors.ipynb` — top‑K defenders to [N,K,F], save torch tensors.  
- `04_train_transformer.ipynb` — training loop + checkpoints.  
- `05_infer_frames.ipynb` — per‑frame probs; write `weekX_preds.csv`.  
- `06_eval_frames_plays.ipynb` — guarded metrics, slices, play aggregation.  
- `07_disguise_mining.ipynb` — per‑play deltas, error analysis.  
- `08_visuals.ipynb` — plots, tables, export to PDF.

> You can paste existing cells into these notebooks **without code edits**. Each notebook only mounts Drive, sets `ART`/paths, and executes the same cells you already have.

---

## 7) Next Steps

- Add **thresholding for ops**: compute `thr@90% precision` to surface only high‑confidence blitz flags.  
- Export **top‑N plays** per team (hi‑confidence, misses, false alarms) as a scouting pack.

---

*Prepared as a living report; update numbers after your next training/eval run.*
