# Interpretability & Demographic Evaluation

This branch packages the scripts I used to audit the fusion (genomic + 2.5D MRI) model across demographic cohorts and to prepare figures for sharing.

## Included scripts

| Script | Purpose |
| --- | --- |
| `scripts/make_demographic_splits.py` | Filter a clinical CSV into male/female and young/old cohorts. |
| `scripts/eval_radio_2p5d.py` | Evaluate a checkpoint on any CSV subset and emit metrics/predictions/calibration tables. |
| `scripts/plot_demographic_eval.py` | Turn the evaluation CSV/JSON outputs into bar charts, calibration curves, and risk-distribution plots. |
| `scripts/eval_splits.py` | Reload the original split CSVs and compute fold-by-fold c-index to confirm there is no leakage. |

See `requirements.txt` for the Python dependencies needed to run the tooling inside a clean environment (`conda create -n mm-xai --file requirements.txt` works on our cluster).

## Re-running the demographic analysis

1. **Prepare the cohorts**  
   ```
   PYTHONPATH=. python scripts/make_demographic_splits.py \
       data/processed_tabular_data/mut_cna_177_patients.csv \
       --outdir data/demographic_splits
   ```
   This now also saves `*_age_distribution.png` in the output folder, showing the
   age histogram for the full cohort with a median line (the same cutoff used for
   young vs. old splits).

2. **Evaluate each subset** (example for females)  
   ```
   PYTHONPATH=. python scripts/eval_radio_2p5d.py \
       --mode genomic_radio_2.5D \
       --csv data/demographic_splits/mut_cna_177_patients_female.csv \
       --ckpt results/radio_2.5D_d3/best_model_genomic_radio_2.5D.pt \
       --results-dir results/demographic_eval \
       --tag female_subset | tee results/demographic_eval/female_subset.log
   ```
   Repeat for the male, young, and old CSVs by only changing `--csv` and `--tag`.

3. **Generate plots for sharing**  
   ```
   PYTHONPATH=. python scripts/plot_demographic_eval.py
   ```
   This now writes c-index and Brier bar charts, calibration curves, per-bin calibration
   gap grids, fairness gap bars, risk-score histograms, risk-coverage curves, and discrete
   risk-group bars under `results/demographic_eval/`.

4. **Quantify fairness metrics + coverage curves**  
   ```
   PYTHONPATH=. python scripts/analyze_demographic_metrics.py \
       --folder results/demographic_eval/gtf_radiomic
   ```
   This emits `demographic_summary.csv`, `fairness_metrics.csv`, and `risk_coverage.csv`
   inside the folder; rerun `plot_demographic_eval.py` afterward to add the new figures.

5. **Verify against the original cross-validation splits**  
   ```
   PYTHONPATH=. python scripts/eval_splits.py \
       --args-pkl results/radio_2.5D_d3/args_genomic_radio_2.5D_d3.pkl \
       --results-dir results/radio_2.5D_d3
   ```
   The script prints the c-index for each `model_fold_i.pt` and stores a summary in `cv_eval_metrics.json`.

## Using the alternative XAI model

The `xai` branch contains Ahmed’s scripts/config tweaks for the interpretability-friendly model (notably `models/Encoder/DeepRiskA.py` plus the existing launcher scripts). You can copy whatever files you need without switching branches:

```
git fetch origin xai
git checkout xai -- scripts/xattn.sh models/Encoder/DeepRiskA.py
```

After copying the relevant code onto this branch, drop the new weights that Ahmed shared into `results/xai_model/` (or another folder outside Git) and point `scripts/eval_radio_2p5d.py --ckpt` to that path. The `--mode` flag must match the model definition from the `xai` branch (e.g., `--mode genomic_radio_2.5D` or `--mode xai_radio_2.5D` depending on how the class is registered). Once evaluated, the rest of the workflow (plots, calibration, CV check) is identical.

> **Note:** Keep the large MRI/weights/results artifacts under `data/` or `results/` so that Git ignores them. Only the scripts, README, and `requirements.txt` are committed in this branch.
