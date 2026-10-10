# PAR_UPAR — Shift-robust calibration and deployment

Goal: improve **hidden-domain robustness** without contaminating the private test or overwriting the prior Prototype checkpoint. There is **no guarantee** of scoring >0.78; the real Codabench result must be measured.

## What changed

- `calibrate_shift_robust.py` (new): one-shot, fit-only search on stable 70% source validation; locked 30% audit compares exactly the fit-selected challenger against the original Prototype anchor. Balanced BCE training/checkpoints remain intact; student checkpoints are optional.
- Source-domain-balanced threshold regularization (pooled + median-of-domain BA thresholds) with gentle shrinkage; label mask excludes unknown annotations.
- Candidate pool: Prototype checkpoint seed42/123 mixtures, conservative blends with any existing Contrastive student, and Flip or Dual-Scale Flip TTA.
- `submission_generalization.py` (modified): implements `dual_scale_flip` identically to calibration (interpolate the normalized 288×144 tensor to 320×160, evaluate original + mirrored at both resolutions, average logits). Retains plain Flip if selected.
- `export_generalization.py` (modified): passes through the chosen TTA type and dual-scale dimensions; packages an isolated ZIP; automatically imports it and tests `load_model`, `predict_image`, `predict_batch` with 40 outputs.
- `prototype_par.py` (included): architecture used by saved checkpoints, matching the patch from the uploaded notebook.
- Other existing trainer and audit modules included to keep compatibility. **Do not delete/retrain original Prototype checkpoints.**

## Inputs and outputs

- Requires official dataset in `/content/UPAR-Challenge-2027/data`, original pretrained checkpoints `Checkpoints/prototype_convnext_tiny_seed42_best.pt`, `seed123_best.pt`.
- Optionally reads `Checkpoints/DomainContrastiveTraining_SupCon/prototype_contrastive_convnext_tiny_seed{42,123}_best.pt`.
- Optionally reads `Checkpoints/prototype_ensemble_calibration.pt` to preserve its exact original anchor thresholds.
- Saves prediction caches, calibration and locked-audit report to `Results/ShiftRobust/`.
- Exports `Results/Submission_UPAR_ShiftRobust.zip` and unpacked `Results/Submission_UPAR_ShiftRobust/`.

## Experimental discipline

- Source validation is **not** true unseen-domain validation if the checkpoint already saw those domains while training. It is a useful diagnostic, not evidence that the hidden Codabench metric will improve.
- Audit is meant as one locked check. Avoid repeatedly changing hyperparameters based on the audit result.
- If the chosen challenger fails, the script **reverts to original Prototype**; it does not force SupCon weights or uncalibrated thresholds.
- Check competition rules for model count/ensembles and pretrained weights prior to submission.
- Original GitHub could not be fetched from the authoring container; this source patch is based on the uploaded Colab's embedded source and available accompanying source archive snapshot. On Google Colab the notebook clones the user-provided GitHub URL, then overlays these reviewed files.
