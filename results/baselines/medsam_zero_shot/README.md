# MedSAM zero-shot baseline

## Status
Completed inference-only evaluation. This is not an ACDC fine-tuned model.

## Protocol
- Pretrained MedSAM ViT-B checkpoint; weights frozen.
- No ACDC optimization or fine-tuning.
- Fixed class-specific box prompts derived from training-patient masks only.
- Locked test cohort: 20 patients, 398 slices.
- Test masks were used for evaluation, not prompt construction or tuning.
- HD95 used the fallback implementation because `medpy` was unavailable.
- The original JSON is preserved alongside this summary.

## Results
See `summary.csv` for the reported aggregate metrics and
`medsam_zeroshot_metrics.json` for the full original output.

## Important
Keep this result separate from `MedSAM fine-tuned on ACDC`.
Do not overwrite this artifact when running fine-tuning.
