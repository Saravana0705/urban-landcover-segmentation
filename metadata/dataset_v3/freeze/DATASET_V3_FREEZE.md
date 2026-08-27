# Dataset V3 Freeze Record

- Status: **FROZEN**
- Frozen at: **2026-08-27T02:12:11.736213+00:00**
- Cities: **20**
- Tiles: **1,664**
- Split: **974 train / 351 validation / 339 test**

## Scientific controls

- City-wise 14/3/3 geographic split preserved.
- Validation/test selection used coverage only, not class composition.
- Unknown pixels remain label 255 and are excluded from loss/metrics.
- Normalization inherits unchanged full training-city SAR statistics after SHA-256 verification.
- Automated tile QA passed for all 1,664 tiles.
- Targeted manual visual QA passed for 16 tiles and two conflict overlays.

## Mandatory checks

- `manifest_has_1664_unique_tiles`: **True**
- `split_tile_counts_match`: **True**
- `split_city_counts_match`: **True**
- `city_splits_disjoint`: **True**
- `all_tile_qa_passed`: **True**
- `tile_qa_manifest_hash_matches`: **True**
- `materialization_passed`: **True**
- `selection_passed`: **True**
- `normalization_provenance_passed`: **True**
- `normalization_excludes_validation_and_test`: **True**
- `normalization_copy_hash_matches`: **True**
- `manual_visual_qa_passed`: **True**
- `class_pixels_reconcile_to_valid_pixels`: **True**
- `all_manifest_rows_pass_qa`: **True**
