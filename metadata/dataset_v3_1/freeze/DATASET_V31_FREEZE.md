# Dataset V3.1 Freeze Record

- Status: **FROZEN**
- Frozen at: **2026-08-31T22:40:24.875537+00:00**
- Cities: **20**
- Tiles: **1,754**
- Split: **1,072 train / 346 validation / 336 test**

## Scientific controls

- City-wise 14/3/3 geographic split preserved.
- V3.1 label policy applied identically to train, validation and test cities.
- Validation/test tile selection used coverage only, not class composition.
- Unknown and deliberately excluded labels remain 255 and are excluded from loss/metrics.
- Frozen V3 training-only normalization inherited after SAR SHA-256 verification.
- Automated QA passed all 1,754 tile groups (10,524 GeoTIFFs).
- Manual visual QA was not required because V3.1 adds no positive labels.

## Mandatory checks

- `manifest_has_1754_unique_tiles`: **True**
- `split_tile_counts_match`: **True**
- `split_city_counts_match`: **True**
- `city_splits_disjoint`: **True**
- `all_tile_qa_passed`: **True**
- `tile_qa_manifest_hash_matches`: **True**
- `tile_qa_counts_match_manifest`: **True**
- `materialization_passed`: **True**
- `materialization_split_counts_match`: **True**
- `selection_passed`: **True**
- `normalization_provenance_passed`: **True**
- `normalization_is_v31`: **True**
- `normalization_excludes_validation_and_test`: **True**
- `normalization_copy_hash_matches`: **True**
- `class_pixels_reconcile_to_valid_pixels`: **True**
- `all_manifest_rows_pass_qa`: **True**
- `manual_visual_qa_not_required`: **True**
- `legacy_dataset_artifacts_unmodified`: **True**
