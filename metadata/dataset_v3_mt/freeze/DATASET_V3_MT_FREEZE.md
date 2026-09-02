# Dataset V3-MT Freeze Record

- Status: **FROZEN**
- Frozen at: **2026-09-02T00:06:39.809367+00:00**
- Input: **8 quarterly Sentinel-1 VV/VH bands**
- Tiles: **1,664**
- Split: **974 train / 351 validation / 339 test**
- Labels and city split: **unchanged from Dataset V3**
- Independent tile QA: **1,664/1,664 PASS**

## Mandatory checks

- `manifest_has_1664_unique_tiles`: **True**
- `manifest_hash_is_authoritative`: **True**
- `split_tile_counts_match`: **True**
- `split_city_counts_match`: **True**
- `city_splits_disjoint`: **True**
- `training_city_ids_match`: **True**
- `evaluation_city_ids_match`: **True**
- `tile_qa_passed`: **True**
- `tile_qa_manifest_hash_matches`: **True**
- `tile_qa_split_counts_match`: **True**
- `materialization_passed`: **True**
- `materialization_manifest_hash_matches`: **True**
- `materialization_split_counts_match`: **True**
- `source_qa_passed`: **True**
- `source_qa_hash_matches_materialization`: **True**
- `normalization_passed`: **True**
- `normalization_hash_matches_materialization`: **True**
- `normalization_provenance_passed`: **True**
- `normalization_provenance_hash_matches_materialization`: **True**
- `class_pixels_reconcile_to_valid_pixels`: **True**
- `all_manifest_rows_pass_qa`: **True**
- `all_manifest_rows_materialized`: **True**
- `all_image_paths_are_v3_mt`: **True**
- `legacy_dataset_artifacts_unmodified`: **True**
