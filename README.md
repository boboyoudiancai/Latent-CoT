# LIBERO Spatial-CoT Annotations

This branch contains the Spatial-CoT annotations used by the LaRA-VLA
Spatial-CoT experiments. Files preserve the dataset-relative layout:

```text
<dataset>/annotations/episode_spatial_cot.jsonl
```

Each JSONL record contains an `episode_index`, `num_steps`, and a `steps`
mapping from step index to the spatial-relation annotation.

| Dataset | Episodes |
| --- | ---: |
| `libero_10_no_noops_1.0.0_lerobot` | 379 |
| `libero_goal_no_noops_1.0.0_lerobot` | 428 |
| `libero_object_no_noops_1.0.0_lerobot` | 454 |
| `libero_spatial_no_noops_1.0.0_lerobot` | 432 |
