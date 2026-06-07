---
license: apache-2.0
tags:
- robotics
- libero
- vision-language-action
---

# LaRA-VLA LIBERO Model

This model is trained on the LIBERO benchmark for multi-stage robotic manipulation tasks.

## Model Details

- **Architecture**: DIT-B
- **Learning Rate**: 1E-4 → 1E-5
- **Batch Size**: 14
- **Training Steps**: 40K
- **Configuration**: 1lr

## Files

- `model.pt`: Model checkpoint (9.7GB)
- `config.json`: Model configuration
- `config.yaml`: Training configuration
- `dataset_statistics.json`: Dataset statistics
- `summary.jsonl`: Training summary

## Usage

```python
import torch

# Load the model
model = torch.load('model.pt')
```

## Citation

If you use this model, please cite the relevant papers.
