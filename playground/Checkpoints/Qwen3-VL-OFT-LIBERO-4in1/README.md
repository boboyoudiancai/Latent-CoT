# LIBERO Evaluation Results

## Qwen3-VL-OFT-LIBERO-4in1 Evaluation Results

| Steps | libero_object | libero_spatial | libero_goal | libero_10 | Avg |
|-------|---------------|----------------|-------------|-----------|-----|
| 20000 | 0.97 | 0.984 | 0.97 | 0.874 | 0.9495 |
| 30000 | 0.994 | 0.982 | 0.97 | 0.916 | 0.9655 |
| 40000 | 0.998 | 0.99 | 0.968 | 0.928 | 0.9710 |
| 50000 | 0.996 | 0.99 | 0.986 | 0.948 | **0.9800** |

## 📈 Convergence Trends

**libero_object**: Rapid convergence, reaching peak performance of **0.998** at 40k steps  
**libero_spatial**: Stable high-level performance, all steps **>0.98**  
**libero_goal**: Continuous improvement, achieving best performance **0.986** at 50k steps  
**libero_10**: Most challenging task but consistently improving, from **0.874** to **0.948**

## 🎯 Optimal Configuration

- **Best step count**: 50k steps (average success rate **98.00%**)
- **Most challenging task**: libero_10 (average **91.65%**)
- **Easiest task**: libero_object (average **98.95%**)

## 🔍 Key Findings

1. **Stable convergence**: All datasets reached or approached optimal performance at 50k steps
2. **Generalization capability**: The model shows strong generalization ability, with excellent performance on 4-task joint training
3. **Challenging tasks**: libero_10 remains the most challenging task with room for improvement
4. **Training efficiency**: Stable performance between 40k-50k steps indicates no overfitting
5. **Overall performance**: Average success rate steadily increased from 94.95% at 20k steps to 98.00% at 50k steps