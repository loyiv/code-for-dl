# 实现 Self-Attention 机制

## 题目描述

在本题中，需要你实现 Transformer 模型中的核心组件——**自注意力（Self-Attention）机制**。给定输入矩阵 \(X\) 及其对应的线性映射权重矩阵 \(W_q, W_k, W_v\)，你需要先计算查询（Q）、键（K）、值（V）三组矩阵，然后在 `self_attention` 函数中完成 scaled dot-product attention 的计算，最终返回形状为 \((seq\_len, d_v)\) 的输出矩阵。

```python
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(np.round(output, 6))
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
