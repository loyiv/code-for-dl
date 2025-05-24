

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
```


实现 Transformer 中的 **Self-Attention（自注意力）**。  
给定输入矩阵 `X` 和权重矩阵 `W_q`、`W_k`、`W_v`，先计算查询 `Q`、键 `K`、值 `V`，再完成 **Scaled Dot-Product Attention**，最终输出形状为 `(seq_len, d_v)` 的张量。

---

## 2. 解题思路
| 步骤 | 公式/操作 | 说明 |
|------|-----------|------|
| ① 线性映射 | `Q=XW_q`, `K=XW_k`, `V=XW_v` | 把同一输入投影到三个子空间 |
| ② 相似度分数 | `scores = Q @ K.T` | 点积衡量 Query 与 Key 的相关性 |
| ③ 缩放 | `scores /= √d_k` | 防止分数随维度增大而过大 |
| ④ Softmax | `weights = softmax(scores)` | 行级归一化得到注意力权重 |
| ⑤ 加权求和 | `output = weights @ V` | 用权重汇聚 Value 得到上下文表示 |

---

## 3. 完整代码,这里我用的是numpy

```python
import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V



def self_attention(Q, K, V):
    d_k = Q.shape[-1]# why 后面要➗d_k ？？（经常问到）


    scores = Q @ K.T
    scores /= np.sqrt(d_k)

    #  Softmax
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    output = weights @ V                  
    return output

