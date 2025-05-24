# Self-Attention 机制实现说明

## 1. 题目简介
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

## 3. 完整代码 (self_attention.py)

```python
import numpy as np

# ---------- ① 线性映射 ----------
def compute_qkv(X, W_q, W_k, W_v):
    """
    将输入 X 分别映射到 Query / Key / Value 子空间。
    参数
    ----
    X   : (seq_len, d_model)
    W_q : (d_model, d_k)
    W_k : (d_model, d_k)
    W_v : (d_model, d_v)
    返回
    ----
    Q   : (seq_len, d_k)
    K   : (seq_len, d_k)
    V   : (seq_len, d_v)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


# ---------- ②~⑤ Scaled Dot-Product Attention ----------
def self_attention(Q, K, V):
    """
    Scaled Dot-Product Self-Attention.
    公式:
        scores  = (Q @ Kᵀ) / √d_k
        weights = softmax(scores)
        output  = weights @ V
    """
    d_k = Q.shape[-1]

    # ② 相似度分数
    scores = Q @ K.T                       # shape: (seq_len, seq_len)

    # ③ 缩放
    scores /= np.sqrt(d_k)

    # ④ Softmax (数值稳定处理)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # ⑤ 加权求和
    output = weights @ V                   # shape: (seq_len, d_v)
    return output


# ---------- 运行示例 ----------
if __name__ == "__main__":
    X   = np.array([[1, 0], [0, 1]])
    W_q = np.eye(2)
    W_k = np.eye(2)
    W_v = np.array([[1, 2], [3, 4]])

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    out = self_attention(Q, K, V)
    print(np.round(out, 6))
    # [[1.660477 2.660477]
    #  [2.339523 3.339523]]
