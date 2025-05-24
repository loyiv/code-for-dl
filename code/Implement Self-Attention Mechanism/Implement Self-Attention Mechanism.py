import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


def self_attention(Q, K, V):
    d_k = Q.shape[-1]

    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)

    # softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    attention_output = weights @ V  # (seq_len, d_v)
    return attention_output