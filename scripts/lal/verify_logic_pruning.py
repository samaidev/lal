#!/usr/bin/env python3
"""验证核心思路：把 GPT-2 的一个真实权重矩阵用 LAL 逻辑原语表达，做阈值剪枝，看推理精度。

实验：
1. 取 GPT-2 第 0 层的 mlp.c_fc 权重 W [768, 3072]（真实训练权重）
2. 随机取一个输入向量 x [768]
3. 基准：y = x @ W  （标准矩阵乘，全精度）
4. LAL 表达：把 W 的每一列 j 看作一个 relate：
     relate fc_j(x) = dot(x, w_j) @ all_dims
   其中 w_j 是 W 的第 j 列。这把"矩阵乘"变成"3072 个 relate 调用"。
5. 逻辑剪枝：把 |w_j[i]| < threshold 的连接丢掉（弱关系），只保留强关系
   剪枝后 relate 变成稀疏的：只 dot 保留的维度
6. 对比：基准 y vs 剪枝后 y_pruned 的误差
7. 测量：剪枝掉多少权重、误差多大

如果这一步证明"逻辑剪枝后精度损失可接受"，那整个 GPT-2 都可以这么编译。
"""
import numpy as np

def main():
    from transformers import GPT2Model
    import torch

    print("[*] 加载 GPT-2 真实权重...")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()

    # 取第 0 层 MLP 的 c_fc 权重 [768, 3072]
    W = sd["h.0.mlp.c_fc.weight"].numpy().astype(np.float32)  # [768, 3072]
    b = sd["h.0.mlp.c_fc.bias"].numpy().astype(np.float32)    # [3072]
    print(f"[*] W shape: {W.shape}, 总参数: {W.size}")

    # 随机输入（模拟上一层输出）
    np.random.seed(42)
    x = np.random.randn(768).astype(np.float32)

    # === 基准：标准矩阵乘 ===
    y_ref = x @ W + b  # [3072]

    # === LAL 表达 + 剪枝 ===
    # 把 W 的每一列 j 看作一个 relate：y[j] = dot(x, W[:,j]) + b[j]
    # 剪枝：对每个 j，只保留 |W[i,j]| > threshold 的维度 i
    print("\n[*] 逻辑剪枝实验（把弱连接当冗余丢掉）:")
    print(f"{'阈值':>8} | {'保留连接数':>12} | {'剪枝率':>8} | {'最大误差':>10} | {'平均误差':>10} | {'相对误差':>10}")
    print("-" * 80)

    for threshold in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        # 对每个输出维度 j，找出 |W[i,j]| > threshold 的输入维度 i
        # 这就是 LAL relate 的 bound：只 dot 这些维度
        abs_W = np.abs(W)
        mask = abs_W > threshold  # [768, 3072], True = 保留

        n_kept = mask.sum()
        n_total = W.size
        prune_ratio = 1.0 - n_kept / n_total

        # 剪枝后的推理：y_pruned[j] = sum(x[i] * W[i,j] for i where mask[i,j]) + b[j]
        # 用 mask 把弱连接置零
        W_pruned = W * mask
        y_pruned = x @ W_pruned + b

        # 误差
        max_err = np.max(np.abs(y_ref - y_pruned))
        mean_err = np.mean(np.abs(y_ref - y_pruned))
        # 相对误差（相对于 y_ref 的范围）
        rel_err = mean_err / (np.max(np.abs(y_ref)) + 1e-9)

        print(f"{threshold:>8.3f} | {n_kept:>12,} | {prune_ratio:>7.1%} | {max_err:>10.4f} | {mean_err:>10.4f} | {rel_err:>10.4f}")

    # === 验证 GELU 之后误差是否放大 ===
    print("\n[*] 经过 GELU 激活后的误差（看剪枝是否在实际推理中可接受）:")
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))

    y_ref_gelu = gelu(y_ref)

    print(f"{'阈值':>8} | {'剪枝率':>8} | {'GELU后最大误差':>14} | {'GELU后平均误差':>14} | {'top-1是否改变':>14}")
    print("-" * 80)
    ref_top1 = np.argmax(y_ref_gelu)
    for threshold in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]:
        abs_W = np.abs(W)
        mask = abs_W > threshold
        W_pruned = W * mask
        y_pruned = x @ W_pruned + b
        y_pruned_gelu = gelu(y_pruned)
        max_err = np.max(np.abs(y_ref_gelu - y_pruned_gelu))
        mean_err = np.mean(np.abs(y_ref_gelu - y_pruned_gelu))
        prune_ratio = 1.0 - mask.sum() / W.size
        top1_changed = "是" if np.argmax(y_pruned_gelu) != ref_top1 else "否"
        print(f"{threshold:>8.3f} | {prune_ratio:>7.1%} | {max_err:>14.4f} | {mean_err:>14.4f} | {top1_changed:>14}")

    print("\n[*] 结论:")
    print("    - 如果阈值 0.05 能剪掉 50%+ 连接且 top-1 不变，说明 LAL 逻辑剪枝可行")
    print("    - 这就是把'不透明权重'变成'稀疏逻辑关系'的核心收益")

if __name__ == "__main__":
    main()
