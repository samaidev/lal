# LAL — Logic-Assembly Language
#
# 项目根目录结构：
#
# lal/
# ├── README.md                 # 项目总览
# ├── Makefile                  # 顶层构建入口
# ├── .gitignore
# │
# ├── compiler/                 # LAL 编译器（Python）
# │   ├── lal.py                # 编译器核心：解析器 + AST + C 代码生成
#   └── __init__.py
# │
# ├── runtime/                  # C 运行时（推理 + 训练）
# │   ├── gpt2_runtime.c        # GPT-2 推理（float + binary）
# │   ├── gpt2_binary.c         # binary 权重加载 + XNOR+popcount matmul
# │   └── gpt2_train.c          # GPT-2 训练（全 popcount，3.6ms/step）
# │
# ├── demos/                    # LAL 示例程序
# │   ├── basic/                # 基础 demo
# │   │   ├── demo.lal          # 掩码分类器
# │   │   ├── syllogism.lal     # VSA 三段论推理
# │   │   ├── vsa_ops.lal       # VSA 算子
# │   │   ├── recursion_demo.lal # 递归
# │   │   ├── graph_traversal.lal # 图遍历 + if/else
# │   │   ├── guard_demo.lal    # 规则 guard
# │   │   └── pattern_match_demo.lal # 模式匹配
# │   ├── embedding/            # 嵌入相关
# │   │   ├── embed_demo.lal    # word2vec 加载
# │   │   ├── embed_300d.lal    # 300 维分类
# │   │   ├── embed_768d.lal    # 768 维 BERT 规模
# │   │   ├── sentiment_768d.lal # 情感分类
# │   │   ├── gpt2_demo.lal     # GPT-2 词分类
# │   │   ├── gpt2_analogy.lal  # king-man+woman=queen
# │   │   └── lal_train_demo.lal # LAL 原生训练
# │   └── data/                 # 嵌入数据文件
# │       ├── embeddings.txt
# │       ├── embeddings_300d.txt
# │       ├── embeddings_768d.txt
# │       ├── gpt2_embeddings.txt
# │       └── sentiment_embeddings.txt
# │
# ├── tools/                    # 工具脚本
# │   ├── export_gpt2_weights.py   # 导出 GPT-2 权重到 GPW2 格式
# │   ├── export_gpt2_full.py      # 导出权重 + BPE 分词器
# │   ├── export_gpt2_embeddings.py # 导出 GPT-2 token embedding
# │   ├── export_bert_embeddings.py # 导出 BERT embedding
# │   ├── train_binary_gpt2.py     # PyTorch STE 训练（对比用）
# │   ├── gen_embeddings.py        # 生成合成 embedding
# │   └── verify.py                # 验证脚本
# │
# └── docs/                     # 文档
#     └── DESIGN.md             # 设计文档
