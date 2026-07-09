# LAL Distributed Inference (Q4 Pipeline Parallelism)

Pipeline parallelism for multi-CPU distributed inference using Q4 quantization.

## Architecture

```
Node 1 (arm64, 8-core)           Node 2 (MacBook Air, 4-core)
┌──────────────────┐             ┌──────────────────┐
│ Embedding        │             │                  │
│ Layer 0-5 (Q4)   │ ─3KB/token→ │ Layer 6-11 (Q4)  │
│ Tokenizer        │             │                  │
│                  │ ←3KB/token─ │ LM Head (int8)   │
└──────────────────┘             └──────────────────┘
  ~7MB Q4 weights                  ~7MB Q4 weights
```

## Benchmark (arm64 + MacBook Air, WiFi LAN)

| Connection | Avg latency | Throughput |
|-----------|-------------|------------|
| Direct LAN (WiFi) | 150-170 ms/tok | ~6 tok/s |
| aitun relay | 480 ms/tok | 2.1 tok/s |
| Wired LAN (expected) | <1 ms/tok | 1000+ tok/s |

## Q4 advantage: 7MB per node (vs 54MB float)

## Usage
```bash
# Worker: ./dist_worker --worker --layers 6-12 --port 8090
# Master: ./dist_client --worker-url http://worker-ip:8090 --tokens 10
```
