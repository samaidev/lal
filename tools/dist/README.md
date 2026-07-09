# LAL Distributed Inference

Pipeline parallelism for multi-CPU distributed inference using Q4 quantization.

## Architecture

```
Node 1 (master)                    Node 2 (worker)
┌─────────────────┐               ┌─────────────────┐
│ Embedding       │               │                 │
│ Layer 0-5 (Q4)  │ ──768 floats→ │ Layer 6-11 (Q4) │
│                 │   3KB/token   │                 │
│ Tokenizer       │ ←─768 floats─ │ LM Head (int8)  │
└─────────────────┘               └─────────────────┘
```

- Q4 weights: 14MB total → ~7MB per node (half layers)
- Network transfer: 768 floats = 3KB per token (negligible on LAN)
- Each node only needs half the model in memory

## Usage

### Start worker (node 2)
```bash
./dist_worker --worker --layers 6-12 --port 8091
```

### Start master (node 1)
```bash
./dist_worker --master --layers 0-6 --port 8090
```

### Test latency
```bash
./dist_client --worker-url http://worker-ip:8091 --tokens 10
```

## Benchmark

Network round-trip (via aitun relay):
- 6KB/token transfer (768 floats × 2 directions)
- ~1s/token via relay (aitun relay adds latency)
- Expected <1ms/token on direct LAN (1Gbps)

Q4 advantage for distributed:
- float weights: 108MB → 54MB per node (slow to distribute)
- Q4 weights: 14MB → 7MB per node (fast to distribute)
- Hidden state transfer: 3KB (same for all modes)
