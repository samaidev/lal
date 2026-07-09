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

- Q4 weights: 14MB total → 7MB per node (half layers)
- Network transfer: 768 floats = 3KB per token
- Each node only needs half the model in memory

## Benchmark (arm64 + MacBook Air, WiFi LAN)

| Connection | Avg latency | Throughput | Notes |
|-----------|-------------|------------|-------|
| Direct LAN (192.168.x.x) | 150-170 ms/token | ~6 tok/s | WiFi + router hop |
| Ping (ICMP) | 34-721 ms | — | WiFi unstable |
| aitun relay | 480 ms/token | 2.1 tok/s | Through aitun.cc server |

On wired LAN (same switch): expected <1ms/token → 1000+ tok/s network ceiling.
Compute (Q4, 49 tok/s single-node) would be the bottleneck, not network.

## Q4 advantage for distributed

| Quant | Total weights | Per node | Distribution time (1Gbps) | Hidden transfer |
|-------|--------------|----------|--------------------------|-----------------|
| float | 108 MB | 54 MB | ~430ms | 3 KB |
| Q8 | 27 MB | 13.5 MB | ~108ms | 3 KB |
| **Q4** | **14 MB** | **7 MB** | **~56ms** | 3 KB |

Q4 enables fast model distribution to edge nodes (7MB downloads in <1s on WiFi).

## Usage

### Build (on each node)
```bash
gcc -O2 -o dist_worker tools/dist/dist_worker.c -lm -lpthread
gcc -O2 -o dist_client tools/dist/dist_client.c -lm
```

### Start worker (node 2, e.g. MacBook)
```bash
./dist_worker --worker --layers 6-12 --port 8090
```

### Run from master (node 1, e.g. arm64)
```bash
./dist_client --worker-url http://192.168.0.12:8090 --tokens 10
```

### Health check
```bash
curl http://worker-ip:8090/health
```
