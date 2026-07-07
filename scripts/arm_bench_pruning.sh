#!/data/data/com.termux/files/usr/bin/bash
# benchmark_pruning.sh — test inference speed at different prune levels
cd ~/lal
mkdir -p ~/bench_results

for PRUNE in 0.0 0.3 0.5 0.7; do
    pkill -f gpt2_server 2>/dev/null
    sleep 1

    if [ "$PRUNE" = "0.0" ]; then
        nohup setsid ./prebuilt/gpt2_server 8080 > ~/bench_results/srv.log 2>&1 < /dev/null &
    else
        nohup setsid ./prebuilt/gpt2_server 8080 --prune-vocab $PRUNE > ~/bench_results/srv.log 2>&1 < /dev/null &
    fi
    disown
    SRV_PID=$!
    sleep 6

    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "prune=$PRUNE: SERVER DIED" > ~/bench_results/bench_${PRUNE}.txt
        cat ~/bench_results/srv.log >> ~/bench_results/bench_${PRUNE}.txt
        continue
    fi

    # Warmup
    curl -s --max-time 120 -X POST http://localhost:8080/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"warmup","n_tokens":3}' > /dev/null

    # Benchmark
    RESULT=$(curl -s --max-time 300 -X POST http://localhost:8080/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"The quick brown fox","n_tokens":10}')

    echo "prune=$PRUNE: $RESULT" > ~/bench_results/bench_${PRUNE}.txt
    cat ~/bench_results/bench_${PRUNE}.txt

    pkill -f gpt2_server 2>/dev/null
    sleep 2
done

echo "=== ALL DONE ===" > ~/bench_results/DONE
cat ~/bench_results/bench_*.txt
