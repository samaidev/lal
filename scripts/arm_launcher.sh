#!/data/data/com.termux/files/usr/bin/bash
# launcher.sh — start gpt2_server as a true daemon
# Uses double-fork + setsid to fully detach from parent shell
cd ~/lal
pkill -f gpt2_server 2>/dev/null
sleep 1

# Double-fork via nohup + setsid + disown, with all stdio redirected
nohup setsid ./prebuilt/gpt2_server 8080 > server.log 2>&1 < /dev/null &
DAEMON_PID=$!
disown $DAEMON_PID

# Wait a bit and check it's still alive
sleep 4
if kill -0 $DAEMON_PID 2>/dev/null; then
    echo "OK pid=$DAEMON_PID"
    echo "--- log ---"
    cat server.log
else
    echo "FAIL: server died"
    cat server.log
fi
