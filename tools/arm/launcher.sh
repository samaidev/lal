#!/data/data/com.termux/files/usr/bin/bash
# launcher.sh — start gpt2_server as a true daemon on Termux/Android
#
# On Termux, processes started from `samcommand /exec` or `sshd` get killed
# when the parent shell exits, even with nohup+disown. The trick is to use
# `setsid` to start a new session, then redirect all stdio to /dev/null so
# there's no controlling terminal that could forward SIGHUP.
#
# Usage:
#   bash launcher.sh           # default port 8080
#   bash launcher.sh 9000      # custom port

set -e
cd "$(dirname "$0")/.."   # repo root

PORT="${1:-8080}"

# Kill any previous instance
pkill -f gpt2_server 2>/dev/null || true
sleep 1

# Start as detached daemon
nohup setsid ./prebuilt/gpt2_server "$PORT" > server.log 2>&1 < /dev/null &
DAEMON_PID=$!
disown "$DAEMON_PID"

# Wait for startup
sleep 4

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "OK pid=$DAEMON_PID port=$PORT"
    echo "--- log ---"
    cat server.log
    exit 0
else
    echo "FAIL: server died during startup"
    cat server.log
    exit 1
fi
