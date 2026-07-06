#!/bin/bash
# push-to-github.sh — pushes the LAL repo to github.com/samaidev/lal
#
# *** IMPORTANT: SECURITY ***
# The token you pasted in chat is COMPROMISED. Before running this:
#   1. Go to https://github.com/settings/tokens
#   2. REVOKE the token starting with "ghp_D2yp..."
#   3. Generate a NEW token with repo scope
#   4. Use the new token below (or better: use `gh auth login`)
#
# This script does NOT use any token from chat. You provide a fresh one.

set -e

REPO_URL="https://github.com/samaidev/lal.git"
REMOTE_NAME="origin"

echo "=== LAL — push to GitHub ==="
echo
echo "Target: $REPO_URL"
echo
echo "Choose authentication method:"
echo "  1) gh auth login (RECOMMENDED — no token in shell history)"
echo "  2) Personal access token (you'll be prompted, not echoed)"
echo "  3) SSH (requires you've set up SSH keys with GitHub)"
echo
read -p "Method [1/2/3]: " method

case "$method" in
  1)
    echo "Running: gh auth login"
    gh auth login --git-protocol https
    git remote add "$REMOTE_NAME" "$REPO_URL" 2>/dev/null || git remote set-url "$REMOTE_NAME" "$REPO_URL"
    git push -u "$REMOTE_NAME" main
    ;;
  2)
    # Use credential helper — token is read via stdin, never logged
    read -s -p "Paste your NEW GitHub personal access token: " GH_TOKEN
    echo
    # Set the remote URL with the token embedded (will be stored in .git/config)
    # Better: use credential helper
    git remote add "$REMOTE_NAME" "$REPO_URL" 2>/dev/null || git remote set-url "$REMOTE_NAME" "$REPO_URL"
    echo "protocol=https
host=github.com
username=samaidev
password=$GH_TOKEN" | git credential-store store 2>/dev/null || true
    echo "Pushing..."
    GIT_ASKPASS=echo git push -u "$REMOTE_NAME" main
    # Clear the token from memory
    unset GH_TOKEN
    ;;
  3)
    SSH_URL="git@github.com:samaidev/lal.git"
    git remote add "$REMOTE_NAME" "$SSH_URL" 2>/dev/null || git remote set-url "$REMOTE_NAME" "$SSH_URL"
    git push -u "$REMOTE_NAME" main
    ;;
  *)
    echo "Invalid choice. Aborting."
    exit 1
    ;;
esac

echo
echo "=== Done ==="
echo "Your repo should now be at: $REPO_URL"
echo
echo "If the push failed because the remote repo doesn't exist yet:"
echo "  1. Go to https://github.com/new"
echo "  2. Owner: samaidev, Name: lal, Private"
echo "  3. Don't initialize with README (we already have one)"
echo "  4. Click 'Create repository'"
echo "  5. Re-run this script"
