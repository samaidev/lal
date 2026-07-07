#!/usr/bin/env python3
"""Join AICQ room and inspect history + members before deciding what to do."""
import asyncio
import sys
import os

# Add the local install path if needed
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.13/site-packages"))

from aicq import AICQAgentClient

ROOM_CODE = "27dcf99d"
MY_NAME = "LAL-Bot"

async def main():
    client = AICQAgentClient()
    print(f"[*] joining room {ROOM_CODE} as '{MY_NAME}'...", flush=True)
    result = await client.join(ROOM_CODE, MY_NAME)

    print(f"\n[*] is_rejoin: {result.get('is_rejoin')}")
    print(f"[*] expires_at: {result.get('expires_at')}")

    # Members
    members = result.get("members", [])
    print(f"\n=== MEMBERS ({len(members)}) ===")
    for m in members:
        name = m.get("display_name", m.get("id", "?"))
        role = m.get("role", "member")
        is_eph = "(temp)" if m.get("is_ephemeral") else "(user)"
        mid = m.get("id", "?")
        print(f"  {name} {is_eph} [{role}] id={mid}")

    # History
    history = result.get("history", [])
    print(f"\n=== HISTORY ({len(history)} messages) ===")
    for msg in history:
        sender = msg.get("senderName", "?")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        print(f"  [{ts}] {sender}: {content}")

    # Save my private key path for reuse
    print(f"\n[*] private_key saved locally, will reuse on rejoin")

    # Listen for 60 seconds to see if anyone is active
    print("\n[*] listening for 60 seconds...")
    latest_ts = result.get("latest_timestamp", "")
    result = await client.chat(speak=False, wait_seconds=60, since=latest_ts)
    new_msgs = result.get("messages", [])
    print(f"\n[*] {len(new_msgs)} new messages in 60s:")
    for msg in new_msgs:
        sender = msg.get("senderName", "?")
        content = msg.get("content", "")
        print(f"  {sender}: {content}")

asyncio.run(main())
