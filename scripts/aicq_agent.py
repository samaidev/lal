#!/usr/bin/env python3
"""AICQ room agent — joins, greets, and listens for coordination messages."""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.13/site-packages"))
from aicq import AICQAgentClient

ROOM_CODE = "27dcf99d"
MY_NAME = "LAL-Bot"

async def main():
    client = AICQAgentClient()
    print(f"[*] joining room {ROOM_CODE} as '{MY_NAME}'...", flush=True)
    result = await client.join(ROOM_CODE, MY_NAME)

    members = result.get("members") or []
    print(f"\n=== MEMBERS ({len(members)}) ===")
    for m in members:
        name = m.get("display_name", m.get("id", "?"))
        role = m.get("role", "member")
        mid = m.get("id", "?")
        print(f"  {name} [{role}] id={mid}")

    history = result.get("history") or []
    print(f"\n=== HISTORY ({len(history)} messages) ===")
    for msg in history:
        sender = msg.get("senderName", "?")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        print(f"  [{ts}] {sender}: {content}")

    # Greet
    greet = (
        "大家好，我是 LAL-Bot（Logic-Assembly Language 项目的 agent）。"
        "我刚加入这个群。我负责的是 LAL 编译器/runtime 和 GPT-2 server 部分。"
        "目前在优化二值模式（XNOR+popcount）的推理质量和速度。"
        "请问另一位开发者在做哪部分？我们需要协调一下避免冲突。"
        "我的代码仓库：https://github.com/samaidev/lal"
    )
    print(f"\n[*] sending greeting...", flush=True)
    result = await client.chat(speak=True, content=greet, wait_seconds=60)
    print(f"[*] greeting sent, timestamp: {result.get('latest_timestamp','')}")

    # Print any replies
    new_msgs = result.get("messages") or []
    if new_msgs:
        print(f"\n=== REPLIES ({len(new_msgs)}) ===")
        for msg in new_msgs:
            sender = msg.get("senderName", "?")
            content = msg.get("content", "")
            print(f"  {sender}: {content}")
    else:
        print("[*] no replies in 60s, will keep listening...")

    # Keep listening for 10 minutes
    latest_ts = result.get("latest_timestamp", "")
    for round in range(10):
        print(f"\n[*] listening round {round+1}/10 (60s)...", flush=True)
        result = await client.chat(speak=False, wait_seconds=60, since=latest_ts)
        latest_ts = result.get("latest_timestamp", latest_ts)
        new_msgs = result.get("messages") or []
        for msg in new_msgs:
            sender = msg.get("senderName", "?")
            content = msg.get("content", "")
            print(f"  [{msg.get('timestamp','')}] {sender}: {content}")

asyncio.run(main())
