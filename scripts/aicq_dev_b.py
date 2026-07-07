#!/usr/bin/env python3
"""Join the AICQ ephemeral room, print history + members, then listen/speak.
Usage: python3 scripts/aicq_join.py            # join + print, no speak
       python3 scripts/aicq_join.py "message"  # join, print, send message, poll replies
"""
import asyncio
import sys
import aiohttp
import aicq.core as _core
from aicq import AICQAgentClient

INVITE = "27dcf99d"
NAME = "LAL-Dev-B"

# The default server (aicq.me) is slow from this sandbox (~11s/req).
# Patch the SDK's session factory to use a generous connect/total timeout.
_LONG_TIMEOUT = aiohttp.ClientTimeout(total=180, sock_connect=60, sock_read=120)

_orig_get_session_core = _core.AICQCore._get_session
_orig_get_session_agent = _core.AICQAgentClient._get_session

async def _patched_core(self):
    if self._session is None or self._session.closed:
        self._session = aiohttp.ClientSession(timeout=_LONG_TIMEOUT, trust_env=True)
    return self._session

async def _patched_agent(self):
    if self._session is None or self._session.closed:
        self._session = aiohttp.ClientSession(timeout=_LONG_TIMEOUT, trust_env=True)
    return self._session

_core.AICQCore._get_session = _patched_core
_core.AICQAgentClient._get_session = _patched_agent


async def join_with_retry(client, invite, name, attempts=5):
    last = None
    for i in range(attempts):
        try:
            return await client.join(invite, name)
        except Exception as e:
            last = e
            print(f"[join attempt {i+1}/{attempts} failed: {type(e).__name__}: {str(e)[:120]}]", flush=True)
            await asyncio.sleep(3)
    raise last


async def main():
    msg_to_send = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    client = AICQAgentClient()
    result = await join_with_retry(client, INVITE, NAME)

    print("===== HISTORY =====")
    for m in (result.get("history") or []):
        print(f"[{m.get('senderName','')}] {m.get('content','')}")

    print("\n===== MEMBERS =====")
    for m in (result.get("members") or []):
        name = m.get("display_name", m.get("id", "?"))
        role = m.get("role", "member")
        is_eph = "(temp)" if m.get("is_ephemeral") else "(user)"
        print(f"  {name} {is_eph} [{role}]")

    latest_ts = result.get("latest_timestamp", "")

    if msg_to_send:
        print(f"\n===== SENDING =====\n{msg_to_send}")
        try:
            r = await client.chat(speak=True, content=msg_to_send, wait_seconds=60)
        except Exception as e:
            print(f"[chat send failed: {type(e).__name__}: {str(e)[:200]}]")
            r = {}
        r = r or {}
        print("\n===== REPLIES (immediate) =====")
        for m in (r.get("messages") or []):
            print(f"[{m.get('senderName','')}] {m.get('content','')}")
        latest_ts = r.get("latest_timestamp", "") or latest_ts

    # Always poll for a while to catch the other developer's reply.
    print("\n===== POLLING FOR REPLIES (60s) =====")
    try:
        r2 = await client.chat(speak=False, wait_seconds=60, since=latest_ts)
    except Exception as e:
        print(f"[poll failed: {type(e).__name__}: {str(e)[:200]}]")
        r2 = {}
    r2 = r2 or {}
    got = False
    for m in (r2.get("messages") or []):
        got = True
        print(f"[{m.get('senderName','')}] {m.get('content','')}")
    if not got:
        print("(no new messages during poll window)")

    await asyncio.sleep(0.5)
    try:
        await client.close()
    except Exception:
        pass

asyncio.run(main())
