# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import json
import os
import re
import time
import warnings
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")
# Durable companion to SESSION_FILE: started_at/last_activity/
# last_input_tokens, used by main.py's launch-time resume check.
# SESSION_FILE's own mtime is useless for that -- it gets rewritten on
# every completed turn, so a conversation that just never restarts
# always looks "today" no matter how old or huge it's gotten. Root
# cause of the 2026-10-05 five-hour-limit incident: an un-relaunched
# overnight session kept resuming its own ballooning history past the
# old launch-only day check. See the Jarvis Stack Runbook for the full
# incident.
SESSION_META_FILE = os.path.join(CFG["signals_dir"],
                                 ".backtalk_session_meta.json")
# 2026-10-12: Patrick decided the mid-session auto-recycle (both the
# new-day and idle-duration triggers) was costing more in lost context
# than it saved -- he'd rather notice and say "clear"/"compact" himself.
# brain.py no longer recycles on its own. This constant survives only
# because main.py's own launch-time resume check (a separate, one-time
# decision: reattach to a saved session or start clean) still uses it.
_STALE_IDLE_SECONDS = 2 * 60 * 60
# Token counts are retained for reporting, never used to force a reset.


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None, resume_meta: dict | None = None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use: a desync rebuild in
        # reset_turn() must always start FRESH: a rebuild means a turn
        # went sideways mid-stream, the wrong moment to gamble on
        # reattaching. (Community proposal, issue #1.)
        self._resume_id = resume_id
        # main.py's pre-loaded SESSION_META_FILE contents for resume_id,
        # if it decided resuming was still safe. Applied once in
        # start(), then dropped -- same consume-on-use shape as
        # _resume_id, for the same reason (a rebuild must never inherit
        # a stale clock).
        self._resume_meta = resume_meta
        # The staleness guard's own clock: when THIS conversation
        # (fresh or resumed) actually started, when its last turn
        # finished, and how much context that turn carried. Defaults
        # assume fresh; overwritten from _resume_meta in start() if
        # resuming.
        self._session_started = time.time()
        self._last_activity = None
        self._last_input_tokens = 0
        # True while a query's response hasn't been consumed through its
        # ResultMessage — i.e. the shared message pipe may hold leftovers.
        self._dirty = False

    async def start(self):
        signals.set_context_usage(None)
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" = the SDK's "default" mode with gated calls
        # routed to the spoken can_use_tool gate.
        sdk_mode = "default" if mode == "ask" else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE},
                include_partial_messages=True,
                max_buffer_size=16 * 1024 * 1024,
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                resume=rid,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                meta, self._resume_meta = self._resume_meta, None  # consume once
                if meta:
                    self._session_started = meta.get("started_at",
                                                      self._session_started)
                    self._last_activity = meta.get("last_activity")
                    self._last_input_tokens = meta.get("last_input_tokens", 0)
                log(f"[brain] resumed session {resume[:8]}")
                await self._publish_context_usage()
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        # Fresh conversation: never inherit a stale clock from a resume
        # attempt that didn't pan out.
        self._resume_meta = None
        self._session_started = time.time()
        self._last_activity = None
        self._last_input_tokens = 0
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()
        await self._publish_context_usage()

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    async def _publish_context_usage(self):
        """A bounded control query, not a new model turn or an auto-reset."""
        if not CFG.get("show_usage"):
            signals.set_context_usage(None)
            return
        try:
            usage = await asyncio.wait_for(self.context_usage(), 2)
        except Exception:
            usage = None
        signals.set_context_usage(usage)

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Also updates
        the staleness guard's own durable clock (SESSION_META_FILE) --
        unlike SESSION_FILE's mtime, started_at is never overwritten by
        this, which is the whole point: it has to survive every
        completed turn unchanged so a conversation's true age stays
        knowable. Must never break a turn; silence on any failure."""
        if not CFG.get("resume_last_session"):
            return
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        self._last_activity = time.time()
        self._last_input_tokens = self._measured_input_tokens(rm)
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
            with open(SESSION_META_FILE, "w") as f:
                json.dump({
                    "session_id": sid,
                    "started_at": self._session_started,
                    "last_activity": self._last_activity,
                    "last_input_tokens": self._last_input_tokens,
                }, f)
        except OSError:
            pass

    @staticmethod
    def _measured_input_tokens(rm) -> int:
        """Total measured input context on a completed turn: live input
        plus whatever the prompt cache read or had to recreate. The
        proxy the staleness guard sizes a carried conversation by --
        not a claim about Anthropic's actual quota formula, which isn't
        published."""
        try:
            u = getattr(rm, "usage", None) or {}
            return (int(u.get("input_tokens") or 0)
                    + int(u.get("cache_read_input_tokens") or 0)
                    + int(u.get("cache_creation_input_tokens") or 0))
        except Exception:
            return 0

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) through the normal stream and return whatever text the
        CLI answered with (confirmations, errors). Slash-command replies
        arrive as COMPLETE AssistantMessages, not stream deltas, so
        ask_stream cannot see them. Bounded like reset_turn is: this
        stream is not trusted to always deliver, and an unbounded await
        here would deafen the whole voice loop. On timeout the pipe is
        left marked dirty so the next reset_turn drains or rebuilds."""
        self._dirty = True
        # Clear/compact/model commands may change the context or its capacity.
        signals.set_context_usage(None)
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif t == "ResultMessage":
                    self._dirty = False
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)
                    break

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            return "error: the command timed out"
        await self._publish_context_usage()
        return " ".join(texts).strip()

    async def interrupt(self):
        if self._client:
            await self._client.interrupt()

    async def reset_turn(self, timeout: float = 8.0):
        """Re-align the message pipe after an interrupted/failed turn.

        THE OFF-BY-ONE BUG, and why this method exists: the SDK client
        has ONE shared message stream and receive_response() stops at
        the FIRST ResultMessage it sees — there is no pairing between a
        query and its response. A cancelled turn stops consuming
        mid-stream, leaving the dead turn's remaining messages
        (including its ResultMessage) buffered. The next query then
        pairs with those leftovers: the first ask lands on the stale
        ResultMessage and yields nothing, and every ask after that
        answers the PREVIOUS question — for the rest of the session.
        So: interrupt the dead turn, then drain the pipe through its
        stale ResultMessage before the next query goes out. No-op when
        the last turn was consumed clean."""
        if not self._client or not self._dirty:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            async for msg in self._client.receive_response():
                n += 1
                if type(msg).__name__ == "ResultMessage":
                    return n
            # EOF is not recovery: the SDK reader may have died before
            # emitting the result that makes the shared pipe safe again.
            raise RuntimeError("Response stream ended without ResultMessage")

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            # Can't re-align — rebuild the session rather than run
            # desynced. Loses this voice session's conversation memory;
            # better than answering every question one turn late for the
            # rest of the day.
            log("[brain] stream desynced beyond repair — rebuilding the "
                "session (conversation memory for this session resets)")
            # Invalidate the failed conversation before reconnecting so
            # even an unsuccessful rebuild cannot resume it at launch.
            for path in (SESSION_FILE, SESSION_META_FILE):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            self._resume_id = None
            try:
                await asyncio.wait_for(self._client.disconnect(), 5)
            except Exception:
                pass
            self._client = None
            self._resume_meta = None
            self._session_started = time.time()
            self._last_activity = None
            self._last_input_tokens = 0
            await asyncio.wait_for(self.start(), 20)
            self._dirty = False
            return True  # caller must disclose that conversation was reset

    async def stop(self):
        signals.set_context_usage(None)
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model."""
        # 2026-10-12: no more automatic mid-session recycling here --
        # Patrick's own "clear"/"compact" commands are the only thing
        # that resets this conversation now. See _STALE_IDLE_SECONDS.
        self._dirty = True             # in flight until its ResultMessage
        await self._client.query(utterance)
        buf = ""
        async for msg in self._client.receive_response():
            # Reassert "thinking" on every raw stream message, tool calls
            # included. This is deliberately NOT the can_use_tool gate
            # (main.py's gate()) -- found 2026-09-02 that the SDK silently
            # shadows that callback entirely in bypassPermissions/autoapprove
            # mode, so a reassertion living only there never fires in normal
            # operation. The raw message stream isn't gated by permission
            # mode at all, so hooking here actually closes the gap: mouth.py
            # legitimately goes idle between sentences of the same reply
            # (upstream's own documented behavior, see signals.reply_done),
            # and without this, that idle sits stale through any real work
            # -- tool calls, reasoning -- happening in that gap.
            signals.set_state("thinking")
            t = type(msg).__name__
            if t == "StreamEvent":
                ev = getattr(msg, "event", {}) or {}
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {}) or {}
                    if delta.get("type") == "text_delta":
                        buf += delta.get("text", "")
                        # emit any complete sentences
                        while True:
                            m = _SENTENCE_END.search(buf)
                            if not m:
                                break
                            sentence, buf = (buf[:m.end()].strip(),
                                             buf[m.end():])
                            if sentence:
                                yield sentence
                elif ev.get("type") == "content_block_stop":
                    # End of a speech block (e.g. right before a tool
                    # call): flush NOW. Without this, pre-tool filler
                    # ("On it — let me grab that.") sits silent in the
                    # buffer through the whole tool run, then plays
                    # glued to the answer: long dead air, then two
                    # thoughts at once.
                    tail = buf.strip()
                    buf = ""
                    if tail:
                        yield tail
            elif t == "ResultMessage":
                self._dirty = False    # turn fully consumed — pipe aligned
                self._tally(msg)
                self._remember_session(msg)
                await self._pull_rate_limits()
                await self._publish_context_usage()
                break
        tail = buf.strip()
        if tail:
            yield tail


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
