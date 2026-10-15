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
from backtalk.hermes_workers import HermesWorkers, ROOT as HERMES_ROOT

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


# ---- BACKGROUND TASKS: the real SDK field names, not guesses.
#
# Verified by reading the installed SDK's own source (claude-agent-sdk
# 0.2.143, a tag-matched clone at reference/claude-agent-sdk-python-
# v0.2.143). Two earlier passes at this feature guessed and got it
# wrong, so the shapes are written down here:
#
#   TaskStartedMessage       task_id, description, uuid, session_id,
#                            tool_use_id, task_type
#   TaskProgressMessage      task_id, description, usage, uuid,
#                            session_id, tool_use_id, last_tool_name
#   TaskNotificationMessage  task_id, status, output_file, summary,
#                            uuid, session_id, tool_use_id, usage
#                            -- NO description field. The human-readable
#                            label has to be captured from the started/
#                            progress frames and remembered, or the only
#                            thing left to say out loud is a raw uuid.
#   TaskUpdatedMessage       task_id, patch (dict), status, session_id,
#                            uuid
#
# TaskUpdatedMessage.status IS a real attribute in this version, but
# message_parser.py only ever fills it from patch.get("status") -- the
# patch is the source of truth and .status is a derived copy that is
# None whenever the patch carried no status. So read the patch FIRST and
# treat .status as the fallback, never the other way round.
#
# Terminal completion can arrive as EITHER a task_notification or a
# task_updated patch; the SDK's own docstrings say so outright (a task
# killed via TaskStop reports only the task_updated). Both are handled.
try:
    from claude_agent_sdk import TERMINAL_TASK_STATUSES
except ImportError:                       # older SDKs: the real values
    TERMINAL_TASK_STATUSES = frozenset(
        {"completed", "failed", "stopped", "killed"})

try:
    # PRIVATE on purpose, and a reader should know it rather than
    # discover it: the SDK exports no public equivalent. This is the
    # set of task types that actually REACH a terminal status -- real
    # delegated agent/workflow work. A background SHELL (Bash with
    # run_in_background on a dev server, a tail -f) is reported through
    # the very same lifecycle frames but may never report terminal at
    # all, so anything that tracks "still outstanding" has to exclude
    # it or the tracker wedges permanently. The SDK hit this exact wall
    # itself -- see _track_task_lifecycle's docstring in
    # _internal/query.py. Shells are still ANNOUNCED when they do
    # finish; they're only kept out of the outstanding set.
    from claude_agent_sdk._internal.query import DEFERRING_TASK_TYPES
except Exception:
    DEFERRING_TASK_TYPES = frozenset({"local_agent", "local_workflow"})

# How long to hold a finished-task announcement before handing it over,
# so two jobs landing seconds apart become one spoken line.
_ANNOUNCE_COALESCE_S = 2.0
# When a background task lands, the CLI wakes the agent for a follow-up
# turn and the model often writes its own summary with nobody having
# asked. That text is already paid for, so it beats anything we could
# synthesize -- but it takes real seconds to arrive. These bound the
# wait for it: give up early at _IDLE once nothing is streaming, or at
# _MAX regardless. Both are judgement calls, not measurements.
_ANNOUNCE_FOLLOWUP_IDLE_S = 6.0
_ANNOUNCE_FOLLOWUP_MAX_S = 14.0
# Long sessions shouldn't accumulate task bookkeeping forever.
_TASK_MEMORY_CAP = 200


class _TurnSlot:
    """The single-slot turn channel: where the pump puts the messages
    belonging to the ONE live foreground turn.

    main.py already serializes every input source down to one live turn
    at a time (handle() cancels and awaits the previous speak_task before
    starting another), so one slot is enough -- and it reproduces
    receive_response()'s semantics exactly, minus the shared-stream
    off-by-one bug, because a retired turn's leftovers die with its slot
    instead of pairing with the next question.
    """
    __slots__ = ("q",)

    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()


class _StreamFailure:
    """Sentinel: the pump stopped feeding this slot.

    exc carries the REAL exception object the SDK raised, never a
    re-worded stand-in. That matters concretely: speak_reply's crash
    handler does a substring check for "maximum buffer size" to tell the
    oversized-tool-result case (a real, repeated failure on this
    machine) apart from a generic internal error, and synthesizing a
    tidy RuntimeError here would destroy that diagnostic. exc is None
    ONLY for a genuinely clean end-of-stream with no underlying error.
    """
    __slots__ = ("exc",)

    def __init__(self, exc: Exception | None):
        self.exc = exc


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
        # ---- THE PERMANENT MESSAGE PUMP.
        # One long-lived task drains receive_messages() for the entire
        # life of the connection and never goes idle. Not a refactor for
        # tidiness: the SDK's own reader task routes the MCP control
        # protocol (control_response/control_request) on the SAME loop
        # that hands SDK messages to a bounded memory stream. Nobody
        # draining that stream means the reader BLOCKS on send, and
        # control responses stop being routed with it -- upstream
        # anthropics/claude-agent-sdk-python#425. Per-turn
        # receive_response() calls leave exactly that gap between turns.
        self._pump: asyncio.Task | None = None
        # Set ONLY by the pump's exit path. This is what tells "the turn
        # retired normally" apart from "the slot was force-cleared
        # because the stream died" -- both leave _slot None, and
        # treating the second as healthy is how a dead connection gets
        # announced as "the connection is ready again."
        self._stream_dead = False
        self._stream_error: Exception | None = None
        self._slot: _TurnSlot | None = None
        # Bumped per foreground turn, so a finished task can be told
        # apart from one that finished inside the very turn that spawned
        # it (that turn narrates its own subagents; announcing them too
        # would be duplicate noise).
        self._turn_gen = 0
        # task_id -> {"label", "task_type", "gen"}
        self._tasks_seen: dict = {}
        self._outstanding: set = set()     # deferring-type tasks only
        self._announced: set = set()       # dedup on the REAL task_id
        self._ann_pending: list = []
        self._ann_event = asyncio.Event()
        self._hermes = HermesWorkers(
            HERMES_ROOT / "profiles/jarvis-workers/jobs", self._hermes_changed)
        self._hermes_mcp = self._hermes.mcp_server()
        # An unprompted SDK follow-up turn being collected right now.
        self._orphan_text: list = []
        self._orphan_active = False

    def _hermes_changed(self, job):
        """External jobs share the queue, but are not fabricated SDK messages.

        The queued item carries a DECISION, not a pre-written line:
        needs_review says whether this ending is worth a real model turn.

        - completed: always. The worker's output is untrusted text from an
          external model, so the orchestrator has to read it before a word
          of it is spoken -- and then actually report what it says. The old
          version wrote a fixed "its result is ready for review" summary
          here and leaned on main.py's thin-summary gate to escalate; that
          string is 63 characters against a 60-character threshold, so the
          escalation missed by three characters EVERY time and the cheap
          path announced a review that never happened.
        - failed: the diagnostic is worth a turn (unchanged behavior).
        - cancelled / interrupted: Patrick stopped it himself and already
          knows why, so a cheap line is the whole truth (changed
          2026-10-14: these used to escalate just for being != completed).

        NOTHING from the job row's own text is copied in. `result` is
        worker-controlled on more than the completed path -- a worker that
        stops with stop_reason != "end_turn" lands on status "cancelled"
        carrying its own raw output -- so a reviewed item fetches the real
        thing through the MCP result tool and an unreviewed one says only
        that it stopped.
        """
        tid = "hermes:" + job["id"]
        if job["status"] == "running":
            self._outstanding.add(tid)
            return
        self._outstanding.discard(tid)
        if self._hermes.closed or tid in self._announced:
            return
        self._announced.add(tid)
        status = str(job["status"])
        self._ann_pending.append({
            "task_id": tid, "job_id": job["id"], "label": job["label"],
            "task_type": "hermes", "status": status,
            "needs_review": status in ("completed", "failed"),
            "summary": "",
            # A Hermes job can never legitimately carry one: it is an
            # external process, and nothing wakes the agent for a
            # follow-up turn when it finishes. See _collect_orphan.
            "model_text": "",
        })
        self._ann_event.set()

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
                mcp_servers={"hermes-workers": self._hermes_mcp},
                resume=rid,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                self._start_pump()
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
        self._start_pump()
        await self._publish_context_usage()

    # ------------------------------------------------------------------
    # The permanent pump, the turn channel, and the announcement queue
    # ------------------------------------------------------------------

    def _start_pump(self):
        """Install a fresh pump over the freshly-connected client.

        Also the one place every piece of per-connection state is reset,
        so a rebuild can never inherit a dead stream's flags, a retired
        slot, or stale task bookkeeping. _ann_event and _ann_pending are
        CLEARED rather than rebound on purpose: main.py's announcer may
        already be parked on that exact Event object, and swapping it
        would orphan the waiter.
        """
        self._stream_dead = False
        self._stream_error = None
        self._slot = None
        self._turn_gen = 0
        self._tasks_seen.clear()
        self._outstanding = {tid for tid in self._outstanding if tid.startswith("hermes:")}
        self._announced = {tid for tid in self._announced if tid.startswith("hermes:")}
        self._ann_pending[:] = [item for item in self._ann_pending
                               if item["task_id"].startswith("hermes:")]
        self._ann_event.clear()
        if self._ann_pending:
            self._ann_event.set()
        self._orphan_text.clear()
        self._orphan_active = False
        # The client is bound into the coroutine rather than read off
        # self, so a pump that outlives a rebuild by a few ticks can
        # never read the NEW connection.
        self._pump = asyncio.create_task(self._pump_loop(self._client))
        self._pump.add_done_callback(self._pump_died)

    async def _pump_loop(self, client):
        """Drain the SDK's message stream forever, routing as it goes."""
        try:
            async for msg in client.receive_messages():
                self._route(msg)
        except asyncio.CancelledError:
            raise                 # deliberate teardown; _stop_pump's job
        except Exception as exc:
            # The REAL exception object goes to whoever is waiting on the
            # live turn. receive_messages() already re-raises the typed
            # original (ProcessError/ResultError/CLIJSONDecodeError/the
            # transport's buffer-overflow error) rather than flattening
            # it -- see _internal/query.py's error frame -- so there is
            # nothing to gain and a real diagnostic to lose by wrapping.
            log(f"[brain] message pump stopped on an error: {exc!r}")
            self._mark_stream_dead(exc)
            return
        # Clean EOF: the only case that genuinely has no exception to
        # carry, so the generic "ended without ResultMessage" error is
        # synthesized HERE and nowhere else.
        log("[brain] message pump reached end of stream")
        self._mark_stream_dead(None)

    def _pump_died(self, task: asyncio.Task):
        """Done-callback: the liveness backstop.

        Without this, a pump that died some way the body couldn't catch
        (a BaseException, or a cancellation from something other than
        _stop_pump) would leave the brain silently bricked -- callers
        parked on slot.q.get() forever, with no timeout and nothing
        watching. Idempotent: the normal paths already marked the stream
        dead and this leaves them alone.
        """
        if task is not self._pump:
            return                # superseded by a rebuild; not ours
        if task.cancelled():
            return                # deliberate teardown
        if self._stream_dead:
            return
        try:
            exc = task.exception()
        except Exception:
            exc = None
        log(f"[brain] message pump task exited unexpectedly: {exc!r}")
        self._mark_stream_dead(exc if isinstance(exc, Exception) else None)

    def _mark_stream_dead(self, exc: Exception | None):
        """The ONE fail-open path that clears the live turn slot.

        Everything downstream keys off _stream_dead rather than off
        "_slot is None", precisely because this path and a clean turn
        retirement both leave the slot empty.
        """
        self._stream_dead = True
        if exc is not None and self._stream_error is None:
            self._stream_error = exc
        slot, self._slot = self._slot, None
        if slot is not None:
            try:
                slot.q.put_nowait(_StreamFailure(exc))
            except Exception:
                pass

    async def _stop_pump(self):
        """Cancel the pump and wait for it, WITHOUT eating a cancellation
        the caller needs to see.

        asyncio.wait() is deliberate: it never re-raises the awaited
        task's own CancelledError (so the expected teardown stays quiet)
        and it does not shield us from ours (so a rebuild that is itself
        cancelled by new input still propagates). A blanket
        `except (CancelledError, Exception): pass` around `await task`
        cannot tell those two apart and swallows both.
        """
        task, self._pump = self._pump, None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.wait({task})

    def _begin_turn(self) -> _TurnSlot:
        """Install the turn channel for one foreground turn.

        Liveness is checked HERE rather than discovered later: installing
        a slot over a dead pump hands the caller a queue nothing will
        ever feed, and slot.q.get() has no timeout. Raising instead sends
        the turn down speak_reply's crash handler, which calls
        reset_turn(), which sees _stream_dead and rebuilds -- and because
        the stored exception is the original, the handler's
        "maximum buffer size" check still names the real cause even one
        turn later.
        """
        if self._client is None:
            raise RuntimeError("No Claude session is connected")
        if self._stream_dead or self._pump is None or self._pump.done():
            if not self._stream_dead:
                self._mark_stream_dead(None)
            raise (self._stream_error
                   or RuntimeError("The Claude message pump is not "
                                   "running; the session needs "
                                   "rebuilding"))
        self._turn_gen += 1
        slot = _TurnSlot()
        self._slot = slot
        return slot

    async def _turn_stream(self, slot: _TurnSlot):
        """receive_response()'s semantics, fed by the pump: every message
        up to and including this turn's ResultMessage, then stop."""
        while True:
            item = await slot.q.get()
            if isinstance(item, _StreamFailure):
                if item.exc is not None:
                    raise item.exc            # the real one, untouched
                raise RuntimeError(
                    "Response stream ended without ResultMessage")
            if type(item).__name__ == "ResultMessage":
                # Retire the slot BEFORE yielding: every caller breaks
                # out of the loop on the ResultMessage, so code after
                # the yield would never run.
                if self._slot is slot:
                    self._slot = None
                yield item
                return
            yield item

    # ---- routing -----------------------------------------------------

    def _route(self, msg):
        """Sort one message: live turn, announcement queue, or orphan.

        Strictly synchronous -- no awaits -- so the pump keeps the SDK's
        reader drained and the control protocol flowing.
        """
        t = type(msg).__name__
        if t in ("TaskStartedMessage", "TaskProgressMessage"):
            # Label capture. Both frames carry `description`; the
            # terminal notification does NOT, so if this is skipped the
            # only thing left to announce is a uuid.
            self._note_task(msg)
        elif t == "TaskNotificationMessage":
            self._finish_task(getattr(msg, "task_id", None),
                              getattr(msg, "status", None),
                              getattr(msg, "summary", "") or "")
            return                 # announcement queue only, not the turn
        elif t == "TaskUpdatedMessage":
            patch = getattr(msg, "patch", None)
            status = (patch.get("status") if isinstance(patch, dict)
                      else None) or getattr(msg, "status", None)
            if status in TERMINAL_TASK_STATUSES:
                self._finish_task(getattr(msg, "task_id", None), status, "")
                return             # announcement queue only, not the turn
        slot = self._slot
        if slot is not None:
            slot.q.put_nowait(msg)
            return
        self._collect_orphan(msg)

    def _note_task(self, msg):
        tid = getattr(msg, "task_id", None)
        if not tid:
            return
        rec = self._tasks_seen.get(tid)
        if rec is None:
            if len(self._tasks_seen) >= _TASK_MEMORY_CAP:
                for old in list(self._tasks_seen)[:_TASK_MEMORY_CAP // 2]:
                    self._tasks_seen.pop(old, None)
            rec = {"label": None, "task_type": None, "gen": self._turn_gen}
            self._tasks_seen[tid] = rec
        desc = (getattr(msg, "description", None) or "").strip()
        if desc and not rec["label"]:
            rec["label"] = desc[:160]
        ttype = getattr(msg, "task_type", None)
        if ttype:
            rec["task_type"] = ttype
        if ttype in DEFERRING_TASK_TYPES:
            # Only delegated agent/workflow work goes in here -- a
            # background shell may never report terminal at all.
            self._outstanding.add(tid)

    def _finish_task(self, tid, status, summary: str):
        if not tid:
            return
        self._outstanding.discard(tid)
        status = str(status) if status else "completed"
        summary = (summary or "").strip()
        # Terminal state can arrive twice (a task_updated patch and then
        # the richer task_notification, or the reverse). Merge into the
        # entry still waiting rather than dropping the one that actually
        # carries the summary.
        for pend in self._ann_pending:
            if pend["task_id"] == tid:
                if summary and not pend["summary"]:
                    pend["summary"] = summary
                pend["status"] = status
                return
        if tid in self._announced:
            return                  # dedup on the real id, never a label
        if len(self._announced) >= _TASK_MEMORY_CAP:
            self._announced.clear()
        self._announced.add(tid)
        rec = self._tasks_seen.get(tid) or {}
        if self._slot is not None and rec.get("gen") == self._turn_gen:
            # Finished inside the very turn that spawned it. That turn is
            # about to speak about it itself; announcing would duplicate.
            return
        if rec.get("task_type") not in DEFERRING_TASK_TYPES:
            # Caught live 2026-10-14: a running local_agent/local_workflow
            # dispatch surfaces its OWN internal steps as Task frames on
            # the same connection (file reads, greps, sub-checks). Those
            # finish constantly and are never the thing worth a ping --
            # only the top-level dispatch itself (the type this same gate
            # already uses to populate _outstanding) is a real deliverable.
            # Without this, every nested step queued for announcement,
            # including raw descriptions (a file path, once spoken
            # verbatim and literally read aloud to Patrick mid-session).
            log(f"[brain] non-deliverable task finished, not announcing "
                f"({rec.get('label') or 'unlabelled'}, {status})")
            return
        self._ann_pending.append({
            "task_id": tid,
            "label": rec.get("label"),
            "task_type": rec.get("task_type"),
            "status": status,
            "summary": summary,
            "model_text": "",
        })
        self._ann_event.set()
        log(f"[brain] background task finished "
            f"({rec.get('label') or 'unlabelled'}, {status})")

    def _collect_orphan(self, msg):
        """Collect an UNPROMPTED turn -- one nobody asked for.

        When a background task lands, the CLI wakes the agent for a
        follow-up turn, and the model frequently writes its own account
        of what happened. No slot is installed for it (nothing here
        queried), so without this it lands nowhere. Preferring that text
        over anything synthesized is the cheap path done right: the
        tokens are already spent either way.
        """
        t = type(msg).__name__
        if t in ("AssistantMessage", "StreamEvent"):
            self._orphan_active = True
        if t == "AssistantMessage":
            for b in getattr(msg, "content", []) or []:
                txt = getattr(b, "text", None)
                if txt:
                    self._orphan_text.append(txt)
            return
        if t != "ResultMessage":
            return
        text = " ".join(" ".join(self._orphan_text).split()).strip()
        self._orphan_text.clear()
        self._orphan_active = False
        try:
            # Honest bookkeeping: this turn cost real tokens even though
            # nobody here asked for it. Not counted as a conversation
            # turn, same treatment console commands get.
            self._tally(msg, count_turn=False)
            self._remember_session(msg)
        except Exception:
            pass
        if not text:
            return
        attached = False
        for pend in self._ann_pending:
            if pend.get("task_type") == "hermes":
                # NEVER a Hermes job. This text belongs to whatever woke
                # the agent, and nothing wakes it for a Hermes job -- it's
                # an external process the CLI knows nothing about. So an
                # SDK task's follow-up coalescing into the same batch
                # would get pasted onto the Hermes item and spoken as if
                # it were about the worker, AND it would satisfy the
                # model_text short-circuit in main._compose_announcement,
                # skipping the completion review entirely.
                continue
            if not pend["model_text"]:
                pend["model_text"] = text[:1500]
                attached = True
        if attached:
            self._ann_event.set()
        else:
            log(f"[brain] unprompted follow-up with nothing pending: "
                f"{text[:140]}")

    # ---- the announcement queue, read by main.py ---------------------

    def has_outstanding_tasks(self) -> bool:
        """True while a delegated agent/workflow task is still running.

        GUARD POINT, left here deliberately: there is no context
        recycler in this build (Patrick removed the mid-session
        auto-recycle 2026-10-12). If one is ever added back, it must
        consult this before clearing the conversation -- wiping context
        out from under an outstanding background task destroys the
        agent's ability to say anything useful when it lands. Only
        deferring task types count; a background shell may never report
        terminal and would wedge this True forever.
        """
        return bool(self._outstanding)

    async def wait_for_announcements(self) -> list:
        """Park until a background task finishes, then hand over every
        finished task as one coalesced batch.

        A Hermes item counts as already satisfied: the unprompted
        follow-up this loop is waiting for can only come from the CLI
        waking the agent, which never happens for an external worker
        process. Waiting on it just burned the full
        _ANNOUNCE_FOLLOWUP_IDLE_S as dead air before every single Hermes
        announcement."""
        await self._ann_event.wait()
        await asyncio.sleep(_ANNOUNCE_COALESCE_S)
        started = time.monotonic()
        while self._ann_pending:
            if all(p["model_text"] or p.get("task_type") == "hermes"
                   for p in self._ann_pending):
                break
            waited = time.monotonic() - started
            if waited >= _ANNOUNCE_FOLLOWUP_MAX_S:
                break
            if not self._orphan_active and waited >= _ANNOUNCE_FOLLOWUP_IDLE_S:
                break
            await asyncio.sleep(0.5)
        items = list(self._ann_pending)
        self._ann_pending.clear()
        self._ann_event.clear()
        return items

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
        slot = self._begin_turn()
        self._dirty = True
        # Clear/compact/model commands may change the context or its capacity.
        signals.set_context_usage(None)
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._turn_stream(slot):
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
        the last turn was consumed clean.

        With the permanent pump in place the drain reads the dead turn's
        OWN slot rather than a shared stream, so the leftovers can't be
        mistaken for the next question's answer. The pump keeps the SDK
        reader drained throughout, so the drain can never deadlock
        against the control protocol."""
        # FIRST, before any early return: a dead stream also leaves the
        # slot empty, and "empty slot" is exactly what a clean turn
        # retirement looks like. Checking _dirty/_slot first would
        # diagnose a dead connection as healthy and let speak_reply's
        # handler tell Patrick "the connection is ready again" about a
        # brain that answers nothing.
        if self._stream_dead:
            log("[brain] the message pump is dead — rebuilding rather "
                "than reporting a false recovery")
            return await self._rebuild()
        if not self._client or not self._dirty:
            return
        slot = self._slot
        if slot is None:
            # Marked in flight, yet its channel is gone and the pump
            # never reported a failure. Unexplained — so don't assume
            # healthy.
            log("[brain] a dirty turn with no live channel — rebuilding")
            return await self._rebuild()
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            # _turn_stream raises the pump's real exception on a dead
            # stream and returns at the ResultMessage; EOF is never
            # mistaken for recovery.
            async for _msg in self._turn_stream(slot):
                n += 1
            return n

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            return await self._rebuild()

    async def _rebuild(self):
        """Last resort: throw the connection away and start clean.

        Loses this voice session's conversation memory; better than
        answering every question one turn late for the rest of the day,
        and far better than a bricked pump nobody noticed. Returns True
        so the caller discloses the reset out loud."""
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
        # Not wrapped in a swallow-everything: if THIS rebuild is itself
        # cancelled by new input arriving, the caller has to see it.
        await self._stop_pump()
        if self._client is not None:
            try:
                await asyncio.wait_for(self._client.disconnect(), 5)
            except Exception:
                pass
        self._client = None
        self._resume_meta = None
        self._session_started = time.time()
        self._last_activity = None
        self._last_input_tokens = 0
        await asyncio.wait_for(self.start(), 20)   # _start_pump resets the flags
        self._dirty = False
        return True  # caller must disclose that conversation was reset

    async def stop(self):
        signals.set_context_usage(None)
        await self._hermes.close()
        await self._stop_pump()
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model."""
        # Only literal human input can approve the staged job. An MCP call,
        # model claim, or a worker's returned text cannot mint approval.
        approval = re.sub(r"[.!?]+$", "", utterance.strip()).lower()
        if approval in {"approve hermes worker", "approve the hermes worker"}:
            try:
                job = self._hermes.approve()
                utterance += ("\n(Jarvis worker control: the pending, approved Hermes task "
                              f"has started; job_id={job['id']}. Do not start a duplicate.)")
            except ValueError as exc:
                yield str(exc) + ". Over to you."
                return
        # 2026-10-12: no more automatic mid-session recycling here --
        # Patrick's own "clear"/"compact" commands are the only thing
        # that resets this conversation now. See _STALE_IDLE_SECONDS.
        # Claim the turn channel BEFORE the query goes out, and let a
        # dead pump raise here rather than hand back a queue nothing
        # feeds. The raise lands in speak_reply's crash handler, which
        # recovers exactly the way it already does.
        slot = self._begin_turn()
        self._dirty = True             # in flight until its ResultMessage
        await self._client.query(utterance)
        buf = ""
        async for msg in self._turn_stream(slot):
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
