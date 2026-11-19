from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any

Clock = Callable[[], float]

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

API_KEY = os.environ.get("SMALLEST_API_KEY", "")

HOSTS = {
    "auto": "api.smallest.ai",
    "india": "api.india.smallest.ai",
    "us": "api.us.smallest.ai",
}
HOST = HOSTS["auto"]

RUNS = 20                 # samples per transport in bench.py
SAMPLE_RATE = 24000
OUTPUT_FORMAT = "wav"     # wav so the container can be verified from magic bytes
VOICE = "meher"
MODEL = "lightning_v3.1_pro"

IST = timezone(timedelta(hours=5, minutes=30))

# Paths copied from the API reference pages, not inferred. A guessed path
# returns 404 and a 404 reads as "undocumented behaviour", which would put a
# fabricated finding in the report.
#   https://docs.smallest.ai/api-reference/models/text-to-speech/synthesize-speech
#   https://docs.smallest.ai/api-reference/models/text-to-speech/tts
#   https://docs.smallest.ai/api-reference/models/text-to-speech/list-voice-clones
#   https://docs.smallest.ai/api-reference/models/text-to-speech/get-pronunciation-dicts
#   https://docs.smallest.ai/api-reference/models/llm-chat-completions/electron-chat-completions
PATH_TTS = "/waves/v1/tts"
PATH_TTS_LIVE = "/waves/v1/tts/live"   # SSE (POST) and WebSocket (wss) share it
PATH_VOICE_CLONES = "/waves/v1/voice-cloning"
PATH_PRON_DICTS = "/waves/v1/pronunciation-dicts"
PATH_CHAT = "/waves/v1/chat/completions"
PATH_GET_VOICES = "/waves/v1/lightning-v3.1/get_voices"
PATH_MINT = "/console/v1/auth/token"
PATH_MINT_LEGACY = "/waves/v1/auth/token"

# Documented constants, so a probe asserts against the spec rather than a guess.
WS_IDLE_DEFAULT = 60     # seconds, TTS WebSocket reference
WS_IDLE_MAX = 180        # larger ?timeout= values are clamped to this
TEXT_MAX = 8000          # 1-8000 characters after trimming
SAMPLE_RATES = (8000, 16000, 24000, 44100)

# `model` is derived from the voice when omitted: Pro-catalog voices resolve to
# lightning_v3.1_pro, every other voice to lightning_v3.1. So testing the
# retiring standard model means using a standard voice, not just dropping the
# field — with a Pro voice, omitting `model` still lands on Pro.
VOICE_PRO = "meher"
VOICE_STANDARD = "magnus"
MODEL_PRO = "lightning_v3.1_pro"
MODEL_RETIRING = "lightning_v3.1"   # retirement date to be verified

TEXT = "Your appointment is confirmed for Tuesday at four."

# Long enough that synthesis is still running when the next call lands. Used by
# the slot-release and cross-transport probes, where a short utterance can
# finish first and produce a false result.
TEXT_LONG = (
    "Thank you for calling. I can help you reschedule that appointment today. "
    "Looking at the calendar now, I have Tuesday the fourteenth at four in the "
    "afternoon, Wednesday morning at half past nine, or Thursday at two. "
    "Each of those is a thirty minute slot with the same clinician you saw "
    "last time, and I can hold any of them for up to an hour while you check. "
    "If none of those work, I can look at the following week, or add you to "
    "the cancellation list so we can call you if something opens up sooner. "
    "Which of those would you like me to do for you?"
)   # ~500 chars; doubled where a probe needs longer

# Gap after every single sample. The documented concurrency is 1 across REST,
# SSE and WS, and the slot is not necessarily released the instant a response
# finishes. Without this the harness manufactures its own 429s and then reports
# them as the product's rate limiting.
GAP = 0.5

LOG = "results.txt"

# --------------------------------------------------------------------------
# Secret scrubbing
# --------------------------------------------------------------------------
# Nothing written to results.txt or the JSON files may carry the API key or a
# minted access token. The sharp edge is exception text: a websockets failure
# on wss://...?token=sk_live_... puts the credential straight into the error
# string, which then lands in the evidence file that gets attached to a report.

_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Call this for every token the run mints, as soon as it exists."""
    if value and len(value) > 8:
        _SECRETS.add(value)


def scrub(obj: Any) -> Any:
    """Recursively replace known secrets with a placeholder."""
    if isinstance(obj, str):
        out = obj
        for s in _SECRETS:
            if s in out:
                out = out.replace(s, f"<redacted:{s[:3]}...{len(s)}chars>")
        return out
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return scrub(asdict(obj))
    return obj


def say(msg: str = "") -> None:
    """Print and append, with secrets removed. Timestamps are IST so lines can
    be joined against server-side logs when raising a ticket."""
    line = f"[{datetime.now(IST).strftime('%H:%M:%S')}] {scrub(msg)}" if msg else ""
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def require_key() -> str:
    if not API_KEY:
        sys.exit(
            "SMALLEST_API_KEY is not set.\n"
            "  export SMALLEST_API_KEY=sk_...\n"
            "Create the key on the console API Keys page; it is shown only once."
        )
    register_secret(API_KEY)
    return API_KEY


def auth(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or API_KEY}",
            "Content-Type": "application/json"}


def nonce(n: int = 6) -> str:
    return uuid.uuid4().hex[:n]


def unique_text(prefix: str = TEXT) -> str:
    """Fresh text per call, as ONE sentence.

    Appending "Ref abc123." as a second sentence is unsafe inside a
    context_id continuation: a segment is released at each sentence boundary,
    so a two-sentence fragment can produce two `complete` frames. A reader that
    stops at the first leaves the second segment's chunks queued on the socket,
    where they arrive at the start of the next sample and register as
    near-instant first audio — silently inventing a fast warm column.

    So the nonce goes inside the sentence, before the single full stop.
    """
    body = prefix.rstrip().rstrip(".")
    return f"{body}, reference {nonce()}."


def tts_body(text: str | None = None, **overrides: Any) -> dict[str, Any]:
    body = {
        "text": text if text is not None else unique_text(),
        "voice_id": VOICE,
        "model": MODEL,
        "sample_rate": SAMPLE_RATE,
        "output_format": OUTPUT_FORMAT,
    }
    body.update(overrides)
    return body


def ws_body(text: str | None = None, **overrides: Any) -> dict[str, Any]:
    """WebSocket request fields. The WS reference lists voice_id and text as the
    only required fields and does not take output_format — audio comes back as
    base64 PCM in chunk frames regardless."""
    body = {
        "text": text if text is not None else unique_text(),
        "voice_id": VOICE,
        "model": MODEL,
        "sample_rate": SAMPLE_RATE,
    }
    body.update(overrides)
    return body


def sse_audio(frame: dict[str, Any]) -> str | None:
    """Pull base64 audio out of a frame whichever shape it arrives in.

    The WS reference documents audio at `data.audio`; the SSE guide shows a
    flat `audio`. Reading only one shape means every sample on that transport
    silently records no audio and the whole column reports as failed."""
    data = frame.get("data")
    if isinstance(data, dict) and data.get("audio"):
        return data["audio"]
    return frame.get("audio")


def is_terminal(frame: dict[str, Any]) -> bool:
    """End-of-stream across both documented spellings. Missing the terminal
    frame leaves the reader blocking until the 60-second idle timeout."""
    return (frame.get("done") is True
            or frame.get("status") in ("complete", "200")
            or str(frame.get("status")) == "200")


# --------------------------------------------------------------------------
# Evidence records
# --------------------------------------------------------------------------


@dataclass
class Sample:
    """One timed synthesis. Times in seconds; None means it never happened."""

    transport: str
    host: str
    run: int
    ok: bool = False
    status: int | None = None
    request_id: str | None = None
    t_connect: float | None = None        # TCP+TLS+WS handshake, WS only
    t_first_byte: float | None = None     # response headers
    t_first_message: float | None = None  # first frame of any kind (WS/SSE)
    t_first_audio: float | None = None    # first byte of actual audio
    t_total: float | None = None
    audio_bytes: int = 0
    audio_seconds: float | None = None
    audio_sha: str | None = None
    audio_magic: str | None = None
    text_sent: str | None = None
    error: str | None = None
    note: str | None = None

    def rtf(self) -> float | None:
        """Real-time factor: synthesis wall time / audio duration. Below 1.0
        means generation outpaces playback, which is the bar for an agent that
        must not stutter."""
        if self.t_total and self.audio_seconds:
            return self.t_total / self.audio_seconds
        return None

    def log(self) -> None:
        f = lambda v: f"{v:.3f}s" if v is not None else "-"  # noqa: E731
        say(f"{self.transport:26} run={self.run:<3} http={self.status} "
            f"connect={f(self.t_connect)} first_msg={f(self.t_first_message)} "
            f"first_audio={f(self.t_first_audio)} total={f(self.t_total)} "
            f"bytes={self.audio_bytes} sha={self.audio_sha} "
            f"magic={self.audio_magic} xid={self.request_id}")
        if self.note:
            say(f"{'':26} note: {self.note}")
        if self.error:
            say(f"{'':26} error: {self.error[:200]}")


@dataclass
class Probe:
    """One docs-vs-reality assertion."""

    name: str
    claim: str        # what the docs say
    expected: str
    observed: str
    verdict: str      # MATCH | MISMATCH | UNDOCUMENTED | INCONCLUSIVE
    evidence: dict[str, Any] = field(default_factory=dict)
    doc_url: str | None = None

    def log(self) -> None:
        mark = {"MATCH": "ok  ", "MISMATCH": "FAIL",
                "UNDOCUMENTED": "??  ", "INCONCLUSIVE": "--  "}.get(self.verdict, "    ")
        say(f"  [{mark}] {self.name}: expected {self.expected} / observed {self.observed}")


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

# Container signatures, so a response can be checked against the requested
# output_format instead of trusted. Requesting wav and receiving raw PCM is a
# silent failure: the bytes arrive, the length looks right, playback is noise.
MAGIC = {
    "wav": ["52494646"],                        # RIFF
    "mp3": ["fffb", "fff3", "fffa", "494433"],  # frame sync, or ID3
    "pcm": [], "ulaw": [], "alaw": [],          # headerless by definition
}


def fingerprint(data: bytes) -> tuple[str, str]:
    """sha256 prefix and leading bytes. The hash detects caching and
    non-determinism; the magic bytes detect a container mismatch."""
    return hashlib.sha256(data).hexdigest()[:12], data[:4].hex()


def container_ok(fmt: str, magic_hex: str) -> bool | None:
    """True/False where the format has a signature, None where it has none."""
    sigs = MAGIC.get(fmt, [])
    if not sigs:
        return None
    return any(magic_hex.lower().startswith(s) for s in sigs)


def wav_info(data: bytes) -> dict[str, Any]:
    """Parse the RIFF/WAVE header: what the file actually says it is.

    Magic bytes only prove the container. They say nothing about whether the
    server honoured the requested sample rate, and a 44100 Hz file returned for
    a 24000 Hz request plays at the wrong pitch rather than failing.
    """
    import struct

    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {"error": "not a RIFF/WAVE file", "leading": data[:4].hex()}
    info: dict[str, Any] = {"riff_size": struct.unpack("<I", data[4:8])[0]}
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        try:
            size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        except struct.error:
            break
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            fmt, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            info.update(audio_format=fmt, channels=ch, sample_rate=rate,
                        bits_per_sample=bits)
        elif cid == b"data":
            info["data_bytes"] = size
            info["data_offset"] = pos + 8
        pos += 8 + size + (size % 2)   # chunks are word-aligned
    if {"sample_rate", "channels", "bits_per_sample", "data_bytes"} <= set(info):
        per_sec = info["sample_rate"] * info["channels"] * info["bits_per_sample"] // 8
        if per_sec:
            info["duration_s"] = round(info["data_bytes"] / per_sec, 3)
    return info


def check_audio(data: bytes, requested_rate: int, requested_format: str,
                content_type: str | None) -> list[str]:
    """Return a list of complaints; empty means the response matched the
    request. Covers container, declared sample rate, channel count and the
    Content-Type header."""
    problems: list[str] = []

    ok = container_ok(requested_format, data[:4].hex())
    if ok is False:
        problems.append(f"container: leading bytes {data[:4].hex()} are not "
                        f"a {requested_format} signature")

    if requested_format == "wav":
        w = wav_info(data)
        if "error" in w:
            problems.append(f"wav header: {w['error']}")
        else:
            if w.get("sample_rate") != requested_rate:
                problems.append(f"sample_rate: requested {requested_rate}, "
                                f"file declares {w.get('sample_rate')}")
            if w.get("channels") != 1:
                problems.append(f"channels: expected mono, file declares "
                                f"{w.get('channels')}")
            if w.get("bits_per_sample") != 16:
                problems.append(f"bit depth: docs say 16-bit, file declares "
                                f"{w.get('bits_per_sample')}")
            declared = w.get("data_bytes")
            offset = w.get("data_offset")
            if declared and offset is not None:
                actual = len(data) - offset
                if abs(declared - actual) > 64:
                    problems.append(f"data chunk declares {declared} bytes, "
                                    f"payload carries about {actual}: "
                                    f"truncated or padded")

    if content_type:
        expect = {"wav": ("audio/wav", "audio/x-wav", "audio/wave"),
                  "mp3": ("audio/mpeg", "audio/mp3"),
                  "pcm": ("audio/pcm", "application/octet-stream", "audio/l16"),
                  "ulaw": ("audio/basic", "audio/pcmu", "application/octet-stream"),
                  "alaw": ("audio/pcma", "application/octet-stream")}
        allowed = expect.get(requested_format, ())
        if allowed and not any(a in content_type.lower() for a in allowed):
            problems.append(f"Content-Type: {content_type!r} for an "
                            f"output_format of {requested_format}")
    return problems


def pcm_seconds(n_bytes: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Duration of 16-bit mono PCM: 2 bytes per sample, one channel, as the
    streaming page states for the default output."""
    return n_bytes / (2 * sample_rate)


def write_wav(path: str, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Wrap raw PCM so the audio can be listened to. Quality is part of QA; a
    latency table cannot tell you the voice clipped."""
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(samples: list[Sample], attr: str = "t_first_audio") -> dict[str, Any]:
    """Latency over successful samples only. Rate-limited samples are counted
    separately and never enter the distribution: a 429 is the absence of a
    measurement, not a slow one, and averaging it in would be nonsense."""
    vals = [getattr(s, attr) for s in samples
            if s.ok and getattr(s, attr) is not None]
    rate_limited = sum(1 for s in samples if s.status == 429)
    errors = sum(1 for s in samples if not s.ok and s.status != 429)
    if not vals:
        return {"n": 0, "errors": errors, "rate_limited": rate_limited}
    return {
        "n": len(vals),
        "errors": errors,
        "rate_limited": rate_limited,
        "min": round(min(vals) * 1000, 1),
        "p50": round(pct(vals, 0.50) * 1000, 1),
        # p95 at n=20 interpolates between the 19th and 20th sample, so it is
        # roughly "second worst" rather than a tail estimate. Reported as
        # indicative only; min/p50/max are the defensible figures.
        "p95_indicative": round(pct(vals, 0.95) * 1000, 1),
        "max": round(max(vals) * 1000, 1),
        "stdev": round(statistics.stdev(vals) * 1000, 1) if len(vals) > 1 else 0.0,
    }


def table(rows: list[tuple[str, dict[str, Any]]], title: str) -> None:
    say()
    say(title)
    hdr = (f"{'transport':<28} {'n':>3} {'err':>4} {'429':>4} "
           f"{'min':>8} {'p50':>8} {'max':>8} {'stdev':>8} {'p95*':>8}")
    say(hdr)
    say("-" * len(hdr))
    for label, s in rows:
        if not s.get("n"):
            say(f"{label:<28} {'0':>3} {s.get('errors', 0):>4} "
                f"{s.get('rate_limited', 0):>4}   (no successful samples)")
            continue
        say(f"{label:<28} {s['n']:>3} {s['errors']:>4} {s['rate_limited']:>4} "
            f"{s['min']:>8.1f} {s['p50']:>8.1f} {s['max']:>8.1f} "
            f"{s['stdev']:>8.1f} {s['p95_indicative']:>8.1f}")
    say("milliseconds. 429s are excluded from the distribution and counted "
        "separately.")
    say("* p95 at this sample count is about the second-worst sample, not a "
        "tail estimate. Indicative only; quote min/p50/max.")


def curl_for(url: str, body: dict[str, Any] | None, headers: dict[str, str]) -> str:
    """The reproducible command for the report, with the key redacted."""
    parts = ["curl", "-sS", "-X", "POST", url]
    for k, v in headers.items():
        shown = "Bearer $SMALLEST_API_KEY" if k.lower() == "authorization" else v
        parts += ["-H", f"{k}: {shown}"]
    if body is not None:
        parts += ["-d", json.dumps(body, separators=(",", ":"))]
    parts += ["-D", "-", "-o", "out.bin", "-w", "\\n%{time_starttransfer} %{time_total}\\n"]
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def environment() -> dict[str, Any]:
    """Context the numbers are meaningless without. A TTFA of 900 ms on a 15 ms
    link and the same figure on a 250 ms link are different findings."""
    env: dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    for host in HOSTS.values():
        env[f"dns:{host}"] = _resolve(host)
        env[f"rtt_ms:{host}"] = _ping(host)
    return env


def _resolve(host: str) -> str:
    try:
        return socket.gethostbyname(host)
    except OSError as exc:
        return f"unresolved ({exc})"


def _ping(host: str, count: int = 3) -> float | None:
    """Median ICMP RTT in ms, parsed from `time=` only.

    Splitting the whole output on whitespace and taking every number sweeps up
    `ttl=64`, `icmp_seq=1` and the byte count, so the median lands somewhere
    between the TTL and the payload size and the reported RTT is fiction.
    """
    import re

    flag = "-n" if platform.system() == "Windows" else "-c"
    try:
        out = subprocess.run(["ping", flag, str(count), host],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    # Linux/macOS: "time=12.3 ms"   Windows: "time=12ms" or "time<1ms"
    times = [float(m) for m in re.findall(r"time[=<]\s*([\d.]+)\s*ms", out)]
    return round(statistics.median(times), 1) if times else None


def balance_checkpoint(label: str) -> None:
    """Record where billing stood at each section boundary.

    There is no verified Waves balance endpoint in the API reference, so this
    does not guess one — guessing a path is how a 404 turns into a fabricated
    finding. It logs a timestamped marker to read against the console's credit
    balance afterwards, which is what makes a surprising charge attributable to
    a section rather than to the run as a whole.

    Worth watching: an 8001-character request is over the documented 1-8000
    limit, so a 200 there means the call was synthesized and billed.
    """
    say(f"--- balance checkpoint: {label} "
        f"({datetime.now(IST).strftime('%H:%M:%S')} IST) - note the console "
        f"credit balance now; stop the run if it has moved unexpectedly")


def save_json(path: str, payload: Any) -> None:
    """Write evidence with every known secret removed. scrub() walks the whole
    structure, so an API key or minted token that reached an exception string
    or a response header never lands in a file that gets attached to a report."""
    with open(path, "w") as fh:
        json.dump(scrub(payload), fh, indent=2)