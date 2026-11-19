from __future__ import annotations

import asyncio
import base64
import json
import random
import statistics
import sys
import time

import common as C

try:
    import httpx
    import websockets
except ImportError:
    sys.exit('pip install httpx "websockets>=14"')

if tuple(int(p) for p in websockets.__version__.split(".")[:1]) < (14,):
    sys.exit(f'websockets {websockets.__version__} is too old; '
             'pip install "websockets>=14"')

clock: C.Clock = time.perf_counter


async def rest(client: httpx.AsyncClient, run: int) -> C.Sample:
    """
    One request, audio streamed back as a single body. The slow baseline.
    """
    body: dict[str, str] = C.tts_body()
    s: C.Sample = C.Sample(transport="REST sync", host=C.HOST, run=run, text_sent=body["text"])
    t0 = clock()
    try:
        async with client.stream("POST", f"https://{C.HOST}{C.PATH_TTS}", json=body, headers=C.auth()) as r:
            s.status = r.status_code
            s.request_id = r.headers.get("x-request-id")
            s.t_first_byte = clock() - t0
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                if chunk and s.t_first_audio is None:
                    s.t_first_audio = clock() - t0
                buf += chunk
            s.t_total = clock() - t0
            s.audio_bytes = len(buf)
            if r.status_code != 200:
                s.error = buf[:300].decode("utf-8", "replace")
                return s
            s.audio_sha, s.audio_magic = C.fingerprint(bytes(buf))
            s.audio_seconds = C.pcm_seconds(len(buf))
            problems = C.check_audio(bytes(buf), C.SAMPLE_RATE, C.OUTPUT_FORMAT, r.headers.get("content-type"))
            if problems:
                s.note = "; ".join(problems)
            s.ok = True
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    return s


async def sse(client: httpx.AsyncClient, run: int) -> C.Sample:
    """
    One request, audio streamed back as data: frames.
    """
    body = C.tts_body()
    s = C.Sample(transport="SSE", host=C.HOST, run=run, text_sent=body["text"])
    headers = {**C.auth(), "Accept": "text/event-stream"}
    buf = bytearray()
    t0 = clock()
    try:
        async with client.stream("POST", f"https://{C.HOST}{C.PATH_TTS_LIVE}", json=body, headers=headers) as r:
            s.status = r.status_code
            s.request_id = r.headers.get("x-request-id")
            s.t_first_byte = clock() - t0
            if r.status_code != 200:
                s.error = (await r.aread())[:300].decode("utf-8", "replace")
                return s
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                if s.t_first_message is None:
                    s.t_first_message = clock() - t0
                try:
                    frame = json.loads(payload)
                except json.JSONDecodeError:
                    s.note = f"non-JSON data frame: {payload[:60]!r}"
                    continue
                audio = C.sse_audio(frame)
                if audio:
                    if s.t_first_audio is None:
                        s.t_first_audio = clock() - t0
                    buf += base64.b64decode(audio)
                if C.is_terminal(frame):
                    break
            s.t_total = clock() - t0
            s.audio_bytes = len(buf)
            if buf:
                s.audio_sha, s.audio_magic = C.fingerprint(bytes(buf))
                s.audio_seconds = C.pcm_seconds(len(buf))
            s.ok = s.t_first_audio is not None
            if not s.ok:
                s.error = "stream completed with no audio frame"
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    return s


def connect(token: str | None = None, query: str | None = None, **kw) -> websockets.client.WebSocketClientProtocol:
    url = f"wss://{C.HOST}{C.PATH_TTS_LIVE}" + (f"?{query}" if query else "")
    return websockets.connect(
        url, additional_headers={"Authorization": f"Bearer {token or C.API_KEY}"},
        open_timeout=20, **kw)


async def synthesize(ws, body: dict, s: C.Sample) -> None:
    """Send one request on an open socket and time the audio."""
    t1 = clock()
    await ws.send(json.dumps(body))
    buf = bytearray()
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=30)
        if s.t_first_message is None:
            s.t_first_message = clock() - t1
        if isinstance(msg, bytes):
            if s.t_first_audio is None:
                s.t_first_audio = clock() - t1
            buf += msg
            s.note = "server sent raw binary frames (docs describe JSON+base64)"
            continue
        frame = json.loads(msg)
        status = frame.get("status")
        if status == "chunk":
            if s.t_first_audio is None:
                s.t_first_audio = clock() - t1
            a = (frame.get("data") or {}).get("audio") or frame.get("audio")
            if a:
                buf += base64.b64decode(a)
        elif status == "complete":
            break
        elif status == "word_timestamp":
            continue
        elif status == "error" or frame.get("error") or frame.get("error_code"):
            s.error = json.dumps(frame)[:300]
            break
        if not s.request_id:
            s.request_id = frame.get("request_id") or frame.get("session_id")
    s.t_total = clock() - t1
    s.audio_bytes = len(buf)
    if buf:
        s.audio_sha, s.audio_magic = C.fingerprint(bytes(buf))
        s.audio_seconds = C.pcm_seconds(len(buf))
    s.ok = s.t_first_audio is not None and s.error is None


async def ws_cold(run: int, text: str | None = None) -> C.Sample:
    """Fresh connection per synthesis: the handshake is on the critical path.
    What a naive integration gets.
    """
    body = C.ws_body(text) if text else C.ws_body()
    s = C.Sample(transport="WS cold (new conn)", host=C.HOST, run=run,
                 text_sent=body["text"])
    t0 = clock()
    try:
        async with connect() as ws:
            s.t_connect = clock() - t0
            await synthesize(ws, body, s)
            if s.t_first_message is not None:
                s.t_first_message += s.t_connect
            if s.t_first_audio is not None:
                s.t_first_audio += s.t_connect
            if s.t_total is not None:
                s.t_total += s.t_connect
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    return s


async def ws_warm(ws, run: int, context_id: str) -> C.Sample:
    """
    One segment inside an open continuation: no handshake, socket reused.
    """
    body = C.ws_body(context_id=context_id, max_buffer_delay_ms=0, **{"continue": True})
    s = C.Sample(transport="WS warm (continuation)", host=C.HOST, run=run, text_sent=body["text"])
    try:
        await synthesize(ws, body, s)
        leftover: int = 0
        extra_complete: int = 0
        while True:
            try:
                msg: str = await asyncio.wait_for(ws.recv(), timeout=0.4)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            leftover += 1
            if isinstance(msg, str):
                try:
                    if json.loads(msg).get("status") == "complete":
                        extra_complete += 1
                except Exception:  # noqa: BLE001
                    pass
            if leftover > 400:
                break
        if leftover:
            s.note = ((s.note or "") +
                      f" leftover_frames={leftover} "
                      f"extra_complete={extra_complete} - the fragment was "
                      "split into more than one segment; this sample's "
                      "timing is not trustworthy").strip()
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
    return s


async def close_context(ws, context_id: str) -> None:
    """Close the context's input, then the socket. A closing frame may take
    `continue: false` with no text; nothing is echoed back and a close with
    nothing buffered gets no response at all, so this does not wait."""
    try:
        await ws.send(json.dumps({"context_id": context_id, "continue": False}))
        await asyncio.sleep(0.5)
    except Exception:  # noqa: BLE001
        pass


async def cache_check(client: httpx.AsyncClient) -> None:
    """Send one sentence twice and compare the audio byte for byte, with a
    near-identical control to rule out coincidence.
    """
    C.say()
    C.say("Cache / determinism")
    fixed = C.tts_body(f"Cache probe {C.nonce()}.")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    async def once(label: str, body: dict) -> C.Sample:
        s = C.Sample(transport=label, host=C.HOST, run=0, text_sent=body["text"])
        t0 = clock()
        r = await client.post(url, json=body, headers=C.auth())
        s.status, s.t_total = r.status_code, clock() - t0
        s.t_first_audio = s.t_total
        s.audio_bytes = len(r.content)
        s.request_id = r.headers.get("x-request-id")
        if r.status_code == 200 and r.content:
            s.audio_sha, s.audio_magic = C.fingerprint(r.content)
            s.ok = True
        s.log()
        return s

    a = await once("identical #1", fixed)
    await asyncio.sleep(1)
    b = await once("identical #2", fixed)
    await asyncio.sleep(1)
    await once("control (+'!')", dict(fixed, text=fixed["text"] + "!"))

    if not (a.ok and b.ok):
        return C.say("  inconclusive: a call failed")
    same = a.audio_sha == b.audio_sha
    C.say(f"  identical audio bytes: {same}  ({a.audio_sha} vs {b.audio_sha})")
    if same and a.t_total and b.t_total < a.t_total * 0.5:
        C.say("  -> looks cached. Benchmarks reusing one sentence measure cache "
              "hits, and caching is documented nowhere.")
    elif same:
        C.say("  -> deterministic synthesis, no obvious cache")
    else:
        C.say("  -> non-deterministic: identical input, different audio. Breaks "
              "audio regression testing.")


async def audio_correctness(client: httpx.AsyncClient) -> None:
    """Does the audio match what was asked for, beyond the container magic?
    """
    C.say()
    C.say("Audio correctness")
    for rate in (8000, 24000, 44100):
        body = C.tts_body(output_format="wav", sample_rate=rate)
        r = await client.post(f"https://{C.HOST}{C.PATH_TTS}", json=body,
                              headers=C.auth())
        await asyncio.sleep(C.GAP)
        if r.status_code != 200 or not r.content:
            C.say(f"  requested wav @ {rate}: http {r.status_code}, no audio")
            continue
        info = C.wav_info(r.content)
        problems = C.check_audio(r.content, rate, "wav",
                                 r.headers.get("content-type"))
        C.say(f"  requested wav @ {rate}: file declares "
              f"{info.get('sample_rate')} Hz, {info.get('channels')} ch, "
              f"{info.get('bits_per_sample')}-bit, "
              f"{info.get('duration_s')}s, "
              f"content-type={r.headers.get('content-type')!r}")
        for p in problems:
            C.say(f"    -> {p}")
        if not problems:
            C.say("    -> matches the request")

    for fmt in ("pcm", "ulaw"):
        body = C.tts_body(output_format=fmt)
        r = await client.post(f"https://{C.HOST}{C.PATH_TTS}", json=body,
                              headers=C.auth())
        await asyncio.sleep(C.GAP)
        if r.status_code != 200 or not r.content:
            C.say(f"  requested {fmt}: http {r.status_code}, no audio")
            continue
        problems = C.check_audio(r.content, C.SAMPLE_RATE, fmt,
                                 r.headers.get("content-type"))
        C.say(f"  requested {fmt}: http {r.status_code}, {len(r.content)} bytes, "
              f"content-type={r.headers.get('content-type')!r}"
              + ("" if not problems else "  -> " + "; ".join(problems)))


async def rtf_check(client: httpx.AsyncClient) -> None:
    """Real-time factor on a long utterance.
    """
    C.say()
    C.say("Real-time factor on a long utterance "
          f"({len(C.TEXT_LONG)} chars, <1.0 beats playback)")
    for label, n, note in (("REST sync", 3, "synthesis only"),
                           ("WS cold", 3, "includes the handshake")):
        rtfs: list[float] = []
        for i in range(n):
            if label == "REST sync":
                s = C.Sample(transport=label, host=C.HOST, run=i)
                t0 = clock()
                r = await client.post(f"https://{C.HOST}{C.PATH_TTS}",
                                      json=C.tts_body(C.TEXT_LONG),
                                      headers=C.auth())
                s.t_total = clock() - t0
                if r.status_code == 200 and r.content:
                    info = C.wav_info(r.content)
                    s.audio_seconds = (info.get("duration_s")
                                       or C.pcm_seconds(len(r.content)))
                    s.ok = True
            else:
                s = await ws_cold(1000 + i, text=C.TEXT_LONG)
            if s.rtf():
                rtfs.append(s.rtf())
            await asyncio.sleep(C.GAP)
        if rtfs:
            med = statistics.median(rtfs)
            C.say(f"  {label:<12} n={len(rtfs)}  median RTF {med:.3f}  "
                  f"({'faster' if med < 1 else 'slower'} than real time; "
                  f"{note})")
        else:
            C.say(f"  {label:<12} no usable samples")


async def slot_release(client: httpx.AsyncClient) -> None:
    """Is the single concurrency slot released when a request dies badly?

    Three ways to find out, all of which a real client hits: abandon a REST
    response mid-body, drop a WebSocket mid-synthesis, and ask for REST while a
    WS synthesis is still running. A slot that stays held after a client
    disappears means one dropped call wedges the account until a timeout
    expires, which is a severe reliability finding and invisible in any
    happy-path benchmark.
    """
    C.say()
    C.say("Slot release after failure")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    async def probe_free(label: str) -> None:
        """Immediately ask for synthesis. 200 means the slot is free."""
        t0 = clock()
        try:
            r = await client.post(url, json=C.tts_body(), headers=C.auth(),
                                  timeout=30)
            took = (clock() - t0) * 1000
            verdict = ("slot free" if r.status_code == 200 else
                       "SLOT STILL HELD" if r.status_code == 429 else
                       f"unexpected {r.status_code}")
            C.say(f"  {label:<44} http={r.status_code} {took:.0f}ms  -> {verdict}")
            if r.status_code == 429:
                C.say(f"{'':48} retry-after={r.headers.get('retry-after')!r}")
        except Exception as exc:  # noqa: BLE001
            C.say(f"  {label:<44} {type(exc).__name__}: {str(exc)[:70]}")

    long_text = C.TEXT_LONG[:400]
    try:
        await client.post(url, json=C.tts_body(long_text), headers=C.auth(),
                          timeout=0.2)
        C.say("  client timeout did not fire - synthesis finished inside 200ms, "
              "so this case did not test a mid-synthesis disconnect")
    except Exception as exc:  # noqa: BLE001
        C.say(f"  disconnected during synthesis after 200ms "
              f"({type(exc).__name__} on a {len(long_text)}-char text)")
    await probe_free("immediately after a mid-synthesis disconnect")
    await asyncio.sleep(C.GAP)

    ws = None
    try:
        ws = await connect().__aenter__()
        await ws.send(json.dumps(C.ws_body(C.TEXT_LONG)))
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        mid = frame.get("status") != "complete"
        await ws.close(code=1001)
        C.say(f"  closed a WS with code 1001 "
              f"{'mid-synthesis' if mid else 'but it had already completed'}")
    except Exception as exc:  # noqa: BLE001
        C.say(f"  WS mid-synthesis close raised {type(exc).__name__}")
    finally:
        if ws is not None:
            try:
                await ws.close(code=1001)
            except Exception:  # noqa: BLE001
                pass
    await probe_free("immediately after a WS mid-synthesis close")
    await asyncio.sleep(C.GAP)

    try:
        async with connect() as ws:
            await ws.send(json.dumps(C.ws_body(C.TEXT_LONG * 2)))
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if first.get("status") == "complete":
                C.say("  cross-transport: WS finished before REST could be "
                      "issued; inconclusive, lengthen the text")
            else:
                r = await client.post(url, json=C.tts_body(), headers=C.auth(),
                                      timeout=30)
                still_running = True
                try:
                    nxt = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    still_running = nxt.get("status") != "complete"
                except asyncio.TimeoutError:
                    pass
                if not still_running:
                    C.say(f"  cross-transport: http={r.status_code} but the WS "
                          "had completed by then - inconclusive")
                elif r.status_code == 429:
                    C.say(f"  REST during an active WS synthesis  http=429  -> "
                          "matches the documented single slot shared across "
                          f"transports; retry-after="
                          f"{r.headers.get('retry-after')!r}")
                elif r.status_code == 200:
                    C.say("  REST during an active WS synthesis  http=200  -> "
                          "the slot is NOT shared across transports, so the "
                          "limits page understates capacity (confirmed: the WS "
                          "was still synthesizing)")
                else:
                    C.say(f"  REST during an active WS synthesis  "
                          f"http={r.status_code} - unexpected")
    except Exception as exc:  # noqa: BLE001
        C.say(f"  cross-transport probe raised {type(exc).__name__}: "
              f"{str(exc)[:70]}")
    await asyncio.sleep(C.GAP)
    await probe_free("after the cross-transport probe")


async def sample_audio(client: httpx.AsyncClient) -> None:
    """One listenable artifact. A latency table says nothing about whether the
    audio was correct."""
    r = await client.post(f"https://{C.HOST}{C.PATH_TTS}",
                          json=C.tts_body(C.TEXT, output_format="pcm"),
                          headers=C.auth())
    if r.status_code == 200 and r.content:
        C.write_wav("sample.wav", r.content)
        C.say(f"\nsample.wav written ({len(r.content)} bytes, "
              f"{C.pcm_seconds(len(r.content)):.2f}s) - listen to it")


async def main() -> None:
    C.require_key()
    C.say("=" * 70)
    C.say(f"LATENCY  host={C.HOST}  model={C.MODEL}  voice={C.VOICE}  "
          f"{C.OUTPUT_FORMAT}@{C.SAMPLE_RATE}  runs={C.RUNS}")

    env = C.environment()
    C.say(f"client: {env['platform']}")
    for k, v in env.items():
        if k.startswith("rtt_ms:"):
            C.say(f"  rtt {k.split(':')[1]:<26} {v} ms")

    groups: dict[str, list[C.Sample]] = {
        "REST sync (full file)": [],
        "SSE (first audio frame)": [],
        "WS cold (incl. handshake)": [],
        "WS warm (continuation)": [],
    }

    limits = httpx.Limits(max_keepalive_connections=4, max_connections=8)
    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        warm = None
        warm_ctx = f"qa-warm-{C.nonce(8)}"
        rate_limited_other: list[tuple[int, str, int | None, str]] = []
        try:
            warm = await connect().__aenter__()
            C.say(f"warm WS connected, continuation context_id={warm_ctx}")
        except Exception as exc:  # noqa: BLE001
            C.say(f"warm WS unavailable: {type(exc).__name__}: {exc}")

        C.say()
        for i in range(C.RUNS + 1):
            warmup = i == 0
            plan = [
                ("REST sync (full file)", lambda: rest(client, i)),
                ("SSE (first audio frame)", lambda: sse(client, i)),
                ("WS cold (incl. handshake)", lambda: ws_cold(i)),
            ]
            if warm is not None:
                plan.append(("WS warm (continuation)",
                             lambda: ws_warm(warm, i, warm_ctx)))
            random.shuffle(plan)

            samples: list[tuple[str, C.Sample]] = []
            for label, make in plan:
                samples.append((label, await make()))
                await asyncio.sleep(C.GAP)
            C.say(f"{'':26} round {i} order: "
                  + " -> ".join(lbl.split(' ')[0] + lbl.split(' ')[1][:1]
                                for lbl, _ in plan))

            for label, s in samples:
                if warmup:
                    s.note = (s.note or "") + " [warmup, discarded]"
                s.log()
                if not warmup:
                    groups[label].append(s)
                is_warm = s.transport.startswith("WS warm")
                if not is_warm and (s.status == 429 or not s.ok):
                    rate_limited_other.append(
                        (i, s.transport, s.status,
                         (s.error or "")[:160] or "no error text"))
                if s.status == 429:
                    C.say(f"{'':26} 429 on a sequential run"
                          + (" WHILE the warm continuation socket is open - "
                             "the open context is the likely cause, not the "
                             "product; re-run with the warm column dropped "
                             "before reporting anything about limits"
                             if warm is not None else
                             " - either the slot was not released, or the gap "
                             "is too short"))

            dead = [s for _, s in samples
                    if s.transport.startswith("WS warm") and not s.ok
                    and "closed" in (s.error or "").lower()]
            if dead:
                C.say(f"{'':26} continuation socket closed - harness issue with "
                      "the context_id setup, not a product finding; dropping "
                      "the warm column")
                warm = None

        if warm is not None:
            await close_context(warm, warm_ctx)
            await warm.close()

        warm_samples = groups["WS warm (continuation)"]
        if warm_samples:
            split = [x for x in warm_samples if x.note and "leftover_frames" in x.note]
            if split:
                C.say(f"\nWARM COLUMN SUSPECT: {len(split)}/{len(warm_samples)} "
                      "fragments were split into more than one segment, so "
                      "leftover chunks landed in later samples. Treat the warm "
                      "figures as unusable and shorten the fragment text.")
            else:
                C.say(f"\nwarm continuation: {len(warm_samples)} segments, one "
                      "`complete` each, no leftover frames - timings are sound")
        if rate_limited_other and warm_samples:
            C.say(f"WARM SOCKET SUSPECT: {len(rate_limited_other)} failed or "
                  "rate-limited samples on other transports while the "
                  "continuation was open. The open context is the likely "
                  "holder of the single concurrency slot. Do NOT report this "
                  "as a limits finding; re-run without the warm column to "
                  "separate harness from product.")
            for run_i, transport, status, err in rate_limited_other[:12]:
                C.say(f"    run {run_i:<3} {transport:<26} http={status} {err}")
        elif warm_samples:
            C.say("no failures on other transports while the continuation was "
                  "open - the open context does not appear to hold the slot")

        if warm_samples:
            wf = [x.t_first_audio for x in warm_samples
                  if x.ok and x.t_first_audio]
            if wf:
                med_ms = C.pct(wf, 0.5) * 1000
                near_3s = sum(1 for v in wf if 2.7 <= v <= 3.4)
                if near_3s >= max(2, len(wf) // 3):
                    C.say(f"WARM COLUMN IS BUFFER-BOUND: {near_3s}/{len(wf)} "
                          f"samples sit between 2.7s and 3.4s (p50 "
                          f"{med_ms:.0f}ms). That is the max_buffer_delay_ms "
                          "timer, not model latency. Do not quote this as a "
                          "WebSocket latency figure.")
                else:
                    C.say(f"warm column p50 {med_ms:.0f}ms, "
                          f"{near_3s}/{len(wf)} samples near the 3s buffer "
                          "default - not buffer-bound")

        C.save_json("bench_results.json", {
            "environment": env, "status": "samples complete, extras pending",
            "config": {"host": C.HOST, "runs": C.RUNS, "model": C.MODEL,
                       "voice": C.VOICE, "sample_rate": C.SAMPLE_RATE,
                       "output_format": C.OUTPUT_FORMAT},
            "summary": {k: C.summarize(v) for k, v in groups.items() if v},
            "samples": [s for v in groups.values() for s in v],
        })

        rows = [(k, C.summarize(v)) for k, v in groups.items() if v]
        C.table(rows, "Time to first audio")
        C.table([(k, C.summarize(v, "t_total")) for k, v in groups.items() if v],
                "Time to complete audio")


        for k in ("SSE (first audio frame)", "WS cold (incl. handshake)",
                  "WS warm (continuation)"):
            v = groups.get(k) or []
            fm = [s.t_first_message for s in v if s.ok and s.t_first_message]
            fa = [s.t_first_audio for s in v if s.ok and s.t_first_audio]
            if fm and fa:
                gap = (C.pct(fa, 0.5) - C.pct(fm, 0.5)) * 1000
                C.say(f"\n{k}: first message p50 {C.pct(fm, 0.5)*1000:.1f} ms, "
                      f"first audio p50 {C.pct(fa, 0.5)*1000:.1f} ms (gap {gap:+.1f} ms)")
                if gap > 20:
                    C.say("  -> 'time to first message' would understate real "
                          "TTFA; the opening frame carries no audio")

        conns = [s.t_connect for s in groups["WS cold (incl. handshake)"] if s.t_connect]
        if conns:
            cold = C.summarize(groups["WS cold (incl. handshake)"])
            hot = C.summarize(groups["WS warm (continuation)"])
            C.say(f"\nWS handshake p50 {C.pct(conns, 0.5)*1000:.1f} ms")
            if cold.get("p50") and hot.get("p50"):
                C.say(f"  pooling saves {cold['p50'] - hot['p50']:.1f} ms at p50")

        all_samples = [s for v in groups.values() for s in v]

        ids = [s for s in all_samples if s.request_id]
        C.say(f"\ncorrelation id present on {len(ids)}/{len(all_samples)} samples")
        if not ids:
            C.say("  -> none exposed, though the error reference tells you to "
                  "quote x-request-id in support tickets")

        limited = [s for s in all_samples if s.status == 429]
        if limited:
            C.say(f"\nRate-limited on a sequential run ({len(limited)}) - "
                  "excluded from all latency figures")
            for s in limited[:10]:
                C.say(f"  {s.transport:<22} run {s.run:<3} retry-after unknown; "
                      f"see probes.py limits group")

        failures = [s for s in all_samples if not s.ok and s.status != 429]
        if failures:
            C.say(f"\nErrors ({len(failures)})")
            for s in failures[:15]:
                C.say(f"  {s.transport:<22} run {s.run:<3} http={s.status} "
                      f"{(s.error or '')[:100]}")

        flagged = [s for s in all_samples if s.note and "->" not in (s.note or "")
                   and "warmup" not in (s.note or "")]
        if flagged:
            C.say(f"\nSamples with audio or protocol notes ({len(flagged)})")
            for s in flagged[:10]:
                C.say(f"  {s.transport:<22} run {s.run:<3} {s.note[:140]}")

        C.balance_checkpoint("after latency samples")
        await cache_check(client)
        await rtf_check(client)
        await audio_correctness(client)
        C.balance_checkpoint("after audio correctness")
        await slot_release(client)
        await sample_audio(client)
        C.balance_checkpoint("end of bench")

    C.save_json("bench_results.json", {
        "environment": env,
        "config": {"host": C.HOST, "runs": C.RUNS, "model": C.MODEL,
                   "voice": C.VOICE, "sample_rate": C.SAMPLE_RATE,
                   "output_format": C.OUTPUT_FORMAT},
        "curl": C.curl_for(f"https://{C.HOST}{C.PATH_TTS}", C.tts_body(C.TEXT),
                           {"Authorization": "Bearer KEY",
                            "Content-Type": "application/json"}),
        "summary": {k: C.summarize(v) for k, v in groups.items() if v},
        "samples": all_samples,
    })
    C.say("\nraw samples -> bench_results.json")


if __name__ == "__main__":
    asyncio.run(main())