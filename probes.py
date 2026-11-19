from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import traceback
from typing import Any

import common as C

try:
    import httpx
    import websockets
except ImportError:
    sys.exit('pip install httpx "websockets>=14"')

if tuple(int(p) for p in websockets.__version__.split(".")[:1]) < (14,):
    sys.exit(f'websockets {websockets.__version__} is too old; '
             'pip install "websockets>=14"')

DOCS = {
    "auth": "https://docs.smallest.ai/api-reference/authentication",
    "token": "https://docs.smallest.ai/api-reference/token-based-authentication",
    "limits": "https://docs.smallest.ai/api-reference/concurrency-and-limits",
    "errors": "https://docs.smallest.ai/models/troubleshooting/error-reference",
    "intro": "https://docs.smallest.ai/api-reference/introduction",
    # Guides.
    "stream": "https://docs.smallest.ai/models/text-to-speech/streaming",
    "sync": "https://docs.smallest.ai/models/text-to-speech/sync-async",
    # API reference pages, which are authoritative where a guide disagrees.
    # Check these BEFORE labelling anything UNDOCUMENTED: the guides omit the
    # idle timeout, keep-alive and error frames that the reference specifies.
    "ws": "https://docs.smallest.ai/api-reference/models/text-to-speech/tts",
    "synth": "https://docs.smallest.ai/api-reference/models/"
             "text-to-speech/synthesize-speech",
    "sse": "https://docs.smallest.ai/api-reference/models/"
           "text-to-speech/synthesize-speech-sse",
}

RESULTS: list[C.Probe] = []

# Anything that is an access-control observation rather than a documentation
# finding goes here and is written to security_private.json. It does not belong
# in the report body or on a call; responsible disclosure is a private channel.
SECURITY: list[dict] = []
SECURITY_METHOD: str | None = None


def record(**kw) -> C.Probe:
    p = C.Probe(**kw)
    p.log()
    RESULTS.append(p)
    return p


def record_private(*, name: str, claim: str, expected: str, observed: str,
                   status: int | None, evidence: dict,
                   doc_url: str | None = None) -> None:
    """Record a probe whose failure mode is an access-control issue.

    The verdict comes from the status, not from "did it breach":
      200/201 - the documented restriction did not hold. An access-control
                finding: the detail goes to SECURITY and the public log gets a
                GENERIC entry. The probe name is withheld too, because
                "token blocked: voice clones -> recorded privately" names the
                affected route as surely as printing the status would. All such
                records share one public name, so the public log cannot even
                tell you which of them fired.
      403     - documented behaviour, MATCH, logged in full.
      anything else (401, 404, 429, 500) - the restriction may hold, but this
                did not demonstrate it. UNDOCUMENTED, not MATCH: scoring a 500
                as "matches the documented 403" is a false pass.
    """
    if status in (200, 201):
        SECURITY.append({
            "topic": name, "claim": claim, "expected": expected,
            "observed": observed, "evidence": evidence, "doc_url": doc_url,
            "handling": "private disclosure to Smallest AI only; keep out of "
                        "the QA report and off any call",
        })
        record(name="access-control probe", claim="see private notes",
               expected="see private notes", observed="recorded privately",
               verdict="INCONCLUSIVE", evidence={}, doc_url=None)
    else:
        record(name=name, claim=claim, expected=expected, observed=observed,
               verdict="MATCH" if status == 403 else "UNDOCUMENTED",
               evidence=evidence, doc_url=doc_url)


def body_of(r: httpx.Response, limit: int = 300) -> str:
    try:
        return json.dumps(r.json())[:limit]
    except Exception:  # noqa: BLE001
        return r.text[:limit] if r.text else f"<{len(r.content)} bytes binary>"


def json_obj(r: httpx.Response) -> dict:
    """The body as a JSON object, or an empty dict. A documented 2xx carrying
    an unexpected body is itself a finding, so it has to be recorded rather
    than raise and abort every probe that had not run yet.
    """
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        return {}
    return j if isinstance(j, dict) else {}


def shape(r: httpx.Response) -> str:
    """The introduction page claims error bodies carry a machine-readable
    error_type or code. Is that true?"""
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        return "non-JSON body"
    if not isinstance(j, dict):
        return f"JSON {type(j).__name__}, not an object"
    machine = next((k for k in ("error_type", "code") if k in j), None)
    return f"keys={sorted(j.keys())} machine_readable={machine or 'no'}"


def ws_connect(token: str | None = None, query: str | None = None,
               key_in_query: bool = False, **kw):
    """Open a TTS WebSocket.

    The Authorization header is dropped ONLY when key_in_query is set, i.e. in
    the ?api_key= / ?token= probes that exist to test exactly that path. Keying
    this off "is there a query string" instead silently de-authenticates every
    other query-string probe — ?timeout=5 and ?timeout=600 would connect with
    no credential at all, get rejected at the handshake, and the idle-timeout,
    error-frame and 180s-cap results would all be measuring a 401.

    additional_headers requires websockets>=14; on 12 and 13 the TypeError is
    raised when the connection is awaited rather than when connect() is called,
    so a try/except around the constructor never fires. The version is asserted
    at import instead.
    """
    url = f"wss://{C.HOST}{C.PATH_TTS_LIVE}" + (f"?{query}" if query else "")
    hdr = None if key_in_query else {"Authorization": f"Bearer {token or C.API_KEY}"}
    return websockets.connect(url, additional_headers=hdr, open_timeout=20, **kw)


# --------------------------------------------------------------------------


async def auth_header(client: httpx.AsyncClient) -> None:
    C.say("\n[auth] Authorization header handling")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    record(
        name="key prefix",
        claim="error reference: check that the key starts with sk_",
        expected="starts with sk_",
        observed=f"starts with {C.API_KEY[:3]!r}",
        verdict="MATCH" if C.API_KEY.startswith("sk_") else "MISMATCH",
        evidence={"prefix": C.API_KEY[:3], "length": len(C.API_KEY)},
        doc_url=DOCS["errors"],
    )

    # Happy path first, so a failure below is clearly about the header and not
    # about the account.
    r = await client.post(url, json=C.tts_body(), headers=C.auth())
    record(
        name="valid bearer key",
        claim="auth page: Authorization: Bearer YOUR_API_KEY on every request",
        expected="200 with audio",
        observed=f"{r.status_code}, {len(r.content)} bytes",
        verdict="MATCH" if r.status_code == 200 else "MISMATCH",
        evidence={"status": r.status_code,
                  "content_type": r.headers.get("content-type"),
                  "x_request_id": r.headers.get("x-request-id")},
        doc_url=DOCS["auth"],
    )

    for label, hdrs, strict in [
        ("missing header", {}, True),
        ("bare key, no Bearer", {"Authorization": C.API_KEY}, True),
        ("quoted key", {"Authorization": f'Bearer "{C.API_KEY}"'}, True),
        ("double space", {"Authorization": f"Bearer  {C.API_KEY}"}, True),
        ("trailing whitespace", {"Authorization": f"Bearer {C.API_KEY} "}, True),
        ("invalid key", {"Authorization": "Bearer sk_not_a_real_key_000"}, True),
        ("lowercase bearer", {"Authorization": f"bearer {C.API_KEY}"}, False),
    ]:
        try:
            r = await client.post(url, json=C.tts_body(),
                                  headers={"Content-Type": "application/json",
                                           **hdrs})
        except Exception as exc:  # noqa: BLE001
            record(
                name=f"401: {label}",
                claim="error reference: 401 when the Authorization header is "
                      "missing or invalid; single space, no quotes, no "
                      "trailing whitespace",
                expected="401" if strict else "401 or 200 (docs silent on case)",
                observed=f"never sent: the HTTP client refused the header "
                         f"({type(exc).__name__}), so the server never saw it "
                         f"and the docs cannot be checked this way",
                verdict="INCONCLUSIVE",
                evidence={"client_error": type(exc).__name__},
                doc_url=DOCS["errors"],
            )
            await asyncio.sleep(C.GAP)
            continue
        record(
            name=f"401: {label}",
            claim="error reference: 401 when the Authorization header is missing "
                  "or invalid; single space, no quotes, no trailing whitespace",
            expected="401" if strict else "401 or 200 (docs silent on case)",
            observed=f"{r.status_code}  {shape(r)}",
            verdict="MATCH" if r.status_code == 401
                    else ("UNDOCUMENTED" if not strict else "MISMATCH"),
            evidence={"status": r.status_code, "body": body_of(r),
                      "x_request_id": r.headers.get("x-request-id")},
            doc_url=DOCS["errors"],
        )
        await asyncio.sleep(C.GAP)


# --------------------------------------------------------------------------


async def mint(client: httpx.AsyncClient, ttl=None, *, path=None, token=None):
    r = await client.post(
        f"https://{C.HOST}{path or C.PATH_MINT}",
        json={} if ttl is None else {"ttl_seconds": ttl},
        headers=C.auth(token),
    )
    # Register immediately, before the token can reach a log line, an exception
    # string or an evidence file.
    try:
        C.register_secret(r.json().get("access_token"))
    except Exception:  # noqa: BLE001
        pass
    return r


async def tokens(client: httpx.AsyncClient) -> str | None:
    C.say("\n[tokens] minting, TTL bounds, route scoping")

    r = await mint(client)
    ok = r.status_code == 201
    j = json_obj(r) if ok else {}
    want = {"access_token", "token_type", "expires_in", "expires_at"}
    record(
        name="mint, default TTL",
        claim="token page: POST /console/v1/auth/token returns 201 with "
              "access_token, token_type, expires_in and expires_at; default 300",
        expected="201, expires_in 300, all four fields",
        observed=f"{r.status_code}, expires_in={j.get('expires_in')}, "
                 f"fields={sorted(j.keys()) if ok else body_of(r, 140)}",
        verdict="MATCH" if ok and j.get("expires_in") == 300 and want <= set(j)
                else "MISMATCH",
        evidence={"status": r.status_code, "keys": sorted(j.keys()),
                  "expires_in": j.get("expires_in")},
        doc_url=DOCS["token"],
    )
    token = j.get("access_token")

    for ttl, expect in [(29, "400"), (30, "201"), (900, "201"), (901, "400"),
                        (0, "400"), (-1, "400"), (60.5, "400"), ("300", "400")]:
        r = await mint(client, ttl)
        got = str(r.status_code)
        record(
            name=f"ttl_seconds={ttl!r}",
            claim="token page: ttl_seconds 30 to 900; 400 if outside the range "
                  "or not an integer",
            expected=expect,
            observed=f"{got}  " + (f"expires_in={json_obj(r).get('expires_in')}"
                                   if got == "201" else shape(r)),
            verdict="MATCH" if got == expect else "MISMATCH",
            evidence={"ttl": ttl, "status": r.status_code, "body": body_of(r, 180)},
            doc_url=DOCS["token"],
        )
        await asyncio.sleep(C.GAP)   # the mint endpoint is rate limited

    if not token:
        C.say("  no token minted; skipping scope checks")
        return None

    r = await client.post(f"https://{C.HOST}{C.PATH_TTS}",
                          json=C.tts_body(), headers=C.auth(token))
    record(
        name="token on TTS (allowed)",
        claim="token page: tokens work for TTS, STT, speech-to-speech and "
              "listing public voices",
        expected="200",
        observed=f"{r.status_code}, {len(r.content)} bytes",
        verdict="MATCH" if r.status_code == 200 else "MISMATCH",
        evidence={"status": r.status_code, "body": body_of(r, 140)},
        doc_url=DOCS["token"],
    )

    # Paths copied from their API reference pages. A guessed path 404s, and a
    # 404 here would read as "the documented restriction is not enforced",
    # which is a fabricated finding.
    for label, method, path in [
        ("voice clones", "GET", C.PATH_VOICE_CLONES),
        ("pronunciation dicts", "GET", C.PATH_PRON_DICTS),
        ("LLM chat", "POST", C.PATH_CHAT),
    ]:
        try:
            r = await client.request(method, f"https://{C.HOST}{path}",
                                     json={} if method == "POST" else None,
                                     headers={"Authorization": f"Bearer {token}"})
        except Exception as exc:  # noqa: BLE001
            record(name=f"token blocked: {label}",
                   claim="token page: route not accessible with a token",
                   expected="403", observed=type(exc).__name__,
                   verdict="INCONCLUSIVE", doc_url=DOCS["token"])
            continue
        # A token reaching a route the docs say it cannot reach is an
        # access-control result, not a docs-accuracy one. Detail goes private.
        if r.status_code == 404:
            record(
                name=f"token blocked: {label}",
                claim="token page: voice cloning, pronunciation dictionaries, "
                      "analytics and LLM chat are not accessible with a token",
                expected="403",
                observed=f"404 - path may be wrong, not a finding",
                verdict="INCONCLUSIVE",
                evidence={"status": 404, "path": path,
                          "path_source": "copied from the endpoint's API "
                                         "reference page"},
                doc_url=DOCS["token"],
            )
        else:
            record_private(
                name=f"token blocked: {label}",
                claim="token page: voice cloning, pronunciation dictionaries, "
                      "analytics and LLM chat are not accessible with a token",
                expected="403",
                observed=f"{r.status_code}  {shape(r)}",
                status=r.status_code,
                evidence={"status": r.status_code, "path": path,
                          "body": body_of(r, 180)},
                doc_url=DOCS["token"],
            )
        await asyncio.sleep(C.GAP)

    r = await mint(client, token=token)
    record_private(
        name="token cannot mint a token",
        claim="token page: only an API key can mint; a token gets 403",
        expected="403",
        observed=f"{r.status_code}  {shape(r)}",
        status=r.status_code,
        evidence={"status": r.status_code, "body": body_of(r, 180)},
        doc_url=DOCS["token"],
    )

    r = await mint(client, path=C.PATH_MINT_LEGACY)
    record(
        name="legacy mint endpoint",
        claim="token page: older tokens came from /waves/v1/auth/token; "
              "migrating means changing only the URL",
        expected="still 201, or a clear deprecation signal",
        observed=f"{r.status_code}  deprecation={r.headers.get('deprecation')}",
        verdict="MATCH" if r.status_code == 201 else "UNDOCUMENTED",
        evidence={"status": r.status_code,
                  "deprecation_header": r.headers.get("deprecation"),
                  "body": body_of(r, 180)},
        doc_url=DOCS["token"],
    )
    return token


async def token_expiry(client: httpx.AsyncClient) -> None:
    """The token page states an open WebSocket keeps working after its token
    expires, because the token is only checked at connect. 30s is the shortest
    legal TTL, so this costs 35 seconds of waiting."""
    C.say("\n[tokens] WS survives token expiry")
    r = await mint(client, 30)
    if r.status_code != 201:
        record(name="WS survives token expiry",
               claim="token page: an open WS keeps working after expiry",
               expected="audio after expiry",
               observed=f"could not mint (http {r.status_code})",
               verdict="INCONCLUSIVE", doc_url=DOCS["token"])
        return
    token = json_obj(r).get("access_token")
    if not token:
        record(name="WS survives token expiry",
               claim="token page: an open WS keeps working after expiry",
               expected="audio after expiry",
               observed="minted 201 but no access_token in the body: "
                        f"{body_of(r, 140)}",
               verdict="INCONCLUSIVE", doc_url=DOCS["token"])
        return
    try:
        async with ws_connect(token) as ws:
            C.say("  socket open; waiting 35s for the 30s token to expire...")
            await asyncio.sleep(35)
            # ws_body, not tts_body: output_format is a REST-only field and
            # sending it over the socket risks a validation error that would
            # read as "the expired token was rejected".
            await ws.send(json.dumps(C.ws_body()))
            got = False
            deadline = time.time() + 30
            while time.time() < deadline:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                frame = json.loads(msg) if isinstance(msg, str) else {"status": "chunk"}
                if frame.get("status") == "chunk":
                    got = True
                    break
                if frame.get("status") == "complete" or frame.get("error"):
                    break
            record(
                name="WS survives token expiry",
                claim="token page: an already open WebSocket keeps working after "
                      "its token expires; the token is checked only on open",
                expected="audio still streams 35s after a 30s token",
                observed="audio received" if got else "no audio after expiry",
                verdict="MATCH" if got else "MISMATCH",
                evidence={"ttl": 30, "waited_s": 35},
                doc_url=DOCS["token"],
            )
    except Exception as exc:  # noqa: BLE001
        record(name="WS survives token expiry",
               claim="token page: an open WS keeps working after expiry",
               expected="audio after expiry",
               observed=f"{type(exc).__name__}: {str(exc)[:110]}",
               verdict="MISMATCH", evidence={"ttl": 30}, doc_url=DOCS["token"])


# --------------------------------------------------------------------------


async def limits(client: httpx.AsyncClient) -> None:
    C.say("\n[limits] concurrency, 429 shape, connection cap")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    # Long text, or three short syntheses can complete in sequence before the
    # next lands: 200, 200, 429 looks like a broken limit when in fact nothing
    # ever overlapped.
    rs = await asyncio.gather(
        *[client.post(url, json=C.tts_body(C.TEXT_LONG), headers=C.auth())
          for _ in range(3)],
        return_exceptions=True)
    codes = [getattr(x, "status_code", type(x).__name__) for x in rs]
    limited = [x for x in rs if getattr(x, "status_code", None) == 429]
    record(
        name="concurrency = 1",
        claim="limits page: 1 TTS request concurrency; a second request sent "
              "during processing returns 429",
        expected="exactly one 200, the rest 429",
        observed=f"statuses {codes}",
        verdict="MATCH" if codes.count(200) == 1 and len(limited) == 2 else "MISMATCH",
        evidence={"statuses": [str(c) for c in codes],
                  "bodies": [body_of(x, 140) for x in rs if hasattr(x, "status_code")]},
        doc_url=DOCS["limits"],
    )

    if limited:
        r = limited[0]
        ra = r.headers.get("retry-after")
        record(
            name="429 Retry-After",
            claim="error reference: retry with exponential backoff from 500 ms "
                  "doubling to 30 s, and honour Retry-After if present",
            expected="Retry-After present, so a client can back off correctly",
            observed=f"Retry-After={ra!r}" if ra else "Retry-After absent",
            verdict="MATCH" if ra else "UNDOCUMENTED",
            evidence={"headers": dict(r.headers), "body": body_of(r)},
            doc_url=DOCS["errors"],
        )
        sh = shape(r)
        record(
            name="429 body shape",
            claim="introduction page: error bodies carry a machine-readable "
                  "error_type or code",
            expected="JSON with error_type or code",
            observed=sh,
            verdict="MATCH" if "machine_readable=" in sh
                    and "machine_readable=no" not in sh else "MISMATCH",
            evidence={"body": body_of(r)},
            doc_url=DOCS["intro"],
        )

    # Documented allowance is five connections. This confirms five are accepted
    # and stops there: attempting a sixth would mean holding six at once, which
    # exceeds the documented limit, and a rejection then cannot be read as
    # correct enforcement rather than the account being pushed past its own
    # terms. Whether the sixth is refused cleanly or hangs is worth knowing and
    # is recorded as untested, with the doc cited, rather than guessed at.
    sockets = []
    failed_at = None
    try:
        for i in range(5):
            try:
                sockets.append(await ws_connect().__aenter__())
            except Exception as exc:  # noqa: BLE001
                failed_at = (i + 1, f"{type(exc).__name__}: {str(exc)[:140]}")
                break
        record(
            name="WS connection cap (5 accepted)",
            claim="limits and streaming pages: up to 5 WebSocket connections "
                  "may be open, with one request processed across all of them",
            expected="all 5 connections accepted",
            observed=f"{len(sockets)} of 5 accepted"
                     + (f"; #{failed_at[0]} rejected: {failed_at[1]}" if failed_at else ""),
            verdict="MATCH" if len(sockets) == 5 else "MISMATCH",
            evidence={"opened": len(sockets), "first_failure": failed_at,
                      "sixth_connection": "not attempted - would exceed the "
                                          "documented allowance of five"},
            doc_url=DOCS["limits"],
        )

        # With five open, test the documented serialization: one request
        # processed across all of them. This stays inside the stated limits.
        #
        # Waiting for socket 2's first chunk and calling it "concurrent" is
        # wrong: a queued request still delivers a chunk eventually, well
        # inside any sane timeout. What distinguishes queued from concurrent is
        # ORDER — does socket 2's first chunk arrive before socket 1 says
        # `complete`? Both sockets are read together and the sequence recorded.
        if len(sockets) >= 2:
            try:
                await sockets[0].send(json.dumps(C.ws_body(C.TEXT_LONG)))
                await sockets[1].send(json.dumps(C.ws_body(C.TEXT_LONG)))

                order: list[str] = []
                s1_complete_at: int | None = None
                s2_first_chunk_at: int | None = None

                async def drain(idx: int, ws) -> None:
                    nonlocal s1_complete_at, s2_first_chunk_at
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=40)
                        if isinstance(msg, bytes):
                            continue
                        f = json.loads(msg)
                        st = f.get("status")
                        if st == "chunk":
                            if idx == 1 and s2_first_chunk_at is None:
                                s2_first_chunk_at = len(order)
                                order.append("socket2:first_chunk")
                        elif st == "complete":
                            if idx == 0:
                                s1_complete_at = len(order)
                                order.append("socket1:complete")
                            else:
                                order.append("socket2:complete")
                            return
                        elif st == "error":
                            order.append(f"socket{idx+1}:error "
                                         f"{json.dumps(f)[:80]}")
                            return

                await asyncio.wait_for(
                    asyncio.gather(drain(0, sockets[0]), drain(1, sockets[1]),
                                   return_exceptions=True),
                    timeout=120)

                concurrent = (s2_first_chunk_at is not None
                              and s1_complete_at is not None
                              and s2_first_chunk_at < s1_complete_at)
                queued = (s2_first_chunk_at is not None
                          and s1_complete_at is not None
                          and s2_first_chunk_at > s1_complete_at)
                refused = (s2_first_chunk_at is None
                           and any(o.startswith("socket2:error")
                                   for o in order))
                record(
                    name="one request across 5 sockets",
                    claim="limits page: up to 5 connections may be open, but "
                          "only one request is processed across all of them",
                    expected="socket 2's first chunk arrives only after socket "
                             "1 sends complete (queued), or socket 2 is refused",
                    observed=("socket 2 produced audio BEFORE socket 1 "
                              "completed - genuinely concurrent" if concurrent
                              else "socket 2's audio came after socket 1 "
                                   "completed - queued, as documented" if queued
                              else "socket 2 was refused with an error frame "
                                   "instead of audio - the documented "
                                   "alternative" if refused
                              else f"inconclusive ordering: {order[:6]}"),
                    verdict="MISMATCH" if concurrent
                            else "MATCH" if queued or refused
                            else "INCONCLUSIVE",
                    evidence={"sockets_open": len(sockets), "event_order": order,
                              "s1_complete_index": s1_complete_at,
                              "s2_first_chunk_index": s2_first_chunk_at},
                    doc_url=DOCS["limits"],
                )
            except Exception as exc:  # noqa: BLE001
                record(name="one request across 5 sockets",
                       claim="limits page: one request processed across all "
                             "open connections",
                       expected="queued or refused",
                       observed=f"{type(exc).__name__}: {str(exc)[:110]}",
                       verdict="INCONCLUSIVE",
                       evidence={"sockets_open": len(sockets)},
                       doc_url=DOCS["limits"])
    finally:
        # Drain and close cleanly, then pause. Closing five sockets while
        # synthesis is still running leaves the server finishing work the
        # client has walked away from, and the next section fires immediately
        # into whatever slot state that leaves behind — manufacturing 429s that
        # look like the product's rate limiting.
        for s in sockets:
            try:
                while True:
                    try:
                        await asyncio.wait_for(s.recv(), timeout=1.5)
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break
                await s.close()
            except Exception:  # noqa: BLE001
                pass
        C.say("  drained and closed; settling for 5s before the next section")
        await asyncio.sleep(5)


# --------------------------------------------------------------------------


async def params(client: httpx.AsyncClient) -> None:
    C.say("\n[params] documented bounds and bad input")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    async def post(**over):
        r = await client.post(url, json=C.tts_body(**over), headers=C.auth())
        await asyncio.sleep(C.GAP)
        return r

    # The sync page lists four discrete rates; the streaming page describes a
    # continuous 8000-44100 range. 22050 is legal under one reading only.
    for sr, note in [(22050, "in the continuous range, not in the discrete list"),
                     (48000, "above the documented maximum"),
                     (7999, "below the documented minimum")]:
        r = await post(sample_rate=sr)
        discrete = sr in (8000, 16000, 24000, 44100)
        in_range = 8000 <= sr <= 44100
        record(
            name=f"sample_rate={sr}",
            claim="sync page lists 8000/16000/24000/44100; streaming page says "
                  "the range is 8000 to 44100 Hz",
            expected="400 if the list governs, 200 if the range does"
                     if in_range and not discrete else ("4xx" if not in_range else "200"),
            observed=f"{r.status_code}  ({note})",
            verdict="UNDOCUMENTED" if in_range and not discrete
                    else ("MATCH" if (not in_range and r.status_code >= 400) else "MISMATCH"),
            evidence={"sample_rate": sr, "status": r.status_code, "body": body_of(r, 180)},
            doc_url=DOCS["sync"],
        )

    for sp, expect in [(0.5, 200), (2.0, 200), (0.4, 400), (2.1, 400), (3, 400)]:
        r = await post(speed=sp)
        record(name=f"speed={sp}", claim="sync page: speed 0.5 to 2.0",
               expected=str(expect), observed=str(r.status_code),
               verdict="MATCH" if r.status_code == expect else "MISMATCH",
               evidence={"speed": sp, "status": r.status_code, "body": body_of(r, 140)},
               doc_url=DOCS["sync"])

    # The interesting question for bad input is not whether it fails but
    # whether the 4xx names the offending field. A 500 is a validation gap; a
    # 200 on an unknown voice or language is worse, because the caller gets
    # audio in the wrong voice with nothing signalling it.
    # Each case names the specific field whose name must appear in the error.
    # A generic vocabulary check is worthless: "invalid" and "required" show up
    # in almost any error body, and a bare "text" substring matches "context",
    # so a body saying "invalid context_id" would score as correctly naming the
    # `text` field. None means there is no single field to name.
    drop = lambda k: {kk: vv for kk, vv in C.tts_body().items() if kk != k}  # noqa: E731
    for label, payload, raw, field in [
        ("empty text", C.tts_body(text=""), False, "text"),
        ("whitespace-only text", C.tts_body(text="   "), False, "text"),
        (f"text {C.TEXT_MAX + 1} chars", C.tts_body(text="a" * (C.TEXT_MAX + 1)),
         False, "text"),
        ("missing text", drop("text"), False, "text"),
        ("missing voice_id", drop("voice_id"), False, "voice_id"),
        ("unknown voice_id", C.tts_body(voice_id="does_not_exist_000"),
         False, "voice_id"),
        ("unknown model", C.tts_body(model="lightning_v9_turbo"), False, "model"),
        ("output_format flac", C.tts_body(output_format="flac"),
         False, "output_format"),
        ("speed out of range", C.tts_body(speed=9), False, "speed"),

        ("malformed JSON", "{not json", True, None),
        ("empty body", "", True, None),
        ("JSON array not object", "[]", True, None),
    ]:
        if raw:
            r = await client.post(url, content=payload, headers=C.auth())
        else:
            r = await client.post(url, json=payload, headers=C.auth())
        await asyncio.sleep(C.GAP)
        is4xx = 400 <= r.status_code < 500
        body_text = (r.text or "")
        # Word-boundary match, not a substring one. A plain `"text" in body`
        # still matches "invalid context_id", which is the same false positive
        # one level down: `text` is literally inside `context_id`.
        named = field is not None and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", body_text
        ) is not None
        # The over-limit case is the one where a 200 costs money: 1-8000 chars
        # is the documented range, so anything accepted past it was billed.
        billed = (r.status_code == 200 and label.endswith("chars"))
        record(
            name=f"bad input: {label}",
            claim="synthesize-speech reference: text is 1-8000 characters after "
                  "trimming and whitespace-only is rejected; the introduction "
                  "page promises a machine-readable error_type or code",
            expected=(f"4xx whose body names {field!r}" if field
                      else "4xx with a parse error"),
            observed=f"{r.status_code}"
                     + (f" (names {field!r})" if named else
                        f" (does not name {field!r})" if is4xx and field else "")
                     + ("  BILLED: accepted past the documented limit"
                        if billed else "")
                     + f"  {shape(r)}",
            verdict="MATCH" if is4xx and (named or field is None)
                    else ("MISMATCH" if not is4xx else "UNDOCUMENTED"),
            evidence={"case": label, "status": r.status_code, "body": body_of(r),
                      "expected_field": field, "field_named": named,
                      "possibly_billed": billed,
                      "sent": str(payload)[:150],
                      "x_request_id": r.headers.get("x-request-id")},
            doc_url=DOCS["synth"],
        )

    # Unknown language code gets its own treatment. The voices-and-languages
    # page says `language` takes an ISO 639-1 code but does NOT say what
    # happens to an unknown one - no documented rejection, and no documented
    # English fallback either. The only mismatch behaviour it documents is for
    # model/voice pairing. So a 200 here is a silent fallback the docs never
    # describe (UNDOCUMENTED), and a 4xx is also undocumented, just safer.
    # Calling either a MISMATCH would assert a doc claim that does not exist.
    r = await client.post(url, json=C.tts_body(language="xx"), headers=C.auth())
    await asyncio.sleep(C.GAP)
    accepted = r.status_code == 200
    record(
        name="unknown language code",
        claim="voices-and-languages page: `language` takes an ISO 639-1 code. "
              "The page does not state whether an unknown code is rejected or "
              "silently accepted, and documents no English fallback",
        expected="documented behaviour either way",
        observed=(f"200, {len(r.content)} bytes of audio - accepted silently; "
                  "listen to sample_lang_xx.wav to hear which language it used"
                  if accepted else f"{r.status_code}  {shape(r)}"),
        verdict="UNDOCUMENTED",
        evidence={"language": "xx", "status": r.status_code,
                  "audio_bytes": len(r.content) if accepted else 0,
                  "body": "" if accepted else body_of(r),
                  "why_not_mismatch": "no documented behaviour to contradict; "
                                      "a silent fallback is the finding"},
        doc_url="https://docs.smallest.ai/models/text-to-speech/voices-languages",
    )
    if accepted and r.content:
        # Raw bytes: output_format is already wav, so wrapping it again with
        # write_wav would produce a WAV containing a WAV header as audio.
        try:
            with open("sample_lang_xx.wav", "wb") as fh:
                fh.write(r.content)
        except OSError:
            pass

    # Two pages give two different rules for an omitted `model`, and the
    # response does not say which model answered — so no status code can
    # resolve it. Passing this on a 200 would hide the contradiction behind a
    # successful request. The contradiction is recorded directly, with both
    # statements and both URLs, and the 200 only confirms that a caller who
    # omits the field gets audio without being told which pool served it.
    r = await client.post(url, json=drop("model"), headers=C.auth())
    identified = {k: v for k, v in r.headers.items()
                  if "model" in k.lower() or "lightning" in str(v).lower()}
    record(
        name="omitted model: two docs pages disagree",
        claim="sync/async guide: 'model | string | lightning_v3.1' as the "
              "default. Synthesize Speech reference: an omitted model is "
              "derived from the voice - Pro-catalog voices use "
              "lightning_v3.1_pro, all other voices use lightning_v3.1. These "
              "are different rules: the guide gives a fixed default, the "
              "reference gives a voice-dependent one",
        expected="one documented rule, and a response that says which model "
                 "served the request",
        observed=f"http {r.status_code}, "
                 + (f"model identified by {identified}" if identified
                    else "no header identifies the model that answered, so "
                         "neither rule can be confirmed from the response"),
        verdict="MISMATCH" if not identified else "UNDOCUMENTED",
        evidence={
            "status": r.status_code,
            "guide_says": "model defaults to lightning_v3.1",
            "guide_url": DOCS["sync"],
            "reference_says": "an omitted model is derived from the voice; Pro "
                              "voices use lightning_v3.1_pro, others "
                              "lightning_v3.1",
            "reference_url": DOCS["synth"],
            "voice_used": C.VOICE,
            "under_guide_rule": "lightning_v3.1",
            "under_reference_rule": "lightning_v3.1_pro (meher is a Pro voice)",
            "resolvable_from_response": bool(identified),
            "all_headers": dict(r.headers),
        },
        doc_url=DOCS["synth"])
    await asyncio.sleep(C.GAP)

    r = await client.post(url, json=C.tts_body(),
                          headers={**C.auth(), "x-expire-content": "true"})
    record(name="x-expire-content on non-enterprise",
           claim="auth page: on plans other than Enterprise the header is ignored",
           expected="200, silently ignored",
           observed=str(r.status_code),
           verdict="MATCH" if r.status_code == 200 else "MISMATCH",
           evidence={"status": r.status_code, "body": body_of(r, 140)},
           doc_url=DOCS["auth"])


# --------------------------------------------------------------------------


async def deprecation(client: httpx.AsyncClient) -> None:
    """lightning_v3.1 retires on 30 November. Does the API say so?

    The introduction page states deprecated *endpoints* return a
    `Deprecation: true` header. Nothing says whether a retiring *model* does.
    It matters because the sync page still documents lightning_v3.1 as the
    default when `model` is omitted: a caller who never names a model is
    pointed at the retiring one and, if nothing in the response warns them,
    finds out when it stops working.
    """
    C.say("\n[deprecation] retiring model signalling")
    url = f"https://{C.HOST}{C.PATH_TTS}"
    headers_of_interest = ("deprecation", "sunset", "warning", "link",
                           "x-model", "x-deprecated")

    # Pair each model with a voice from its own catalog. Sending the Pro voice
    # meher with lightning_v3.1 risks an error about the voice/model mismatch,
    # which would be indistinguishable from a deprecation rejection.
    for model, voice, note in [
        (C.MODEL_RETIRING, C.VOICE_STANDARD, "retiring standard pool"),
        (C.MODEL_PRO, C.VOICE_PRO, "current Pro pool, control case"),
    ]:
        r = await client.post(url,
                              json=C.tts_body(model=model, voice_id=voice),
                              headers=C.auth())
        await asyncio.sleep(C.GAP)
        signals = {k: v for k, v in r.headers.items()
                   if k.lower() in headers_of_interest}
        body_warn = ""
        if not r.headers.get("content-type", "").startswith("audio"):
            body_warn = body_of(r, 160)
        record(
            name=f"deprecation signal: {model} + {voice}",
            claim="introduction page: deprecated endpoints return a "
                  "Deprecation: true header; nothing states how a retiring "
                  "model is signalled",
            expected="a Deprecation or Sunset header, or a warning in the body",
            observed=f"http {r.status_code}, "
                     + (f"signals {signals}" if signals else "no deprecation headers")
                     + (f", body: {body_warn}" if body_warn else ""),
            verdict="MATCH" if signals else "UNDOCUMENTED",
            evidence={"model": model, "voice": voice, "note": note,
                      "status": r.status_code, "all_headers": dict(r.headers)},
            doc_url=DOCS["intro"],
        )

    # The case that actually bites. `model` is derived from the voice when
    # omitted: Pro-catalog voices resolve to lightning_v3.1_pro, every other
    # voice to lightning_v3.1. So dropping the field with a Pro voice like
    # meher still lands on Pro and tests nothing — reaching the retiring
    # standard pool needs a standard voice.
    for voice, pool in [(C.VOICE_STANDARD, "standard -> lightning_v3.1, retiring"),
                        (C.VOICE_PRO, "Pro -> lightning_v3.1_pro, control")]:
        body = {k: v for k, v in C.tts_body(voice_id=voice).items()
                if k != "model"}
        r = await client.post(url, json=body, headers=C.auth())
        await asyncio.sleep(C.GAP)
        signals = {k: v for k, v in r.headers.items()
                   if k.lower() in headers_of_interest}
        record(
            name=f"model omitted, voice={voice}",
            claim="synthesize-speech reference: an omitted model is derived "
                  "from the voice - Pro-catalog voices use lightning_v3.1_pro, "
                  "all other voices use lightning_v3.1",
            expected=f"200 routed to the {pool.split(' ->')[0]} pool, and a "
                     "deprecation signal if the resolved model is retiring",
            observed=f"http {r.status_code}, "
                     + (f"signals {signals}" if signals else
                        "no model identified and no deprecation warning"),
            verdict="MATCH" if signals else "UNDOCUMENTED",
            evidence={"voice": voice, "expected_pool": pool,
                      "status": r.status_code, "all_headers": dict(r.headers),
                      "note": "the response does not appear to echo which "
                              "model served the request, so the resolution "
                              "rule cannot be confirmed from the response "
                              "alone - cross-check with get_voices"},
            doc_url=DOCS["synth"],
        )


async def error_consistency(client: httpx.AsyncClient) -> None:
    """One error schema across REST, SSE and WebSocket, or three?

    The same bad input is sent over all three transports. A client that has to
    parse three different error shapes for the same fault cannot share error
    handling between its HTTP and streaming paths, and the introduction page
    promises one machine-readable shape.
    """
    C.say("\n[errors] shape consistency across transports")
    bad = C.tts_body(voice_id="does_not_exist_000")
    # The WS leg needs the WS request shape. Sending output_format over the
    # socket would make the server complain about the field rather than the
    # unknown voice, so the comparison would be testing our own payload.
    bad_ws = C.ws_body(voice_id="does_not_exist_000")
    shapes: dict[str, str] = {}

    # body_of(), not shape(): shape() returns a human summary with no braces,
    # so keyset() below would find nothing to parse and the comparison would
    # silently run on SSE and WS alone — always reporting a mismatch REST never
    # took part in.
    r = await client.post(f"https://{C.HOST}{C.PATH_TTS}", json=bad, headers=C.auth())
    shapes["REST"] = f"http {r.status_code} {body_of(r, 2000)}"
    await asyncio.sleep(C.GAP)

    try:
        async with client.stream("POST", f"https://{C.HOST}{C.PATH_TTS_LIVE}",
                                 json=bad,
                                 headers={**C.auth(), "Accept": "text/event-stream"}) as sr:
            raw = (await sr.aread()).decode("utf-8", "replace")[:2000]
            shapes["SSE"] = f"http {sr.status_code} {raw}"
    except Exception as exc:  # noqa: BLE001
        shapes["SSE"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    await asyncio.sleep(C.GAP)

    try:
        async with ws_connect() as ws:
            await ws.send(json.dumps(bad_ws))
            msg = await asyncio.wait_for(ws.recv(), timeout=20)
            shapes["WS"] = str(msg)[:2000]
    except Exception as exc:  # noqa: BLE001
        shapes["WS"] = f"closed/raised: {type(exc).__name__}: {str(exc)[:120]}"

    for t, s in shapes.items():
        C.say(f"    {t:<5} {s[:200]}")

    def keyset(s: str) -> frozenset:
        try:
            start = s.index("{")
            return frozenset(json.loads(s[start:]).keys())
        except Exception:  # noqa: BLE001
            return frozenset()

    sets = {t: keyset(s) for t, s in shapes.items()}
    unparsed = [t for t, v in sets.items() if not v]
    distinct = {v for v in sets.values() if v}
    record(
        name="error shape across REST / SSE / WS",
        claim="introduction page: error responses include an HTTP status and a "
              "JSON body with a machine-readable error_type or code",
        expected="one schema for the same fault on all three transports",
        observed=f"{len(distinct)} distinct key set(s): "
                 + "; ".join(f"{t}={sorted(v) if v else 'unparsed'}"
                             for t, v in sets.items())
                 + (f"; not comparable on {', '.join(unparsed)}, so the three "
                    "transports were never all compared" if unparsed else ""),
        verdict=("INCONCLUSIVE" if unparsed
                 else "MATCH" if len(distinct) == 1 else "MISMATCH"),
        evidence={"raw": shapes, "key_sets": {t: sorted(v) for t, v in sets.items()}},
        doc_url=DOCS["intro"],
    )


async def protocol(client: httpx.AsyncClient, token: str | None) -> None:
    C.say("\n[protocol] WebSocket auth, frames, idle policy, liveness")

    # Two pages name two different query parameters for the same job. A browser
    # cannot set WS headers, so one of these has to work and only one is right.
    #
    # This is tested with a short-lived ACCESS TOKEN, never the API key. The
    # token page's whole point is that the key never reaches a client, so
    # putting the key in a query string to see whether it works would be
    # testing a path nobody should use, and a positive result would be an
    # access-control observation rather than a docs finding. Any such result
    # goes to the private security file, not into the report body and not onto
    # a call.
    for param, doc in (("api_key", DOCS["token"]), ("token", DOCS["errors"])):
        claim = ("token page: WebSocket connections can take the token as the "
                 "api_key query parameter, since browsers cannot set headers"
                 if param == "api_key" else
                 "error reference: in browser WebSocket handshakes use "
                 "?token=<KEY> if the endpoint supports it")
        if not token:
            record(name=f"WS query auth ?{param}=", claim=claim,
                   expected="accepted with a minted access token",
                   observed="no access token available; not tested with the API "
                            "key by design",
                   verdict="INCONCLUSIVE", evidence={"param": param},
                   doc_url=doc)
            continue
        try:
            # key_in_query=True is the one place the header is withheld: these
            # two probes exist to test query-string auth itself.
            async with ws_connect(query=f"{param}={token}",
                                  key_in_query=True) as ws:
                await ws.send(json.dumps(C.ws_body()))
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                frame = json.loads(msg) if isinstance(msg, str) else {"status": "chunk"}
                worked = frame.get("status") in ("chunk", "word_timestamp", "complete")
                record(name=f"WS query auth ?{param}=", claim=claim,
                       expected="accepted with a minted access token, audio streams",
                       observed="accepted, audio streamed" if worked
                                else f"accepted, first frame {json.dumps(frame)[:100]}",
                       verdict="MATCH" if worked else "MISMATCH",
                       evidence={"param": param, "credential": "minted access token",
                                 "first_frame": json.dumps(frame)[:280]},
                       doc_url=doc)
        except Exception as exc:  # noqa: BLE001
            record(name=f"WS query auth ?{param}=", claim=claim,
                   expected="accepted with a minted access token, audio streams",
                   observed=f"rejected: {type(exc).__name__}: {str(exc)[:100]}",
                   verdict="MISMATCH",
                   evidence={"param": param, "credential": "minted access token",
                             "error": str(exc)[:280]},
                   doc_url=doc)
        await asyncio.sleep(C.GAP)

    # Methodology note, held separately. Appending it to SECURITY
    # unconditionally meant security_private.json was written on every run even
    # with nothing to disclose, which trains the reader to ignore the file.
    global SECURITY_METHOD
    SECURITY_METHOD = (
        "Query-string auth was exercised only with a short-lived minted access "
        "token, never the API key. Whether the long-lived key is also accepted "
        "in a query string was deliberately not tested: it is an access-control "
        "question rather than a docs-accuracy one. If it is ever established, "
        "it belongs in a private disclosure to Smallest AI, not in a QA report "
        "and not discussed on a call."
    )

    # Frame shape.
    try:
        async with ws_connect() as ws:
            await ws.send(json.dumps(C.ws_body(word_timestamps=True)))
            seen, shapes = [], []
            deadline = time.time() + 40
            while time.time() < deadline:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(msg, bytes):
                    seen.append("binary")
                    shapes.append(f"<binary {len(msg)}B>")
                    continue
                f = json.loads(msg)
                seen.append(f.get("status"))
                if len(shapes) < 4:
                    red = {k: v for k, v in f.items() if k not in ("data", "audio")}
                    if isinstance(f.get("data"), dict):
                        red["data"] = {k: (f"<base64 {len(v)}B>"
                                           if k == "audio" and isinstance(v, str) else v)
                                       for k, v in f["data"].items()}
                    shapes.append(json.dumps(red)[:200])
                if f.get("status") == "complete":
                    break
            record(name="WS frame shape",
                   claim="streaming page: status 'chunk' carries base64 PCM at "
                         "data.audio; 'word_timestamp' when enabled; 'complete' "
                         "ends the stream; session_id and request_id informational",
                   expected="chunk / word_timestamp / complete, audio at data.audio",
                   observed=f"statuses seen: {sorted(set(map(str, seen)))}",
                   verdict="MATCH" if "chunk" in seen and "complete" in seen else "MISMATCH",
                   evidence={"statuses": [str(x) for x in seen[:40]],
                             "first_frames": shapes},
                   doc_url=DOCS["stream"])
    except Exception as exc:  # noqa: BLE001
        record(name="WS frame shape", claim="streaming page frame schema",
               expected="chunk/complete frames",
               observed=f"{type(exc).__name__}: {str(exc)[:100]}",
               verdict="INCONCLUSIVE", doc_url=DOCS["stream"])

    await ws_idle()
    await ws_liveness()


async def _hold_idle(query: str | None, window: float) -> dict:
    """Open a socket, send nothing, and record how the server ends it.

    ping_interval=None disables the library's protocol-level keep-alive. The
    documented inactivity timer is reset by any message the client sends, so
    leaving the library's pings on would keep resetting it and the probe would
    silently prove nothing.
    """
    t0 = time.time()
    out: dict = {"closed_after_s": None, "close_code": None, "close_reason": "",
                 "error_frame": None, "frames": []}
    try:
        async with ws_connect(query=query, ping_interval=None) as ws:
            while time.time() - t0 < window:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed as e:
                    out["closed_after_s"] = round(time.time() - t0, 1)
                    rc = e.rcvd
                    out["close_code"] = rc.code if rc else None
                    out["close_reason"] = (rc.reason if rc else "") or ""
                    break
                out["frames"].append(str(msg)[:160])
                try:
                    f = json.loads(msg)
                    if f.get("status") == "error":
                        out["error_frame"] = f
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        out["close_reason"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


async def ws_idle() -> None:
    """Idle timeout, error frame and ?timeout= override — all documented.

    The TTS WebSocket reference states a 60-second default, an override via a
    positive-integer `?timeout=` honoured verbatim for small values and clamped
    at 180, and an `{"status":"error","message":...}` frame carrying
    "Connection timed out after N seconds of inactivity" before the close. So
    these are MATCH/MISMATCH assertions, not undocumented behaviour — and a
    45-second observation window could never have seen a 60-second close.
    """
    # Fast case first: ?timeout=5 should close after about 5 seconds.
    C.say("  holding an idle socket with ?timeout=5 (expect a close near 5s)...")
    fast = await _hold_idle("timeout=5", window=20)
    closed = fast["closed_after_s"]
    near5 = closed is not None and 3 <= closed <= 12
    record(
        name="WS ?timeout=5 honoured verbatim",
        claim="TTS WebSocket reference: smaller ?timeout= values are honoured "
              "as given, e.g. ?timeout=5 closes after 5 seconds of silence",
        expected="connection closed about 5s after the last message",
        observed=(f"closed after {closed}s" if closed is not None
                  else "still open after 20s"),
        verdict="MATCH" if near5 else "MISMATCH",
        evidence=fast, doc_url=DOCS["ws"],
    )
    record(
        name="WS idle error frame shape",
        claim='TTS WebSocket reference: the server sends {"status":"error",'
              '"message":"Connection timed out after N seconds of inactivity"} '
              "before closing an idle connection",
        expected='a status:"error" frame naming the inactivity timeout',
        observed=(json.dumps(fast["error_frame"])[:160] if fast["error_frame"]
                  else f"no error frame; frames seen: {fast['frames'][:3]}"),
        verdict="MATCH" if fast["error_frame"] else "MISMATCH",
        evidence={"error_frame": fast["error_frame"], "frames": fast["frames"][:5]},
        doc_url=DOCS["ws"],
    )
    record(
        name="WS close code",
        claim="the TTS WebSocket reference documents the idle timeout and its "
              "error frame, but states that close codes are not covered",
        expected="a documented close code for an idle timeout",
        observed=f"observed code={fast['close_code']}, "
                 f"reason={fast['close_reason']!r}",
        verdict="UNDOCUMENTED",
        evidence={"close_code": fast["close_code"],
                  "close_reason": fast["close_reason"],
                  "note": "the reference explicitly lists close codes as not "
                          "covered, so this is a genuine gap rather than one "
                          "I failed to look up"},
        doc_url=DOCS["ws"],
    )

    # Then the documented default, which needs a single wait past 60 seconds.
    C.say(f"  holding an idle socket with no ?timeout= "
          f"(expect a close near {C.WS_IDLE_DEFAULT}s; this waits 65s)...")
    default = await _hold_idle(None, window=65)
    closed = default["closed_after_s"]
    near60 = closed is not None and 50 <= closed <= 65
    record(
        name=f"WS default idle timeout = {C.WS_IDLE_DEFAULT}s",
        claim="TTS WebSocket reference: the default idle timeout is 60 seconds",
        expected="connection closed about 60s after the last message",
        observed=(f"closed after {closed}s" if closed is not None
                  else "still open after 65s"),
        verdict="MATCH" if near60 else "MISMATCH",
        evidence=default, doc_url=DOCS["ws"],
    )

    # The 180-second clamp would need a 185-second observation. Recorded as not
    # tested rather than guessed: what is checked here is only that a value
    # above the cap is accepted at handshake.
    try:
        async with ws_connect(query="timeout=600", ping_interval=None) as ws:
            await ws.send(json.dumps(C.ws_body()))
            await asyncio.wait_for(ws.recv(), timeout=20)
            accepted = True
    except Exception as exc:  # noqa: BLE001
        accepted = f"{type(exc).__name__}: {str(exc)[:90]}"
    record(
        name="WS ?timeout= above the 180s cap",
        claim="TTS WebSocket reference: larger ?timeout= values are clamped to "
              "a maximum of 180 seconds",
        expected="?timeout=600 accepted, then clamped to 180s in effect",
        observed=("accepted and synthesis worked; the clamp itself needs a "
                  "185s idle observation and was not measured"
                  if accepted is True else f"rejected: {accepted}"),
        verdict="MATCH" if accepted is True else "MISMATCH",
        evidence={"timeout_sent": 600, "clamp_s": C.WS_IDLE_MAX,
                  "clamp_verified": False,
                  "why": "a 185-second idle wait per run was judged not worth "
                         "the time; the handshake acceptance is what is claimed"},
        doc_url=DOCS["ws"],
    )


async def ws_liveness() -> None:
    """Application-level keep-alive, which is documented.

    The reference states {"type":"ping"} returns {"type":"pong"} and resets the
    inactivity timer, with keepalive, keep_alive and keep-alive as aliases, and
    says the frame is handled at the API edge and never reaches the model. All
    four spellings are therefore MATCH/MISMATCH assertions.
    """
    for payload, label, documented in [
        ({"type": "ping"}, "ping", True),
        ({"type": "keepalive"}, "keepalive alias", True),
        ({"type": "keep_alive"}, "keep_alive alias", True),
        ({"type": "keep-alive"}, "keep-alive alias", True),
        ({"nonsense_field": True}, "unrecognised message", False),
    ]:
        try:
            async with ws_connect() as ws:
                t0 = time.time()
                await ws.send(json.dumps(payload))
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    raw = str(msg)[:120]
                    try:
                        pong = json.loads(msg).get("type") == "pong"
                    except Exception:  # noqa: BLE001
                        pong = False
                    observed = f"replied in {time.time()-t0:.2f}s: {raw}"
                except asyncio.TimeoutError:
                    observed, pong = "no reply within 10s", False
                except websockets.ConnectionClosed as e:
                    observed = (f"closed, code={e.rcvd.code if e.rcvd else None}, "
                                f"reason={(e.rcvd.reason if e.rcvd else '')!r}")
                    pong = False
            if documented:
                verdict = "MATCH" if pong else "MISMATCH"
            else:
                # The reference states the error frame schema is not covered
                # beyond the idle timeout and CONTENT_FILTER_BLOCKED, so
                # whatever happens here is a genuine gap.
                verdict = "UNDOCUMENTED"
        except Exception as exc:  # noqa: BLE001
            observed, verdict = f"{type(exc).__name__}: {str(exc)[:100]}", "INCONCLUSIVE"
        record(
            name=f"WS keep-alive: {label}",
            claim=('TTS WebSocket reference: {"type":"ping"} is answered with '
                   '{"type":"pong"} and resets the inactivity timer; keepalive, '
                   "keep_alive and keep-alive are accepted aliases"
                   if documented else
                   "TTS WebSocket reference: the error frame schema is not "
                   "covered for anything beyond the idle timeout and "
                   "CONTENT_FILTER_BLOCKED, and client message types beyond the "
                   "request payload and ping are not listed"),
            expected='{"type":"pong"}' if documented
                     else "a documented error frame for an unrecognised message",
            observed=observed, verdict=verdict, evidence={"sent": payload},
            doc_url=DOCS["ws"],
        )
        await asyncio.sleep(C.GAP)


# --------------------------------------------------------------------------


async def meta(client: httpx.AsyncClient) -> None:
    C.say("\n[meta] correlation ids, region pinning")
    url = f"https://{C.HOST}{C.PATH_TTS}"

    r = await client.post(url, json=C.tts_body(), headers=C.auth())
    rid = r.headers.get("x-request-id")
    record(name="x-request-id on success",
           claim="error reference: open a support ticket quoting the "
                 "x-request-id response header",
           expected="present on every response",
           observed=f"present: {rid}" if rid else "absent",
           verdict="MATCH" if rid else "MISMATCH",
           evidence={"headers": dict(r.headers)}, doc_url=DOCS["errors"])
    await asyncio.sleep(C.GAP)

    r = await client.post(url, json={"text": ""}, headers=C.auth())
    rid_err = r.headers.get("x-request-id")
    record(name="x-request-id on error",
           claim="error reference: quote x-request-id when reporting a failure",
           expected="present on error responses too",
           observed=f"present: {rid_err}" if rid_err else "absent on a 4xx",
           verdict="MATCH" if rid_err else "MISMATCH",
           evidence={"status": r.status_code, "headers": dict(r.headers)},
           doc_url=DOCS["errors"])
    await asyncio.sleep(C.GAP)

    # Client-supplied correlation ids ARE documented, on the WS reference:
    # `session_id` and `request_id` are accepted request fields, and frames
    # carry them back as `external_session_id` and `external_request_id`
    # "echoed if you sent them". So this is a MATCH/MISMATCH assertion on both
    # transports, not an undocumented gap — the earlier verdict here was the
    # result of reading the guides and not the reference.
    # Wrapped: an unwrapped `async with` here means one connection error aborts
    # the rest of meta(), taking the region-pinning probes with it.
    mine = "qa-" + C.nonce(10)
    seen: dict = {}
    ws_error = None
    try:
        async with ws_connect() as ws:
            await ws.send(json.dumps(C.ws_body(request_id=mine, session_id=mine)))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                if isinstance(msg, bytes):
                    continue
                f = json.loads(msg)
                for k in ("external_request_id", "external_session_id",
                          "request_id", "session_id"):
                    if k in f and k not in seen:
                        seen[k] = f[k]
                if f.get("status") == "complete":
                    break
    except Exception as exc:  # noqa: BLE001
        ws_error = f"{type(exc).__name__}: {str(exc)[:110]}"
    ws_echoed = any(v == mine for v in seen.values())
    record(name="WS echoes client-supplied request_id",
           claim="TTS WebSocket reference: frames carry external_session_id and "
                 "external_request_id, echoed if you sent session_id and "
                 "request_id on the request",
           expected=f"external_request_id == {mine!r} in the response frames",
           observed=(f"fields seen: {seen}" if seen
                     else f"failed: {ws_error}" if ws_error
                     else "no id fields in frames"),
           verdict="INCONCLUSIVE" if ws_error and not seen
                   else ("MATCH" if ws_echoed else "MISMATCH"),
           evidence={"sent": mine, "id_fields_seen": seen, "error": ws_error},
           doc_url=DOCS["ws"])
    await asyncio.sleep(C.GAP)

    # SSE documents the echo as response headers, so it is checkable there too.
    hdrs: dict = {}
    sse_err = None
    try:
        async with client.stream(
                "POST", f"https://{C.HOST}{C.PATH_TTS_LIVE}",
                json=C.tts_body(request_id=mine, session_id=mine),
                headers={**C.auth(), "Accept": "text/event-stream"}) as sr:
            hdrs = {k: v for k, v in sr.headers.items()
                    if "request-id" in k.lower() or "session-id" in k.lower()}
            await sr.aread()
    except Exception as exc:  # noqa: BLE001
        sse_err = f"{type(exc).__name__}: {str(exc)[:110]}"
    sse_echoed = any(v == mine for v in hdrs.values())
    record(name="SSE echoes client-supplied request_id",
           claim="SSE reference: the response returns X-Session-Id, "
                 "X-Request-Id, X-External-Session-Id and X-External-Request-Id "
                 "headers",
           expected=f"X-External-Request-Id == {mine!r}",
           observed=(f"headers: {hdrs}" if hdrs
                     else f"failed: {sse_err}" if sse_err
                     else "no id headers returned"),
           verdict="INCONCLUSIVE" if sse_err and not hdrs
                   else ("MATCH" if sse_echoed else "MISMATCH"),
           evidence={"sent": mine, "id_headers": hdrs, "error": sse_err},
           doc_url=DOCS["sse"])
    await asyncio.sleep(C.GAP)

    # The sync endpoint is the open question: the WS reference documents the
    # echo, the synthesize-speech reference does not mention it, and a client
    # that cannot stamp an id has nothing to correlate on a timeout, where
    # there is no response header to read at all.
    r = await client.post(url, json=C.tts_body(request_id=mine), headers=C.auth())
    hits = {k: v for k, v in r.headers.items() if mine in str(v)}
    record(name="REST echoes client-supplied request_id",
           claim="both the WS reference (external_request_id in frames) and the "
                 "SSE reference (X-External-Request-Id header) document the "
                 "echo; the Synthesize Speech reference does not state whether "
                 "the sync endpoint does the same",
           expected="the same echo as WS, or a doc note that it is WS-only",
           observed=f"http {r.status_code}, "
                    + (f"echoed in {hits}" if hits else "not echoed in any header"),
           verdict="MATCH" if hits else "UNDOCUMENTED",
           evidence={"sent": mine, "status": r.status_code,
                     "matching_headers": hits, "all_headers": dict(r.headers)},
           doc_url=DOCS["synth"])
    await asyncio.sleep(C.GAP)

    for name, host in C.HOSTS.items():
        try:
            rr = await client.post(f"https://{host}{C.PATH_TTS}",
                                   json=C.tts_body(), headers=C.auth(), timeout=45)
            record(name=f"region host: {name}",
                   claim="limits page: pin traffic with api.india.smallest.ai "
                         "or api.us.smallest.ai",
                   expected="200",
                   observed=f"{rr.status_code}, {len(rr.content)} bytes",
                   verdict="MATCH" if rr.status_code == 200 else "MISMATCH",
                   evidence={"host": host, "status": rr.status_code,
                             "x_request_id": rr.headers.get("x-request-id")},
                   doc_url=DOCS["limits"])
        except Exception as exc:  # noqa: BLE001
            record(name=f"region host: {name}", claim="limits page: region pinning",
                   expected="200", observed=f"{type(exc).__name__}: {str(exc)[:90]}",
                   verdict="MISMATCH", evidence={"host": host}, doc_url=DOCS["limits"])
        await asyncio.sleep(C.GAP)


# --------------------------------------------------------------------------


async def main() -> None:
    C.require_key()
    C.say("=" * 70)
    C.say(f"DOCS VS REALITY  host={C.HOST}")

    failed: list[str] = []

    async def run(name: str, section: Any) -> Any:
        """One section, isolated.

        A section that raises must not take the remaining sections with it.
        The run exists to collect evidence, and the first section dying is no
        reason to lose the other eight - a client-side error in particular is
        a harness fault and says nothing about the product.
        """
        try:
            return await section
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001
            failed.append(name)
            C.say(f"\n[{name}] section FAILED; continuing with the next one")
            for line in traceback.format_exc().splitlines():
                C.say(f"  {line}")
            return None

    async with httpx.AsyncClient(timeout=60.0) as client:
        # A balance marker at each boundary, so a surprising charge is
        # attributable to a section instead of to the run as a whole.
        C.balance_checkpoint("start of probes")
        await run("auth", auth_header(client))
        C.balance_checkpoint("after auth")
        token = await run("tokens", tokens(client))
        await run("token expiry", token_expiry(client))
        C.balance_checkpoint("after tokens")
        await run("limits", limits(client))
        C.balance_checkpoint("after limits")
        await run("params", params(client))
        C.balance_checkpoint("after params - the over-limit text case bills if "
                             "it returned 200")
        await run("deprecation", deprecation(client))
        await run("error consistency", error_consistency(client))
        await run("protocol", protocol(client, token))
        await run("meta", meta(client))
        C.balance_checkpoint("end of probes")

    if failed:
        C.say(f"\n{len(failed)} section(s) failed: {', '.join(failed)}")
        C.say("their probes are absent from the counts below, so a low "
              "MISMATCH count here is not evidence of correctness")

    counts: dict[str, int] = {}
    for p in RESULTS:
        counts[p.verdict] = counts.get(p.verdict, 0) + 1
    C.say("\n" + "=" * 70)
    C.say("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    mismatches = [p for p in RESULTS if p.verdict == "MISMATCH"]
    if mismatches:
        C.say(f"\nMismatches ({len(mismatches)}) - candidate findings")
        for p in mismatches:
            C.say(f"\n  {p.name}")
            C.say(f"    docs say: {p.claim}")
            C.say(f"    expected: {p.expected}")
            C.say(f"    observed: {p.observed}")
            C.say(f"    source:   {p.doc_url}")

    undocumented = [p for p in RESULTS if p.verdict == "UNDOCUMENTED"]
    if undocumented:
        C.say(f"\nUndocumented behaviour ({len(undocumented)})")
        for p in undocumented:
            C.say(f"  - {p.name}: {p.observed}")

    C.save_json("probe_results.json",
                {"environment": C.environment(), "host": C.HOST,
                 "results": RESULTS})
    C.say("\nraw evidence -> probe_results.json")
    # stdout only, never C.say: results.txt is the file that gets attached to
    # the report, and a line reading "2 access-control findings" is itself the
    # disclosure. Whoever runs this sees it on screen; the log does not.
    if SECURITY:
        C.save_json("security_private.json",
                    {"methodology": SECURITY_METHOD, "findings": SECURITY})
        print(f"\n[private, not logged] {len(SECURITY)} access-control "
              "finding(s) written to security_private.json - private "
              "disclosure to Smallest AI only; keep out of the report and off "
              "any call")
    else:
        print("\n[private, not logged] no access-control findings; "
              "security_private.json not written")


if __name__ == "__main__":
    asyncio.run(main())