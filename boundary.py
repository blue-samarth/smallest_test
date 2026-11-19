from __future__ import annotations

import asyncio
import sys

import common as C

try:
    import httpx
except ImportError:
    sys.exit('pip install httpx "websockets>=14"')


def filler(n: int) -> str:
    """
    EXACTLY n characters of pronounceable text, with no trailing whitespace.
    """
    unit: str = "The quick brown fox jumps over the lazy dog. "
    out: str = (unit * (n // len(unit) + 1))[:n]
    if out and out[-1].isspace():
        out = out[:-1] + "."
    assert len(out) == n, f"filler produced {len(out)} chars, wanted {n}"
    assert not out[-1:].isspace(), "filler left trailing whitespace"
    return out


async def case(client: httpx.AsyncClient, label: str, text: str, expect: str) -> dict:
    sent = len(text)
    trimmed = len(text.strip())
    r: httpx.Response = await client.post(f"https://{C.HOST}{C.PATH_TTS}", json=C.tts_body(text), headers=C.auth(), timeout=120)
    audio = len(r.content) if r.status_code == 200 else 0
    body = "" if r.status_code == 200 else r.text[:300]
    ok_200 = r.status_code == 200
    if expect == "200":
        verdict = ("as documented" if ok_200
                   else "CONTRADICTS the documented 1-8000 range")
    elif 400 <= r.status_code < 500:
        verdict = "as documented"
    elif ok_200:
        verdict = "CONTRADICTS the documented 1-8000 range"
    else:
        verdict = (f"CONTRADICTS the documented 1-8000 range: http "
                   f"{r.status_code} is a server error, not 4xx validation")
    billed = ok_200 and trimmed > C.TEXT_MAX
    C.say(f"{label:<34} sent={sent} trimmed={trimmed} "
          f"http={r.status_code} audio={audio}B expected={expect} -> {verdict}"
          + ("  BILLED past the documented limit" if billed else ""))
    if body:
        C.say(f"{'':34} body: {body[:200]}")
    await asyncio.sleep(2.0)
    return {"label": label, "chars_sent": sent, "chars_after_trim": trimmed,
            "status": r.status_code, "audio_bytes": audio,
            "expected": expect, "verdict": verdict,
            "possibly_billed": billed, "body": body,
            "x_request_id": r.headers.get("x-request-id")}


async def main() -> None:
    C.require_key()
    C.say("=" * 70)
    C.say(f"TEXT LENGTH BOUNDARY  host={C.HOST}  documented range 1-{C.TEXT_MAX}")
    C.balance_checkpoint("before the boundary pair - note the balance now")

    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        results.append(await case(client, f"{C.TEXT_MAX} chars (at the limit)", filler(C.TEXT_MAX), "200"))
        results.append(await case(client, f"{C.TEXT_MAX + 1} chars (over by one)", filler(C.TEXT_MAX + 1), "4xx"))
        results.append(await case(client, f"{C.TEXT_MAX} + whitespace", "  \n" + filler(C.TEXT_MAX) + "\t ", "200"))
        results.append(await case(client, f"{C.TEXT_MAX + 1} + whitespace", "  \n" + filler(C.TEXT_MAX + 1) + "\t ", "4xx"))

    C.balance_checkpoint("after the boundary pair - compare")
    C.save_json("boundary_results.json",
                {"environment": C.environment(), "host": C.HOST,
                 "documented_max": C.TEXT_MAX, "cases": results})

    contradictions = [x for x in results if "CONTRADICTS" in x["verdict"]]
    billed = [x for x in results if x["possibly_billed"]]
    C.say("")
    if contradictions:
        C.say(f"{len(contradictions)} case(s) contradict the documented range:")
        for x in contradictions:
            C.say(f"  {x['label']}: http {x['status']}, expected {x['expected']}")
    else:
        C.say("the documented 1-8000 range holds exactly at the boundary")
    if billed:
        C.say(f"{len(billed)} case(s) were synthesized past the documented "
              "limit and are therefore billed - check the balance delta")
    C.say("boundary evidence -> boundary_results.json")


if __name__ == "__main__":
    asyncio.run(main())