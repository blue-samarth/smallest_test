from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any

import common as C

VERDICTS: tuple[str, ...] = ("MISMATCH", "UNDOCUMENTED", "INCONCLUSIVE", "MATCH")

STATUS: dict[str, str] = {"MISMATCH": "critical", "UNDOCUMENTED": "warning",
                          "INCONCLUSIVE": "serious", "MATCH": "good"}

MARK: dict[str, str] = {"MISMATCH": "✕", "UNDOCUMENTED": "?",
                        "INCONCLUSIVE": "–", "MATCH": "✓"}

CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --plane: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --bar: #2a78d6;
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface: #1a1a19;
    --plane: #0d0d0d;
    --ink: #f0efec;
    --ink-2: #c3c2b7;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --bar: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 64px;
  background: var(--plane);
  color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 12px; }
.sub { color: var(--ink-2); margin: 0 0 28px; font-size: 13px; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
}
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.tile {
  flex: 1 1 150px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--tint, var(--axis));
  border-radius: 10px;
  padding: 14px 16px;
}
.tile .v { font-size: 28px; font-weight: 600; }
.tile .k { font-size: 12px; color: var(--ink-2); letter-spacing: 0.03em; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  font-weight: 600;
  padding: 0 10px 8px 0;
  border-bottom: 1px solid var(--axis);
}
td { padding: 9px 10px 9px 0; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  padding: 2px 8px;
  border-radius: 999px;
  color: #fff;
  background: var(--tint);
}
.badge.warning { color: #0b0b0b; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 2px; }
.bar-label { flex: 0 0 210px; font-size: 13px; color: var(--ink-2); }
.bar-track { flex: 1 1 auto; background: var(--grid); border-radius: 0 4px 4px 0; }
.bar-fill { height: 18px; background: var(--bar); border-radius: 0 4px 4px 0; }
.bar-value {
  flex: 0 0 80px;
  text-align: right;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
.note { color: var(--ink-2); font-size: 12px; margin: 14px 0 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.claim { color: var(--ink-2); font-size: 12px; margin-top: 3px; }
a { color: var(--bar); }
@media (max-width: 640px) {
  .bar-label { flex-basis: 110px; }
  .tile { flex-basis: 100%; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load(name: str) -> dict[str, Any]:
    """Read one evidence file, or an empty dict if that section never ran.

    The JSON files are the scrubbed copies written by C.save_json, so building
    the page from them rather than from live objects means the report inherits
    the same secret scrubbing instead of needing its own.
    """
    if not os.path.exists(name):
        return {}
    try:
        with open(name) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tiles(probes: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {v: 0 for v in VERDICTS}
    for p in probes:
        verdict = p.get("verdict", "")
        if verdict in counts:
            counts[verdict] += 1
    cells = [
        f'<div class="tile" style="--tint: var(--{STATUS[v]})">'
        f'<div class="v">{counts[v]}</div>'
        f'<div class="k">{MARK[v]} {v}</div></div>'
        for v in VERDICTS
    ]
    return f'<div class="tiles">{"".join(cells)}</div>'


def _bars(summary: dict[str, Any]) -> str:
    """Median time to first audio per transport.

    One measure across a handful of long-labelled categories, so horizontal
    bars in a single hue: there is no second series for a colour to identify,
    and every bar carries its own value label.
    """
    rows = [(k, v) for k, v in summary.items() if v.get("n")]
    if not rows:
        return ""
    peak: float = max(float(v["p50"]) for _, v in rows)
    out: list[str] = []
    for label, v in rows:
        width = max(float(v["p50"]) / peak * 100.0, 1.0) if peak else 1.0
        tip = (f"{label}: p50 {v['p50']} ms, min {v['min']} ms, "
               f"max {v['max']} ms, stdev {v['stdev']} ms, n={v['n']}")
        out.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{_esc(label)}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width: {width:.1f}%" '
            f'title="{_esc(tip)}"></div></div>'
            f'<div class="bar-value">{float(v["p50"]):.0f} ms</div>'
            f'</div>')
    return "".join(out)


def _latency(summary: dict[str, Any]) -> str:
    head = ("<tr><th>Transport</th><th class='num'>n</th><th class='num'>err</th>"
            "<th class='num'>429</th><th class='num'>min</th><th class='num'>p50</th>"
            "<th class='num'>max</th><th class='num'>stdev</th>"
            "<th class='num'>p95*</th></tr>")
    body: list[str] = []
    for label, v in summary.items():
        if not v.get("n"):
            body.append(f"<tr><td>{_esc(label)}</td><td class='num'>0</td>"
                        f"<td class='num'>{v.get('errors', 0)}</td>"
                        f"<td class='num'>{v.get('rate_limited', 0)}</td>"
                        f"<td colspan='5'>no successful samples</td></tr>")
            continue
        cells = "".join(f"<td class='num'>{v[k]}</td>" for k in
                        ("n", "errors", "rate_limited", "min", "p50", "max",
                         "stdev", "p95_indicative"))
        body.append(f"<tr><td>{_esc(label)}</td>{cells}</tr>")
    return (f"<table>{head}{''.join(body)}</table>"
            "<p class='note'>Milliseconds, time to first audio. 429s are excluded "
            "from the distribution and counted separately. p95* interpolates "
            "between the two slowest samples at this n, so it is indicative "
            "rather than a tail estimate.</p>")


def _probes(probes: list[dict[str, Any]]) -> str:
    order = {v: i for i, v in enumerate(VERDICTS)}
    ranked = sorted(probes, key=lambda p: order.get(p.get("verdict", ""), 99))
    head = ("<tr><th>Verdict</th><th>Probe</th><th>Expected</th>"
            "<th>Observed</th></tr>")
    body: list[str] = []
    for p in ranked:
        verdict = p.get("verdict", "")
        tint = STATUS.get(verdict, "")
        badge = (f'<span class="badge {tint}" style="--tint: var(--{tint})">'
                 f'{MARK.get(verdict, "")} {_esc(verdict)}</span>'
                 if tint else _esc(verdict))
        doc = p.get("doc_url")
        link = (f' &middot; <a href="{_esc(doc)}">docs</a>' if doc else "")
        body.append(
            f"<tr><td>{badge}</td>"
            f"<td><strong>{_esc(p.get('name'))}</strong>"
            f"<div class='claim'>{_esc(p.get('claim'))}{link}</div></td>"
            f"<td>{_esc(p.get('expected'))}</td>"
            f"<td class='mono'>{_esc(p.get('observed'))}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _boundary(data: dict[str, Any]) -> str:
    head = ("<tr><th>Case</th><th class='num'>sent</th><th class='num'>trimmed</th>"
            "<th class='num'>http</th><th class='num'>audio</th>"
            "<th>Verdict</th></tr>")
    body: list[str] = []
    for c in data.get("cases", []):
        verdict = str(c.get("verdict", ""))
        tint = "critical" if "CONTRADICTS" in verdict else "good"
        billed = (" <span class='badge' style=\"--tint: var(--critical)\">BILLED"
                  "</span>" if c.get("possibly_billed") else "")
        body.append(
            f"<tr><td>{_esc(c.get('label'))}</td>"
            f"<td class='num'>{c.get('chars_sent')}</td>"
            f"<td class='num'>{c.get('chars_after_trim')}</td>"
            f"<td class='num'>{c.get('status')}</td>"
            f"<td class='num'>{c.get('audio_bytes')}</td>"
            f'<td><span class="badge" style="--tint: var(--{tint})">'
            f"{_esc(verdict)}</span>{billed}</td></tr>")
    return (f"<table>{head}{''.join(body)}</table>"
            f"<p class='note'>Documented maximum "
            f"{_esc(data.get('documented_max'))} characters after trimming.</p>")


def _failures(samples: list[dict[str, Any]]) -> str:
    bad = [s for s in samples if not s.get("ok")]
    if not bad:
        return ""
    head = ("<tr><th>Transport</th><th class='num'>run</th><th class='num'>http</th>"
            "<th>Detail</th></tr>")
    body = "".join(
        f"<tr><td>{_esc(s.get('transport'))}</td>"
        f"<td class='num'>{s.get('run')}</td>"
        f"<td class='num'>{s.get('status')}</td>"
        f"<td class='mono'>{_esc((s.get('error') or s.get('note') or '')[:300])}"
        f"</td></tr>"
        for s in bad[:40])
    return (f"<div class='card'><h2>Failed samples ({len(bad)})</h2>"
            f"<table>{head}{body}</table></div>")


def build(path: str = "report.html") -> str | None:
    """Render the run's evidence as one standalone page, or None if the run
    produced nothing to report.

    Written into the run directory beside the JSON it is built from, with no
    external stylesheet, script or font, so the file stays readable offline
    and can be attached to the report as-is.
    """
    bench = _load("bench_results.json")
    probe = _load("probe_results.json")
    bound = _load("boundary_results.json")
    if not (bench or probe or bound):
        return None

    probes: list[dict[str, Any]] = probe.get("results", [])
    summary: dict[str, Any] = bench.get("summary", {})
    config: dict[str, Any] = bench.get("config", {})
    host = bench.get("host") or probe.get("host") or bound.get("host") or C.HOST
    stamp = datetime.now(C.IST).strftime("%Y-%m-%d %H:%M %Z")

    meta = "  ".join(f"{k}={_esc(v)}" for k, v in config.items())
    parts: list[str] = [
        "<h1>Smallest AI TTS - QA evidence</h1>",
        f"<p class='sub'>host {_esc(host)} &middot; generated {_esc(stamp)}"
        + (f"<br>{meta}" if meta else "") + "</p>",
    ]

    if probes:
        parts.append(_tiles(probes))

    if summary:
        parts.append("<div class='card'><h2>Time to first audio</h2>"
                     + _bars(summary) + _latency(summary) + "</div>")

    if probes:
        parts.append("<div class='card'><h2>Documented behaviour vs observed"
                     f" ({len(probes)} probes)</h2>{_probes(probes)}</div>")

    if bound.get("cases"):
        parts.append("<div class='card'><h2>Text length boundary</h2>"
                     + _boundary(bound) + "</div>")

    parts.append(_failures(bench.get("samples", [])))

    page = (
        "<!doctype html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>TTS QA evidence - {_esc(host)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f"<div class='wrap'>{''.join(parts)}</div>\n</body>\n</html>\n"
    )
    with open(path, "w") as fh:
        fh.write(page)
    return path


if __name__ == "__main__":
    out = build()
    C.say(f"html report -> {out}" if out
          else "no evidence files in this directory; nothing to report")
