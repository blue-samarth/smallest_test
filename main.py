import asyncio
import os
import traceback
from datetime import datetime

import bench
import common as C
import probes
import report


def new_run_dir() -> str:
    """
    A fresh directory per run. Overwriting results.txt and the JSON files
    destroys the only copy of the previous run's evidence — and comparing two
    runs at different hours is explicitly part of the method, so the earlier
    one has to survive.
    """
    stamp: str = datetime.now(C.IST).strftime("%Y-%m-%dT%H-%M%z")
    path: str = os.path.join("runs", stamp)
    suffix: int = 0
    while os.path.exists(path):
        suffix += 1
        path = os.path.join("runs", f"{stamp}-{suffix}")

    os.makedirs(path)
    link: str = os.path.join("runs", "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(path), link)
    except OSError:
        pass
    return path


async def main() -> int:
    C.require_key()
    out: str = new_run_dir()
    os.chdir(out)
    C.say(f"run directory: {out}")

    failed: list[str] = []
    for name, fn in (("bench", bench.main), ("probes", probes.main)):
        try:
            await fn()
        except KeyboardInterrupt:
            C.say(f"\n{name}: interrupted by user; evidence so far is kept")
            raise
        except Exception:  # noqa: BLE001
            failed.append(name)
            C.say(f"\n{name} FAILED; continuing with the next section")
            for line in traceback.format_exc().splitlines():
                C.say(f"  {line}")

    C.say("")
    if failed:
        C.say(f"completed with failures in: {', '.join(failed)}")
        C.say("partial evidence is still written; check the traceback above")
    else:
        C.say("completed with no harness failures")

    page: str | None = None
    try:
        page = report.build()
    except Exception:  # noqa: BLE001
        C.say("report rendering FAILED; the JSON evidence is unaffected")
        for line in traceback.format_exc().splitlines():
            C.say(f"  {line}")

    C.say(f"evidence in {out}/  (results.txt, the JSON files"
          + (f" and {page}" if page else "") + ")")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))