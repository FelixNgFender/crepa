# ruff: noqa: T201
import pathlib
import re

from crepa import settings

ERR_RE = re.compile(r"\*\s+acc@1\s+[0-9.]+\s+acc@5\s+[0-9.]+\s+err@1\s+([0-9.]+)")
DIST_RE = re.compile(r"distortion:\s+([a-z_]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse(args: settings.Parse) -> None:
    rows = _parse_logs(args.input)

    match args.mode:
        case "table":
            print("distortion\tavg_err1_percent\tformula")
            for distortion, avg, _, formula in rows:
                print(f"{distortion}\t{avg:.3f}\t{formula}")
        case "sheet":
            print("\t".join(distortion for distortion, _, _, _ in rows))
            print("\t".join(formula for _, _, _, formula in rows))
        case "formulas":
            print("\t".join(formula for _, _, _, formula in rows))
        case "long":  # long
            print("distortion\tformula")
            for distortion, _, _, formula in rows:
                print(f"{distortion}\t{formula}")


def _parse_logs(inp: pathlib.Path) -> list[tuple[str, float, list[str], str]]:
    text = ANSI_RE.sub("", inp.read_text(errors="ignore"))

    err_tokens = ERR_RE.findall(text)
    distortions = DIST_RE.findall(text)

    if not distortions:
        msg = "no distortion blocks found in log."
        raise ValueError(msg)

    needed = len(distortions) * 5
    if len(err_tokens) < needed:
        msg_0 = f"not enough err@1 entries: found {len(err_tokens)}, need at least {needed}."
        raise ValueError(msg_0)

    # if extra runs exist (e.g., clean), keep only the last distortion-related runs.
    start = len(err_tokens) - needed

    rows = []
    for i, distortion in enumerate(distortions):
        vals = err_tokens[start + i * 5 : start + (i + 1) * 5]
        avg = sum(float(v) for v in vals) / 5.0
        formula = f"=AVERAGE({','.join(vals)})/100"
        rows.append((distortion, avg, vals, formula))
    return rows
