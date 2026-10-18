"""House style shared by every scripts/make_fig_*.py: palette, entity map (SERIES), type sizes, save().

Canvases are drawn at the ICML widths (\\textwidth = 6.75 in, \\columnwidth = 3.25 in), so
\\includegraphics at that width is 1:1 and type lands at its set size.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analysis"
MAT = AN / "matrices"
OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TEXTWIDTH, COLUMNWIDTH = 6.75, 3.25

# type sizes (pt); nothing below 8 pt on the page
BODY, TICK, SMALL = 9.0, 8.0, 8.0

# palette: Crameri oslo (cool) = one model sampled many times, vik warm arm = a shared random
# component, ink = theory
OSLO = ["#010101", "#0E1E2E", "#15395B", "#26578C",
        "#507BBC", "#7B98CA", "#A2B0CA", "#CED1D7", "#FFFFFF"]
COOL, COOL_MD = OSLO[3], OSLO[4]
WARM, WARM_DK = "#B75A26", "#852206"
INK, MUTED, REFLAB, FAINT = "#1A1A1A", "#5B5B5B", "#444444", "#888888"
WHISK, GRID, HAIR = "#3A3A3A", "#DCDCDC", "#D9D9D9"


def tint(hex_color: str, f: float) -> str:
    """Blend `hex_color` a fraction `f` of the way toward white."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    m = lambda c: int(round(c + (255 - c) * f))  # noqa: E731
    return f"#{m(r):02X}{m(g):02X}{m(b):02X}"


# entity map, key -> (label, color, linestyle, marker); figures import it, never redefine it
SERIES = {
    "iid":          ("independent draws", COOL,    "-",  "o"),
    "sharedseed":   ("shared seed",       WARM,    "-",  "s"),
    "sharedprompt": ("shared prompt",     WARM_DK, "-",  "^"),
    "ensemble":     ("24-model ensemble", INK,     "-",  "D"),
    "mp_edge":      (r"MP edge $\lambda_+$", INK,  "--", "x"),
    "null_2pl":     ("2PL null p99",      COOL_MD, "--", "_"),
    "null_bern":    ("Bernoulli null p99", FAINT,  ":",  "_"),
    # letter-aware know-or-guess null fitted on rotation 0
    "null_letter":  ("letter-aware null p99", "#8172B3", "-.", "_"),
    # 3PL with the 4-option guessing floor c = 0.25
    "null_3pl":     ("3PL null p99",      "#55A868", ":",  "_"),
    # Rasch (JML) on the all-4-correct matrix
    "null_rasch":   ("Rasch null p99",    "#7F7F7F", (0, (5.0, 1.5, 1.0, 1.5, 1.0, 1.5)), "_"),
}


def _usetex_available() -> bool:          # cm-super needed for Type-1 embedding
    if shutil.which("pdflatex") is None:
        return False
    try:
        r = subprocess.run(["kpsewhich", "type1ec.sty"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


USETEX = _usetex_available() and os.environ.get("FIG_NO_USETEX") is None
RC = {
    "text.usetex": USETEX,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": BODY, "axes.labelsize": BODY, "axes.titlesize": BODY,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": SMALL,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.2, "patch.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}
if USETEX:
    RC["text.latex.preamble"] = r"\usepackage{times}\usepackage{amsmath}"


def dash() -> str:
    """En dash that renders in both usetex and mathtext builds."""
    return "--" if USETEX else "–"


def bf(s: str) -> str:
    return rf"\textbf{{{s}}}" if USETEX else s


def panel_tag(ax, tag: str, x: float = -0.02, y: float = 1.0) -> None:
    """Bold (a)/(b)/(c) at the top-left corner of the axes box; no titles (the caption names panels)."""
    ax.text(x, y, bf(f"({tag})"), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=BODY, color=INK)


def ygrid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def ref_label(ax, y: float, text: str, x_axes: float = 0.995, above: bool = True,
              color: str = REFLAB) -> None:
    """Label a full-width reference line at its right end, just above (or below) it."""
    ax.annotate(text, xy=(x_axes, y), xycoords=("axes fraction", "data"),
                xytext=(0, 1.5 if above else -1.5), textcoords="offset points",
                ha="right", va="bottom" if above else "top", fontsize=SMALL, color=color)


def load(name: str) -> dict:
    return json.loads((AN / name).read_text())


def save(fig, name: str) -> Path:
    pdf = OUTDIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(OUTDIR / name)],
                       check=False)
    print(f"[saved] {pdf}  usetex={USETEX}")
    return pdf
