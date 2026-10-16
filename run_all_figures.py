"""Run every figure cell from figures.ipynb and save PDFs/PNGs."""
import json
import sys
import os

# Force non-interactive backend before any matplotlib import
os.environ["MPLBACKEND"] = "Agg"

NOTEBOOK = "figures.ipynb"

with open(NOTEBOOK) as f:
    nb = json.load(f)

cells = nb["cells"]

# Build one combined source, prepending the Agg backend fix and
# neutralising plt.show() so it doesn't block.
preamble = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt_orig
import matplotlib.figure as _mpl_fig

_plt_orig.show = lambda *a, **kw: None

# Auto-save PNG alongside every PDF save
_orig_savefig = _mpl_fig.Figure.savefig
def _savefig_also_png(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, str) and fname.lower().endswith(".pdf"):
        png_path = fname[:-4] + ".png"
        kw2 = dict(kwargs)
        kw2.setdefault("dpi", 200)
        _orig_savefig(self, png_path, *args, **kw2)
        print(f"  → also saved {png_path}")
_mpl_fig.Figure.savefig = _savefig_also_png
"""

combined = [preamble]
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not src.strip():
        continue
    combined.append(f"\n# {'='*70}\n")
    combined.append(src)
    combined.append("\n")

full_script = "\n".join(combined)

# Write to a temp file so tracebacks show line numbers
tmp = "_figures_runner_tmp.py"
with open(tmp, "w") as f:
    f.write(full_script)

print("Running figures notebook …")
exec(compile(full_script, NOTEBOOK, "exec"), {"__name__": "__main__"})
print("\nAll done.")
os.remove(tmp)
