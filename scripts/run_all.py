"""Regenerate every figure, animation, data file, and report.

Figure 2 triggers the twelve regime runs and the free-field reference,
which are cached in outputs/cache and reused by Figure 3 and every
animation; Figure 1 caches its verification runs.  An interrupted run
resumes where it stopped.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_geometry.py", "fig01_verification.py", "fig02_fields.py",
         "fig03_keel.py", "fig04_bubble.py", "fig05_regimes.py",
         "anim01_diptych.py", "anim02_six_seas.py", "anim03_ridgeline.py",
         "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
