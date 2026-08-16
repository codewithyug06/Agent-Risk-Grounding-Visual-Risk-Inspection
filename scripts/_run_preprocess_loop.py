"""Self-restarting wrapper for preprocessing.

Runs preprocess_real_data.py in a loop (subprocess) until data/processed/stats.json
exists. The subprocess normally gets killed around the ~14-min mark; this wrapper
restarts it, and the resume-enabled preprocessor continues from its checkpoint.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "stats.json"
CMD = [
    sys.executable, "scripts/preprocess_real_data.py",
    "--config", "configs/data.yaml",
    "--resolution", "224", "224",
    "--train-ratio", "0.70", "--val-ratio", "0.15", "--seed", "42",
]
LOG = ROOT / "data" / "processed" / "_run_loop.log"

attempt = 0
with open(LOG, "a") as logf:
    while not OUT.exists():
        attempt += 1
        msg = f"\n=== attempt {attempt} @ {time.strftime('%H:%M:%S')} ===\n"
        logf.write(msg); logf.flush()
        print(msg, end="")
        proc = subprocess.run(CMD, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
        if OUT.exists():
            break
        logf.write(f"attempt {attempt} exited code {proc.returncode}\n"); logf.flush()
        print(f"attempt {attempt} exited code {proc.returncode}")
        time.sleep(2)

done = f"\n=== DONE: stats.json exists @ {time.strftime('%H:%M:%S')} ===\n"
with open(LOG, "a") as logf:
    logf.write(done)
print(done, end="")
