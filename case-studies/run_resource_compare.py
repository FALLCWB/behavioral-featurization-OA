"""Driver: run the resource worker per (method, domain) in isolated subprocesses,
collect the JSON, and save a comparison table."""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "run_resource.py")

results = []
for method in ["ours", "deepsets"]:
    for domain in ["memory", "graphs", "ecology"]:
        p = subprocess.run([sys.executable, WORKER, "--method", method, "--domain", domain],
                           capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
        if not lines:
            print(f"FAILED {method}/{domain}:\n{p.stderr[-800:]}", flush=True)
            continue
        r = json.loads(lines[-1])
        results.append(r)
        print(f"{method:9s} {domain:8s}: acc={r['test_acc']}  train={r['train_s']}s  "
              f"cpu={r['cpu_s']}s  peak_rss={r['peak_rss_mb']}MB  model_rss={r['model_rss_mb']}MB"
              f"{'  params='+str(r['params']) if r['params'] else '  dim='+str(r['repr_dim'])}", flush=True)

json.dump(results, open(os.path.join(HERE, "results_resource.json"), "w"), indent=2)
print("\nsaved results_resource.json", flush=True)
