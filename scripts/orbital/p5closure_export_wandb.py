"""Extract metric curves from local wandb run dirs into JSON for the web frontend.

Reads pufferlib/wandb/run-*/files/wandb-summary.json and the local SQLite
events, exports a curve-per-metric JSON per run.
"""
import glob, json, os, sys
import wandb
import wandb.apis.reports


def export_run_local(run_dir, out_path):
    """Read local wandb history via wandb's offline reader."""
    # Use wandb's pyarrow-based reader on the .wandb file.
    wandb_files = glob.glob(f"{run_dir}/run-*.wandb")
    if not wandb_files:
        print(f"  no .wandb file in {run_dir}")
        return False
    # Easier path: read wandb-history.jsonl which wandb writes alongside
    history_jsonl = os.path.join(run_dir, "files", "wandb-history.jsonl")
    if not os.path.exists(history_jsonl):
        # Wandb 0.25 stores history in the .wandb file, accessible via wandb.Api offline mode
        # Fallback: try wandb-history-summary.json
        summary = os.path.join(run_dir, "files", "wandb-summary.json")
        if os.path.exists(summary):
            with open(summary) as fh: data = json.load(fh)
            # Just save the summary as a single-point "curve"
            with open(out_path, "w") as fh: json.dump({"summary": data}, fh, indent=2)
            print(f"  wrote summary-only to {out_path}")
            return True
        return False
    rows = []
    with open(history_jsonl) as fh:
        for line in fh:
            try: rows.append(json.loads(line))
            except: pass
    with open(out_path, "w") as fh:
        json.dump({"history": rows}, fh)
    print(f"  wrote {len(rows)} rows to {out_path}")
    return True


def export_run_api(run_id, project, out_path):
    """Use wandb online API."""
    api = wandb.Api()
    run = api.run(f"pufferai/{project}/{run_id}")  # adjust entity
    history = list(run.history(samples=10000))
    summary = dict(run.summary)
    config = dict(run.config)
    out = {
        "run_id": run_id,
        "config": config,
        "summary": summary,
        "history": history,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  API: wrote {len(history)} rows to {out_path}")


def main():
    """Pull from wandb cloud API for the canonical seed-42 retrain."""
    api = wandb.Api()
    runs_by_id = {
        "stage_1_0_seed42": "nz4a121c",
        "stage_4_0_seed42": "qlpmgyko",
    }
    out_dir = "web_data/results/wandb_curves"
    os.makedirs(out_dir, exist_ok=True)
    for label, run_id in runs_by_id.items():
        try:
            run = api.run(f"orbital_phase5e/{run_id}")
        except Exception as e:
            print(f"{label}: API fetch failed: {e}"); continue
        # history() is sampled but fast; scan_history is full but slow.
        history = list(run.history(samples=2000, pandas=False))
        summary = dict(run.summary)
        config = dict(run.config)
        out = {
            "run_id": run_id,
            "name": run.name,
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "summary": {k: v for k, v in summary.items() if not k.startswith("_")},
            "history_n": len(history),
            "history": history,
        }
        out_path = os.path.join(out_dir, f"{label}.json")
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"{label}: {len(history)} rows -> {out_path}")


if __name__ == "__main__":
    main()
