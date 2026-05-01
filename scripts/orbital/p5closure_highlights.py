"""Pick highlight episodes per e_max:
  - most fuel-efficient success
  - longest patient success (most warp steps before final approach)
  - failure with most progress (Φ_orbit went lowest before terminating)
  - "cinematic" — large initial phase gap, large e, clean rendezvous

Saves chosen episodes to web_data/runs/highlights/ as JSON.
"""
import glob, json, math, os, shutil, sys
import numpy as np
import subprocess

MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0
sys.path.insert(0, os.path.dirname(__file__))


def hohmann(r1, r2):
    a_t = 0.5 * (r1 + r2)
    v1c = math.sqrt(MU/r1); v2c = math.sqrt(MU/r2)
    v1t = math.sqrt(MU * (2.0/r1 - 1.0/a_t)); v2t = math.sqrt(MU * (2.0/r2 - 1.0/a_t))
    return abs(v1t - v1c) + abs(v2c - v2t)


def phi_orbit(d, step):
    sa = float(d["sat_a"][step]); se = float(d["sat_e"][step])
    sw = float(d["sat_omega"][step])
    ta = float(d["target_a"][step]); te = float(d["target_e"][step])
    tw = float(d["target_omega"][step])
    da = abs(sa - ta) / SUCCESS_TOL_A
    e_sx = se*math.cos(sw); e_sy = se*math.sin(sw)
    e_tx = te*math.cos(tw); e_ty = te*math.sin(tw)
    de = math.sqrt((e_sx-e_tx)**2 + (e_sy-e_ty)**2)
    return da + de


def score_episode(d):
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    if mask.sum() < 1: return None
    active = np.where(mask)[0]
    er = float(d["episode_reward"][0])
    success = bool(er > 0)
    n = len(active)
    total_dv = float(np.abs(d["delta_v"][active]).sum())
    sat_a0 = float(d["sat_a"][0]); tgt_a0 = float(d["target_a"][0])
    h = hohmann(sat_a0, tgt_a0)
    ratio = total_dv / max(h, 1e-3)
    sat_th0 = float(d["sat_theta"][0])
    tgt_x0 = float(d["target_x"][0]); tgt_y0 = float(d["target_y"][0])
    tgt_om0 = float(d["target_omega"][0])
    tgt_th0 = (math.atan2(tgt_y0, tgt_x0) - tgt_om0) % (2*math.pi)
    phase_gap = abs(math.atan2(math.sin(sat_th0 - tgt_th0), math.cos(sat_th0 - tgt_th0)))
    init_e_sat = float(d["sat_e"][0]); init_e_tgt = float(d["target_e"][0])
    n_warp = int((d["action"][active] == 9).sum())
    # min Φ_orbit during episode
    phi_samples = [phi_orbit(d, int(active[i])) for i in range(0, n, max(1, n//50))]
    min_phi = min(phi_samples) if phi_samples else float("nan")
    return {
        "success": success, "n": n, "dv": total_dv, "ratio": ratio,
        "phase_gap": phase_gap, "e_sat": init_e_sat, "e_tgt": init_e_tgt,
        "n_warp": n_warp, "warp_pct": n_warp / n,
        "min_phi": min_phi, "hohmann": h,
    }


def pick_highlights(npz_dir, label, ckpt):
    files = sorted(glob.glob(f"{npz_dir}/ep_*.npz"))
    scored = []
    for f in files:
        d = np.load(f)
        s = score_episode(d)
        if s is not None:
            scored.append((f, s))
    if not scored: return []
    successes = [(f, s) for f, s in scored if s["success"]]
    failures = [(f, s) for f, s in scored if not s["success"]]
    picks = []
    if successes:
        # most fuel-efficient (closest to Hohmann)
        most_efficient = min(successes, key=lambda x: x[1]["ratio"])
        picks.append(("efficient", most_efficient))
        # most patient (highest warp_pct among decent-Δv successes)
        patient = max(successes, key=lambda x: (x[1]["warp_pct"], -x[1]["ratio"]))
        if patient[0] != most_efficient[0]: picks.append(("patient", patient))
        # cinematic — large phase gap (>120°) + large e (>0.4 if available)
        cinematic_candidates = [(f, s) for f, s in successes
                                if s["phase_gap"] > 2.1 and (s["e_sat"] + s["e_tgt"]) > 0.3]
        if cinematic_candidates:
            cinematic = min(cinematic_candidates, key=lambda x: x[1]["ratio"])
            if cinematic[0] not in [p[1][0] for p in picks]: picks.append(("cinematic", cinematic))
    if failures:
        # near-miss: lowest min_phi among failures
        near_miss = min(failures, key=lambda x: x[1]["min_phi"])
        picks.append(("near_miss_failure", near_miss))
    return picks


def main():
    out = {}
    seed_42_dirs = {
        "e_0.05": ("/tmp/p5closure_escan_puffer_orbital_177765503091_e0.05", 0.05),
        "e_0.20": ("/tmp/p5closure_escan_puffer_orbital_177765503091_e0.20", 0.20),
        "e_0.50": ("/tmp/p5closure_escan_puffer_orbital_177765503091_e0.50", 0.50),
        "e_0.70": ("/tmp/p5closure_escan_puffer_orbital_177765503091_e0.70", 0.70),
    }
    ckpt = "pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt"

    out_dir = "web_data/runs/highlights"
    os.makedirs(out_dir, exist_ok=True)

    for label, (src, e) in seed_42_dirs.items():
        if not os.path.isdir(src): continue
        picks = pick_highlights(src, label, ckpt)
        out[label] = []
        for tag, (npz_path, score) in picks:
            ep_id = os.path.basename(npz_path).replace("ep_", "").replace(".npz", "")
            highlight_label = f"{label}_{tag}_ep{ep_id}"
            json_in = npz_path.replace("/p5closure_escan", "_NOTUSED").replace(".npz", ".json")
            # Re-export this single episode via export_web_data
            # Easier: just call export_episode directly
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from export_web_data import export_episode
            env_config = {"e_max_target": e, "e_max_sat": e,
                          "same_orbit_init": 0, "valid_init_only": 1}
            ep_data = export_episode(npz_path, env_config,
                                     "phase_5e_stage_4_0", ckpt, downsample=1)
            ep_data["highlight"] = {"category": tag, "score": score}
            out_path = os.path.join(out_dir, f"{highlight_label}.json")
            with open(out_path, "w") as fh:
                json.dump(ep_data, fh, separators=(",", ":"))
            out[label].append({"category": tag, "file": out_path, "summary": {
                "success": score["success"], "n_steps": score["n"],
                "total_dv": round(score["dv"], 1),
                "hohmann_dv": round(score["hohmann"], 1),
                "ratio_to_hohmann": round(score["ratio"], 2),
                "init_phase_gap_deg": round(math.degrees(score["phase_gap"]), 1),
                "init_e_sat": round(score["e_sat"], 3),
                "init_e_target": round(score["e_tgt"], 3),
                "warp_pct": round(score["warp_pct"], 3),
                "min_phi_orbit": round(score["min_phi"], 3),
            }})

    with open("web_data/results/highlights_index.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {sum(len(v) for v in out.values())} highlights across {len(out)} buckets")
    for label, picks in out.items():
        print(f"\n  {label}:")
        for p in picks:
            print(f"    [{p['category']}] dv={p['summary']['total_dv']:.0f}m/s "
                  f"({p['summary']['ratio_to_hohmann']}× Hohmann), "
                  f"phase={p['summary']['init_phase_gap_deg']:.0f}°, "
                  f"e=({p['summary']['init_e_sat']:.2f},{p['summary']['init_e_target']:.2f})")


if __name__ == "__main__":
    main()
