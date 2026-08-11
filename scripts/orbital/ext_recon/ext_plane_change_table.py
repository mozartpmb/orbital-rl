"""Classical plane-change / combined-maneuver design numbers for the ext-3d campaign.
Pure analytics; no env import. Writes web_data/results/ext_3d_planechange.csv
"""
import math, csv, os
MU = 3.986004418e14
RE = 6.371e6
BUDGET = 478.0  # m/s, env fuel budget

def vcirc(a): return math.sqrt(MU/a)

def simple_pc(a, di_deg):
    """Pure plane rotation on circular orbit radius a."""
    return 2.0*vcirc(a)*math.sin(math.radians(di_deg)/2.0)

def hohmann(a1, a2):
    at = 0.5*(a1+a2)
    v1 = vcirc(a1); v2 = vcirc(a2)
    vp = math.sqrt(MU*(2.0/a1 - 1.0/at))
    va = math.sqrt(MU*(2.0/a2 - 1.0/at))
    return v1, vp, va, v2

def combined(a1, a2, di_deg, s):
    """Hohmann a1->a2 with fraction s of the plane change at burn1, (1-s) at burn2.
    Law of cosines on each impulse."""
    v1, vp, va, v2 = hohmann(a1, a2)
    d1 = math.radians(di_deg)*s
    d2 = math.radians(di_deg)*(1.0-s)
    dv1 = math.sqrt(v1*v1 + vp*vp - 2*v1*vp*math.cos(d1))
    dv2 = math.sqrt(va*va + v2*v2 - 2*va*v2*math.cos(d2))
    return dv1, dv2, dv1+dv2

def opt_split(a1, a2, di_deg, n=20001):
    best = None
    for k in range(n):
        s = k/(n-1)
        t = combined(a1, a2, di_deg, s)[2]
        if best is None or t < best[1]: best = (s, t)
    return best

def roe_dv_normal(a, di_deg):
    """ROE closed form: one normal impulse at u=atan2(di_y,di_x); dv_N = n*a*|delta_i|."""
    n = math.sqrt(MU/a**3)
    return n*a*math.radians(di_deg)   # = v_circ * di_rad

rows=[]
alts_km = [300, 550, 800, 2000, 8000, 20200, 35786]
dis = [0.01,0.05,0.1,0.25,0.5,1.0,2.0,5.0,10.0,28.5,51.6,90.0]
for alt in alts_km:
    a = RE + alt*1e3
    for di in dis:
        rows.append(dict(case="simple_plane_change", alt_km=alt, a_km=a/1e3,
            v_circ_ms=round(vcirc(a),1), di_deg=di,
            dv_ms=round(simple_pc(a,di),3),
            dv_smallangle_ms=round(roe_dv_normal(a,di),3),
            frac_of_budget=round(simple_pc(a,di)/BUDGET,4)))
# combined maneuver: transfers inside our bands
pairs = [(300,800),(300,2000),(500,800),(400,1000),(300,8000),(300,20200),(185,35786)]
for a1k,a2k in pairs:
    a1=RE+a1k*1e3; a2=RE+a2k*1e3
    for di in [0.1,0.5,1.0,2.0,5.0,10.0,28.5]:
        s,tot = opt_split(a1,a2,di)
        dv_sep = hohmann_dv = None
        v1,vp,va,v2 = hohmann(a1,a2)
        hoh = abs(vp-v1)+abs(v2-va)
        sep = hoh + simple_pc(a2,di)      # Hohmann then plane change at higher orbit
        s0 = combined(a1,a2,di,0.0)[2]    # all plane change at burn2 (classic "combined at apoapsis")
        rows.append(dict(case="combined_hohmann_planechange", alt_km=f"{a1k}->{a2k}",
            a_km=round(a1/1e3,1), v_circ_ms=round(v1,1), di_deg=di,
            dv_ms=round(tot,3), dv_smallangle_ms=round(s0,3),
            frac_of_budget=round(tot/BUDGET,4),
            hohmann_only_ms=round(hoh,3), separate_ms=round(sep,3),
            all_at_apoapsis_ms=round(s0,3), opt_split_s_at_burn1=round(s,5),
            savings_vs_separate_pct=round(100*(1-tot/sep),2)))
os.makedirs("/Users/pete/space_training/web_data/results", exist_ok=True)
out="/Users/pete/space_training/web_data/results/ext_3d_planechange.csv"
keys=["case","alt_km","a_km","v_circ_ms","di_deg","dv_ms","dv_smallangle_ms","frac_of_budget",
      "hohmann_only_ms","separate_ms","all_at_apoapsis_ms","opt_split_s_at_burn1","savings_vs_separate_pct"]
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in keys})
print("wrote",out,len(rows),"rows")
# headline printouts
print("\n-- max affordable inclination change (whole 478 m/s budget, pure plane change) --")
for alt in alts_km:
    a=RE+alt*1e3
    di = 2*math.degrees(math.asin(min(1.0, BUDGET/(2*vcirc(a)))))
    print(f"  alt {alt:6d} km  v={vcirc(a):7.1f} m/s  di_max={di:6.3f} deg")
print("\n-- plane change cost at 550 km --")
a=RE+550e3
for di in [0.01,0.1,0.5,1.0,2.0,5.0]:
    print(f"  di={di:5.2f} deg -> {simple_pc(a,di):8.2f} m/s = {100*simple_pc(a,di)/BUDGET:6.2f}% of budget")
print("\n-- combined 300->800 km, opt split --")
a1=RE+300e3;a2=RE+800e3
v1,vp,va,v2=hohmann(a1,a2); hoh=abs(vp-v1)+abs(v2-va)
print(f"  hohmann only = {hoh:.2f} m/s")
for di in [0.1,0.5,1.0,2.0,5.0]:
    s,tot=opt_split(a1,a2,di)
    sep=hoh+simple_pc(a2,di)
    print(f"  di={di:4.1f}: opt s(burn1)={s:.4f} tot={tot:7.2f}  separate={sep:7.2f}  save={100*(1-tot/sep):5.2f}%")
