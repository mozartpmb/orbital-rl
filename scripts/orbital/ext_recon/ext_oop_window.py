"""Out-of-plane (relative-inclination) design laws for ext-3d, evaluated at our env's
actual altitudes, warp granularities, and success boxes.  Analytics only."""
import math, csv
MU=3.986004418e14; RE=6.371e6; BUDGET=478.0
WARPS_S=[60,300,1800,3600,10800,21600]   # env: 60s substep, warp 5min/30min/1h/3h/6h
WARP_NAME=["60 s","5 min","30 min","1 h","3 h","6 h"]
BOXES_KM=[30,10,5]
rows=[]
print("== cross-track scaling: miss = a*|di|,  dv_null = v*|di| = n*a*|di| ==")
print(f"{'alt km':>7} {'T_orb min':>10} {'v m/s':>8} {'n rad/s':>10} {'dv per km CT':>13} {'di for 30km':>12} {'di for 5km':>11}")
for alt in [300,550,800,2000,8000,20200]:
    a=RE+alt*1e3; v=math.sqrt(MU/a); n=math.sqrt(MU/a**3); T=2*math.pi/n
    print(f"{alt:7d} {T/60:10.1f} {v:8.1f} {n:10.3e} {v*1e3/a:13.4f} "
          f"{math.degrees(30e3/a):12.4f} {math.degrees(5e3/a):11.4f}")
    for bk in BOXES_KM:
        di=bk*1e3/a
        rows.append(dict(kind="crosstrack_box",alt_km=alt,box_km=bk,
            di_deg=round(math.degrees(di),5), dv_null_ms=round(v*di,3),
            frac_budget=round(v*di/BUDGET,5)))

print("\n== out-of-plane burn window (single normal impulse at u* = atan2(di_y,di_x)) ==")
print("cosine efficiency = cos(du);  window half-width for a given loss")
for loss in [0.01,0.05,0.10,0.25]:
    du=math.degrees(math.acos(1-loss))
    print(f"  loss<= {loss*100:5.1f}%  ->  |du| <= {du:6.2f} deg")
print()
hdr=f"{'alt km':>7} {'T min':>7} " + " ".join(f"{w:>9}" for w in WARP_NAME)
print(hdr); print("  (row = deg of argument-of-latitude advanced per action)")
for alt in [300,550,800,2000,8000,20200]:
    a=RE+alt*1e3; n=math.sqrt(MU/a**3); T=2*math.pi/n
    cells=[]
    for dt in WARPS_S:
        du=360.0*dt/T
        cells.append(f"{du:9.2f}")
        rows.append(dict(kind="warp_u_advance",alt_km=alt,warp_s=dt,
            du_deg_per_action=round(du,4),
            cos_eff_worst=round(math.cos(math.radians(min(180,du/2))),5)))
    print(f"{alt:7d} {T/60:7.1f} " + " ".join(cells))

print("\n== worst-case cosine efficiency if the agent can only land within +-half a warp ==")
print(f"{'alt km':>7} " + " ".join(f"{w:>9}" for w in WARP_NAME))
for alt in [300,550,800,2000,8000,20200]:
    a=RE+alt*1e3; n=math.sqrt(MU/a**3); T=2*math.pi/n
    cells=[]
    for dt in WARPS_S:
        du=360.0*dt/T
        eff=math.cos(math.radians(min(90.0,du/2)))
        cells.append(f"{eff:9.3f}")
    print(f"{alt:7d} " + " ".join(cells))

print("\n== dv quantum vs cross-track resolution (env burn quanta 1 and 10 m/s normal) ==")
for alt in [300,550,800,2000,8000,20200]:
    a=RE+alt*1e3; v=math.sqrt(MU/a)
    for q in [1.0,10.0,25.0]:
        di=q/v; ct=di*a
        rows.append(dict(kind="burn_quantum",alt_km=alt,dv_quantum_ms=q,
            di_deg=round(math.degrees(di),6), crosstrack_km=round(ct/1e3,3)))
    print(f"  alt {alt:6d} km: 1 m/s normal = {1.0/v*a/1e3:7.3f} km cross-track "
          f"({math.degrees(1.0/v):.5f} deg);  10 m/s = {10.0/v*a/1e3:7.2f} km")

print("\n== max di reachable within budget fraction (small-angle, single impulse) ==")
for frac in [0.10,0.25,0.50,1.00]:
    print(f"  budget frac {frac:4.2f} = {BUDGET*frac:6.1f} m/s:  " +
          "  ".join(f"{alt}km:{math.degrees(BUDGET*frac/math.sqrt(MU/(RE+alt*1e3))):5.3f}deg"
                    for alt in [300,800,8000,20200]))

out="/Users/pete/space_training/web_data/results/ext_3d_oop_windows.csv"
keys=["kind","alt_km","box_km","warp_s","dv_quantum_ms","di_deg","dv_null_ms","frac_budget",
      "du_deg_per_action","cos_eff_worst","crosstrack_km"]
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,"") for k in keys})
print("\nwrote",out,len(rows),"rows")
