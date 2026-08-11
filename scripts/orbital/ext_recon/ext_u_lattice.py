"""How precisely can the agent land on a target argument of latitude u* using the
env's existing time quanta?  (60 s substep + warps 5min/30min/1h/3h/6h)
Greedy BFS over reachable u-residues mod 360 within a decision budget."""
import math
MU=3.986004418e14; RE=6.371e6
QUANTA=[("coast 60s",60),("warp 5min",300),("warp 30min",1800),
        ("warp 1h",3600),("warp 3h",10800),("warp 6h",21600)]
def analyse(alt_km, budget_decisions=12, sets=None):
    a=RE+alt_km*1e3; T=2*math.pi*math.sqrt(a**3/MU)
    if sets is None: sets=QUANTA
    steps=[(nm, 360.0*dt/T % 360.0, dt) for nm,dt in sets]
    # BFS on residues
    from heapq import heappush, heappop
    best={}  # rounded residue -> (decisions, seconds)
    start=(0.0,0,0)
    frontier=[(0,0,0.0)]  # (decisions, seconds, residue)
    seen=set()
    res_list=[]
    import itertools
    # enumerate all combos up to budget (small alphabet)
    cur={0.0:(0,0)}
    allres={0.0:(0,0)}
    for d in range(budget_decisions):
        nxt={}
        for r,(dd,ss) in cur.items():
            for nm,du,dt in steps:
                r2=round((r+du)%360.0,4)
                if r2 not in allres or (dd+1,ss+dt)<allres[r2]:
                    allres[r2]=(dd+1,ss+dt); nxt[r2]=(dd+1,ss+dt)
        cur=nxt
        if len(allres)>200000: break
    rs=sorted(allres)
    gaps=[(rs[(i+1)%len(rs)]-rs[i])%360.0 for i in range(len(rs))]
    return T, len(rs), max(gaps)
print("Reachable argument-of-latitude residues (mod 360 deg) within N decisions")
print("worst-case gap = largest angular hole -> worst |du| to u* is half the gap\n")
for budget in [3,6,12]:
    print(f"--- decision budget {budget} ---")
    print(f"{'alt km':>7} {'T min':>7} {'FULL set':>28} {'no-60s (warps only)':>30}")
    for alt in [300,550,800,2000,8000,20200]:
        T,n1,g1=analyse(alt,budget)
        T,n2,g2=analyse(alt,budget,sets=QUANTA[1:])
        print(f"{alt:7d} {T/60:7.1f} {n1:9d} pts, gap {g1:7.2f}deg  {n2:9d} pts, gap {g2:7.2f}deg")
    print()
print("cosine-loss thresholds: 1%% -> |du|<=8.11deg, 5%% -> 18.19deg, 10%% -> 25.84deg")
