import sys, math, numpy as np
WT='/Users/pete/space_training-j2'; sys.path.insert(0, WT+'/pufferlib')
from pufferlib.ocean.orbital.orbital import Orbital
MU=3.986004418e14; VE=300.0*9.80665; DRY=850.0
def dv_budget(f): return -VE*math.log(1.0-f)
BASE=dict(num_debris_min=0,num_debris_max=0,same_orbit_init=0,init_phase_gap_max=3.14159,
 valid_init_only=1,gave_up_action='terminate',max_valid_init_attempts=4096,
 rendezvous_radius_m=30000.0,rel_vel_tol_ms=50.0,shaping_mode=2,shape_w_lambda=1.0,
 shape_w_match=0.8166667,shape_dv_ref_ms=700.0,shape_gamma=1.0,phase_gap_mode=1,
 phase_obs_mode=1,cap_terminal_reward=0.0,dim3_mode=1,legacy_action_space=30,
 obs_alt_scale_m=8e6,lvlh_scale_m=1.5e7,episode_cap_steps=3000)
J2=dict(j2_mode=1,i_target_min_rad=math.radians(30),i_target_max_rad=math.radians(60),
        raan_target_sample=0,lvlh_frame_mode=1)
CELLS={
 'X3/E0  e.05 LEO':      dict(e_max_target=0.05,e_max_sat=0.05,a_min_override=6.671e6,a_max_override=7.171e6,di_max_rad=0.017453),
 'E1     e.10':          dict(e_max_target=0.10,de_max=0.05,da_max_m=300e3,a_min_override=6.671e6,a_max_override=7.871e6,di_max_rad=0.017453),
 'E2     e.20':          dict(e_max_target=0.20,de_max=0.065,da_max_m=450e3,a_min_override=6.671e6,a_max_override=9.871e6,di_max_rad=0.013090),
 'E3     e.30':          dict(e_max_target=0.30,de_max=0.08,da_max_m=600e3,a_min_override=6.671e6,a_max_override=14.371e6,di_max_rad=0.013090),
 'J2X    e.05 incl':     dict(e_max_target=0.05,e_max_sat=0.05,a_min_override=6.671e6,a_max_override=7.171e6,di_max_rad=0.017453,**J2),
 'W1     node-dom 2-5':  dict(e_max_target=0.05,e_max_sat=0.05,a_min_override=6.671e6,a_max_override=7.171e6,di_max_rad=math.radians(5.0),di_min_rad=math.radians(2.0),di_phase_mode=1,**J2),
}
print("  required Dv = 0.5*v_c*sqrt(da_rel^2+|de|^2) + v_c*|dh|   (the obs[28] estimate)")
print(f"  {'cell':22s} {'dv_req p50':>10s} {'p90':>8s} {'p99':>8s} | infeasible at budget:")
print(f"  {'':22s} {'':10s} {'':8s} {'':8s} | {'245':>6s} {'350':>6s} {'478':>6s} {'656':>6s}")
for name,kw in CELLS.items():
    env=Orbital(num_envs=256, **dict(BASE,**kw))
    req=[]
    for k in range(8):
        env.reset(seed=7000+k); st=env.get_state()
        a_s,a_t=st[:,0],st[:,15]
        hs,ht=st[:,5:8],st[:,20:23]; es,et=st[:,30:33],st[:,33:36]
        v_c=np.sqrt(MU/a_t)
        da_rel=(a_s-a_t)/a_t
        de=np.linalg.norm(es-et,axis=1)
        dh=np.linalg.norm(hs-ht,axis=1)
        req.append(0.5*v_c*np.sqrt(da_rel**2+de**2)+v_c*dh)
    req=np.concatenate(req)
    fr=[f"{100*np.mean(req>dv_budget(f)):5.1f}%" for f in (0.08,0.113,0.15,0.20)]
    print(f"  {name:22s} {np.median(req):10.1f} {np.percentile(req,90):8.1f} "
          f"{np.percentile(req,99):8.1f} | {' '.join(fr)}")
print(f"\n  budgets: f=0.08 -> {dv_budget(0.08):.0f} m/s | 0.113 -> {dv_budget(0.113):.0f} | "
      f"0.15 -> {dv_budget(0.15):.0f} | 0.20 -> {dv_budget(0.20):.0f}")
print("  NOTE valid_init_only rejects on PERIGEE only; fuel never enters the")
print("  rejection sampler, so a low draw produces FEASIBLE-LOOKING but")
print("  UNSOLVABLE episodes rather than gave_up. That is the failure mode.")
