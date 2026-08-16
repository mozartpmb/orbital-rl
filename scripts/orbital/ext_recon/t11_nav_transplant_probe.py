import sys, os, math, importlib.util, torch
WT='/Users/pete/space_training-j2'
sys.path.insert(0, WT+'/pufferlib')
sp=importlib.util.spec_from_file_location('ev', WT+'/scripts/orbital/nav/eval_relnav3d.py')
ev=importlib.util.module_from_spec(sp); sp.loader.exec_module(ev)
R_EARTH=6.371e6
def transplant(src,dst,af=1.6e6,at=8.0e6,lf=6.371e6,lt=1.5e7):
    sd=torch.load(src,map_location='cpu',weights_only=True)
    key=next(k for k,v in sd.items() if 'encoder' in k and v.dim()==2 and v.shape[1]==38)
    W=sd[key].clone()
    for cols,f in {(0,7):af/at,(17,20):(R_EARTH+af)/(R_EARTH+at),(24,33,34):lf/lt}.items():
        for c in cols: W[:,c]=W[:,c]/f
    sd[key]=W; torch.save(sd,dst)
CK=WT+'/models/t3/j2nav_T-J2BO-nav.pt'; out='/tmp/t11_j2bo_wide.pt'
transplant(CK,out)
J2BAND=dict(i_target_min_rad=math.radians(30),i_target_max_rad=math.radians(60),
            raan_target_sample=0,lvlh_frame_mode=1)
COMMON=dict(num_debris_min=0,num_debris_max=0,e_max_target=0.05,e_max_sat=0.05,
  same_orbit_init=0,init_phase_gap_max=3.14159,valid_init_only=1,
  gave_up_action='terminate',max_valid_init_attempts=4096,
  rendezvous_radius_m=30000.0,rel_vel_tol_ms=50.0,shaping_mode=2,shape_w_lambda=1.0,
  shape_w_match=0.8166667,shape_dv_ref_ms=700.0,shape_gamma=1.0,phase_gap_mode=1,
  phase_obs_mode=1,episode_cap_steps=3000,cap_terminal_reward=0.0,dim3_mode=1,
  di_max_rad=0.017453,legacy_action_space=30,a_min_override=6.671e6,
  a_max_override=7.171e6,j2_mode=1,nav_j2_mode=1,**J2BAND)
NAR=dict(COMMON,obs_alt_scale_m=1.6e6,lvlh_scale_m=6.371e6)
WID=dict(COMMON,obs_alt_scale_m=8.0e6,lvlh_scale_m=1.5e7)
N=60
print(f"  J2BO-nav at J2X-loose, BEARINGS-ONLY (real IOD), {N} eps:")
for lab,kw,ck in (('orig @ narrow (home)',NAR,CK),('orig @ wide (barrier)',WID,CK),
                  ('TRANSPLANT @ wide',WID,out)):
    ev.rollout._mask_rows=None
    r=ev.rollout(ev.make_env('X3','bearings_only',acq='real',**kw),ck,N,123,lab)
    print(f"    {lab:24s} {r['success']:3d}/{r['n_valid']:<3d}  md5 {r['md5'][:12]}")
