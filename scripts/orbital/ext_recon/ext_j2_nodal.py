import math
MU=3.986004418e14; RE=6.371e6; J2=1.0826267e-3; BUDGET=478.0
def n_of(a): return math.sqrt(MU/a**3)
def Odot(a,e,i):
    p=a*(1-e*e); n=n_of(a)
    return -1.5*J2*(RE/p)**2*n*math.cos(i)
print("Differential nodal precession as a FREE plane-change mechanism (if J2 were modelled)")
print("dOmega_dot/domega_dot = -3.5 * da/a  (circular);  free dOmega accrued over the episode clock\n")
for alt in [550, 800, 2000]:
  a=RE+alt*1e3
  for inc in [0.0, 28.5, 51.6, 97.8]:
    i=math.radians(inc); Od=Odot(a,0.0,i)
    print(f"alt {alt:5d} km  i={inc:5.1f} deg  Omega_dot = {math.degrees(Od)*86400:8.4f} deg/day")
    for da_km in [10,50,100,200]:
      dOd = Od*(-3.5)*(da_km*1e3/a)
      for clock_h in [33.3, 100.0, 200.0]:
        dOm = math.degrees(dOd)*(clock_h*3600)
        if clock_h==33.3:
          # equivalent di magnitude and its dv cost
          di = abs(dOm)*math.sin(i)   # deg, delta_i_y component
          dv = math.sqrt(MU/a)*math.radians(di)
          print(f"    da={da_km:4d} km -> dOmega over {clock_h:5.1f} h = {dOm:9.5f} deg "
                f"(di_y={di:8.5f} deg = {dv:7.3f} m/s equiv)")
        else:
          di = abs(dOm)*math.sin(i); dv=math.sqrt(MU/a)*math.radians(di)
          print(f"    da={da_km:4d} km ->                 {clock_h:5.1f} h = {dOm:9.5f} deg "
                f"(di_y={di:8.5f} deg = {dv:7.3f} m/s equiv)")
    print()
