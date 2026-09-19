# Curvilinear vs Rectilinear — Draper Question Prep (2026-09-19)

Prep for a question asked in the Draper interview that Pete did not fully catch: something like *"is your project curvilinear or rectilinear?"*, with the interviewer adding that the curved one is a lot more of a headache.

Two tiers. **Tier 1** is the plain version to actually understand. **Tier 2** is the version to say to a GN&C engineer, with the code and the literature behind it. A **clarifying question to ask back** is at the end of Tier 1, because the phrase has three plausible referents and the answer differs for each.

Everything about the project below was verified against the code on 2026-09-19 (paths are under `pufferlib/pufferlib/ocean/orbital/` and `orbital_nav/`). Literature claims are cited to Vallado & Alfano (AAS 11-464), Bennett & Schaub (AAS 16-336), and the Willis & D'Amico citation already in `scripts/orbital/ext_recon/reports/nav_F_observability.md`.

---

## Tier 1 — the plain version

### What the words mean

When you describe where the chaser is *relative to the target*, you need a coordinate system that rides along with the target. Everyone uses the same three axes:

- **Radial** — up/down, toward or away from Earth.
- **Along-track** — forward/backward along the orbit.
- **Cross-track** — sideways, out of the orbit plane.

The whole question is about the **along-track** axis. The target is moving on a circle around Earth. If the chaser is 100 km "behind" the target, there are two ways to say what "behind" means:

- **Rectilinear** (straight-line): draw a straight line from the target, tangent to its orbit, and measure 100 km along it. This is a flat ruler laid against a curved road.
- **Curvilinear** (curved-line): measure 100 km *along the orbit arc itself*, or equivalently say "the chaser is some angle behind the target on the same circle." This is a tape measure bent to follow the road.

For a few kilometres of separation, the road is flat enough that both agree. At 100 km in low Earth orbit, the straight ruler sticks out of the circle: a point 100 km along the straight tangent line sits about 740 m *higher* than a point 100 km along the arc. That is the whole difference, and it grows with the square of the separation.

Why the curved one is a headache (this is almost certainly what the interviewer meant):

1. Converting between curvilinear coordinates and real positions/velocities is a nonlinear transform, and getting the **velocity** right is the fiddly part.
2. If the target orbit is elliptical, "distance along the arc" is not just angle times radius any more. It becomes an elliptic integral; Vallado and Alfano gave up on the closed form and did it numerically.
3. The transform only makes sense for separations under about half an orbit.

Why people put up with the headache: for large along-track separations (long-range rendezvous, space-situational-awareness tracking), the straight-line version is wrong by hundreds of metres to kilometres, and the curvilinear version fixes that with the same linear equations. Curvilinear also makes bearings-only navigation observable where the straight-line version is not.

### What your project does, in one breath

**Mixed, and it is mixed on purpose.** Your project uses the straight-line (rectilinear) version for the part the policy looks at up close and for the finish line, and the curved (curvilinear, angle-based) version for phasing far away. And underneath both, the actual physics is exact orbital mechanics, not the linearised straight-line-or-curved approximation at all.

Concretely:

- **The chaser's own state** is kept as classical orbital elements (size, shape, angle around the orbit) and advanced with Kepler's equation each step. No numerical integrator, no Clohessy-Wiltshire approximation.
- **The LVLH observation block the policy sees** (five numbers) is a straight-line offset from the target rotated into the target's radial/along-track frame. Rectilinear.
- **The success test** is "inside a 30 km sphere around the target, moving slower than 50 m/s relative." A sphere is a straight-line distance. Rectilinear.
- **The phasing observation and the reward shaping potential** both use the *angle* between chaser and target around the orbit (the mean-longitude gap, written Δλ). An angle around the orbit is a curvilinear along-track coordinate.
- **The navigation filters**: the range-plus-bearing filter estimates the target's position in plain inertial x, y coordinates. The bearings-only filters estimate in modified polar / modified spherical coordinates (angles, log of range), which are curvilinear in the ordinary mathematical sense and were chosen for observability.

### If you get asked again — what to ask back

> "Do you mean the coordinates of the relative state in the LVLH frame, the filter's state parameterisation, or the shape of the approach trajectory? The along-track coordinate in my LVLH observation is a rectilinear chord; the phasing channel and shaping potential use the mean-longitude angle, which is curvilinear; and the truth dynamics are exact Kepler with no linearisation."

That one sentence shows you know the three meanings and have a real answer for each.

---

## Tier 2 — the version for a GN&C engineer

### The three things the question could mean

**A. Relative-state coordinates in the LVLH / Hill frame.** The classic meaning. Rectilinear = Cartesian (x, y, z) in the rotating radial/transverse/normal frame; the Clohessy-Wiltshire (HCW) equations are normally written this way. Curvilinear = radial offset plus along-track *arc* (r·Δθ) or *angle* (Δθ, or D'Amico's relative mean argument of latitude δλ) plus cross-track angle. Multiplying the angle by r only changes units; both are curvilinear. Vallado & Alfano's EQCM (modified equidistant cylindrical) frame uses λ, φ angles. JSC's operational "LVLH Rotating Curvilinear" frame uses the arc.

**B. Navigation filter state parameterisation.** Bennett & Schaub (AAS 16-336) is exactly this: an EKF with the CW integration constants as state, in rectilinear vs curvilinear coordinates, with bearings-only measurements. Their conclusion: *"For larger relative orbits, the curvilinear LROE form provides full state observability with bearings-only measurements and greater fidelity with additional measurements."*

**C. Approach trajectory shape.** Glideslope guidance flies a straight line (V-bar or R-bar) to the target; the natural relative motion is curved. Less likely to be what was asked, but worth having an answer for.

### The project's answer for each

**A. Relative state: mixed, rectilinear up close, curvilinear for phasing.**

- The 38-dim observation contains a five-element LVLH block, obs[33..37] (`orbital.h` around lines 1478–1545). It is the straight inertial difference vector `sat − target` rotated by the target's in-plane angle u = ω + θ into radial and along-track components, with the rotating-frame velocity correction applied using the target's circular mean motion n = √(μ/a³). That along-track component is a chord, not an arc. Rectilinear.
- The phasing block obs[13..16] (around lines 1258–1300) carries the along-track separation as the mean-longitude gap Δλ = (M+ω)_sat − (M+ω)_target, encoded as sin and cos. In 3D it uses a target-plane gauge so the gap is a function of the physical relative state only. An angle around the orbit, never multiplied by r. Curvilinear.
- The shaping potential in the shipped mode (S-R3, `compute_phi` around lines 1578–1605) is −W_λ·|Δλ|/π − W_match·(Δv-to-go). Curvilinear along-track term plus an element-space size/shape term.
- The terminal success test (around lines 1818–1827) is a Euclidean 3D sphere, `sqrt(dx²+dy²+dz²) < 30 km`, and relative speed under 50 m/s. Rectilinear.

Why the angle for phasing: the earlier version used the true-anomaly gap θ_s − θ_t. That gap is sign-wrong versus the physical separation on 39% of e>0 steps and jumps about 86° on a 1 m/s burn near e≈0, because θ is not a clean function of the relative state. The mean-longitude gap is burn-continuous and sign-correct. This was a fix motivated by the dynamics bug found on 2026-08-10, which had been teleporting phase on every burn.

Why the chord for the LVLH block and the success test: the classifier only ever looks inside 30 km, where the rectilinear error is tens of metres.

**How big the discrepancy actually is.** Place a point at rectilinear along-track offset s (on the tangent line) and another at curvilinear arc s (on the circle). The second sits lower by about s²/(2r) radially. That is the "y displacement has a small negative x component" figure in Vallado & Alfano. In LEO (r ≈ 6778 km):

| Along-track separation s | Radial mismatch s²/(2r) |
|---|---|
| 30 km (success box) | ~66 m |
| 100 km | ~740 m |
| 300 km | ~6.6 km |

Note the precise wording: this is a *radial* bookkeeping error between the two conventions, not a difference in the length of the arc versus the chord (that difference is ~s³/(24r²), under a metre at 100 km). The explainer v4 gloss at line 1189 already says "in the radial direction"; keep that phrasing.

**B. Filters: Cartesian for range-bearing, curvilinear (modified polar / spherical) for bearings-only.**

- `BatchedRangeBearingEKF` (`orbital_nav/nav_math.py` ~418–534): 4-state absolute inertial Cartesian target state [x, y, vx, vy], range and inertial-bearing measurements, analytic 2×4 Jacobian, Joseph-form update.
- `BatchedBearingMPC` (~610–757): 4-state modified polar [β, β̇, ρ̇/ρ, ln ρ], bearing-only. The measurement matrix is exactly e₁, so the update is linear; the nonlinearity is pushed into the transition.
- `BatchedBearingMSC6` (`nav_math3d.py` ~739+): 6-state modified spherical [az, el, ω_az, ω_el, ρ̇/ρ, ln ρ], built in a pole frame from the chaser's (r̂, ĥ×r̂, ĥ) at t₀.
- All three propagate with exact two-body Kepler (Lagrange f and g, finite-difference or analytic STM), not CW/Hill. Under J2 there is a C kernel for the same functions.

The reason this matters for the question: the repo's observability report (`nav_F_observability.md` §1.5–1.6) cites Willis & D'Amico (AAS 20-493 / ASR 2024) that Woffinden's bearings-only dilemma *"applies specifically to linear models in Cartesian coordinates … the relative state is observable for linear models in curvilinear coordinates."* The only scale-carrying term in the bearing model is the orbit-curvature term |δλ|/2 = ρ/(2r). Your filters get range observability the same way, by keeping the curvature: fully nonlinear Kepler dynamics plus a polar/spherical state.

**C. Trajectory: no glideslope.** The policy issues impulses and coasts on Keplerian arcs. It does not fly a straight-line V-bar or R-bar approach. The 30 km / 50 m/s box is the handoff point where a glideslope-style proximity-ops law would take over in a real mission.

### Where the dynamics stand relative to the HCW question

There is no HCW anywhere in the environment or the filters (grep confirms zero hits for Clohessy, Hill, or CW as relative-motion terms). The chaser and target are propagated as osculating classical elements through Kepler's equation (Newton, 5 iterations), and burns round-trip elements → Cartesian → +Δv → elements. So the usual "curvilinear HCW is 100× more accurate than rectilinear HCW at large separation" tradeoff (Willis, Alfriend, D'Amico second-order work; Vallado & Alfano) does not apply to the truth model. It applies only to how the relative state is *reported* to the policy, which is the mixed answer above.

### Two limitations to volunteer

1. **The LVLH frame rotates at the target's circular mean motion**, n = √(μ/a³), not the target's instantaneous θ̇. For an eccentric target the along-track velocity component is approximate. The along-track position is still exact.
2. **The default LVLH frame mode (mode 0) is an equatorial projection**, rotating by u = ω + θ only. It equals true LVLH only at i = Ω = 0. Mode 1 builds the true (r̂, ĥ×r̂) triad. The code comment says this outright, and the 3D campaigns use mode 1.

And one J2 caveat that fits the same "we know our approximations" theme: under J2 the environment's state is the secular mean element set with no mean-to-osculating conversion anywhere, and a burn is applied to the mean state through the osculating Gauss response. That inconsistency is O(J2), measured at 83 m / 0.094 m/s per orbit at 5 km separation against a full-J2 Cowell reference.

### The 60-second spoken answer

> "Mixed, and deliberately. The truth dynamics are classical elements propagated in closed form through Kepler's equation, with burns round-tripped through Cartesian, so there's no Clohessy-Wiltshire linearisation anywhere and the curvilinear-versus-rectilinear HCW accuracy question doesn't arise in the physics. Where it does arise is in how I report the relative state. The five-element LVLH observation and the 30-kilometre success sphere are rectilinear: a straight inertial difference vector rotated into the target's radial/along-track frame. The phasing observation and the shaping potential are curvilinear: the mean-longitude gap Δλ, an angle around the orbit. I picked the angle for phasing because the true-anomaly gap is sign-wrong and teleports on burns near circular, and I kept the chord for the terminal box because the rectilinear radial error is s²/2r, about 66 metres across 30 kilometres, which the classifier can't see. For navigation, the range-bearing filter is Cartesian inertial, and the bearings-only filters use modified polar and modified spherical states with exact Kepler propagation, which is the curvilinear route to range observability that Willis and D'Amico describe. The things I'd fix next are the LVLH frame rate, which uses circular mean motion, and the default frame being an equatorial projection that's only exact at zero inclination."

### Vocabulary that will come up

- **LVLH / Hill / RTN / RSW / RIC**: all names for the rotating radial, along-track, cross-track frame centred on one spacecraft. RTN = radial, transverse, normal. NTW is the variant aligned with the velocity vector rather than the transverse axis (matters only for eccentric orbits).
- **HCW / CW**: Hill-Clohessy-Wiltshire, the linearised relative-motion equations for a circular reference orbit. Written in rectilinear coordinates by default; the curvilinear form has the same linear terms and different second-order terms.
- **ROE**: relative orbital elements (D'Amico). Differences of element-like quantities: δa, δλ, δe vector, δi vector. δλ is a curvilinear along-track coordinate.
- **LROE**: linearised relative orbit elements (Schaub). The CW integration constants used as a filter state.
- **Woffinden's dilemma**: with bearings only and linear Cartesian dynamics, range is unobservable unless you manoeuvre. Curvilinear or nonlinear models break the ambiguity through orbit curvature.
- **Modified polar / modified spherical coordinates**: the bearings-only tracking parameterisation [β, β̇, ρ̇/ρ, ln ρ] and its 3D extension. Decouples the observable angles from the weakly-observable range.
- **Glideslope, V-bar, R-bar**: straight-line final-approach guidance along the velocity axis or the radial axis.

---

## Sources

- Vallado, D. A., and Alfano, S., "Curvilinear Coordinates for Covariance and Relative Motion Operations," AAS 11-464. [PDF](https://www.agi.com/getmedia/edad243a-e818-4db3-a3ba-ec8548403084/Curvalinear-Coordinates-for-Covariance-and-Relative-Motion-Operations.pdf?ext=.pdf) — EQCM frame, Figure 2 radial mismatch, elliptic-integral arc length, half-orbit validity limit, "hundreds of metres to many kilometres" of Hill's error even in circular LEO.
- Bennett, T., and Schaub, H., "Relative Motion Estimation Using Rectilinear and Curvilinear Linearized Relative Orbit Elements," AAS 16-336. [PDF](https://hanspeterschaub.info/Papers/Bennett2016.pdf) — curvilinear LROE EKF, bearings-only observability.
- Willis, M., Alfriend, K. T., and D'Amico, S., "Second-Order Solution for Relative Motion on Eccentric Orbits in Curvilinear Coordinates," AAS 19-810. [PDF](https://slab.sites.stanford.edu/sites/g/files/sbiybj25201/files/media/file/aas2019_willisalfrienddamico_final.pdf) — linear terms identical between rectilinear and curvilinear, curvilinear ~100× more accurate at first order.
- Willis & D'Amico on Woffinden's dilemma, as cited in the repo's `scripts/orbital/ext_recon/reports/nav_F_observability.md` §1.5.
- Glideslope background: Ariba & Arzelier, "V-bar and R-bar Glideslope Guidance Algorithms for Fixed-Time Rendezvous." [HAL](https://hal.science/hal-01358188/document)
- Project code: `pufferlib/pufferlib/ocean/orbital/orbital.h` (LVLH obs ~1478–1545, phase obs ~1258–1300, `compute_phi` ~1578–1715, success test ~1818–1827, J2 ~758–830); `pufferlib/pufferlib/ocean/orbital_nav/nav_math.py`, `nav_math3d.py`.
