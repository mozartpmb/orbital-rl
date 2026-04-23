# Orbital RL — Visualization & Web Frontend Spec

---

## Vision

A static website that tells the story of teaching an RL agent to fly spacecraft. Two layers of narrative:

1. **Within a run:** Watch the agent improve — from random thrashing to clean orbital transfers. Animated trajectory replays, training curves, and side-by-side comparisons.
2. **Across the project:** Track the development journey as the environment grows in complexity — circular transfers → fuel efficiency → elliptical orbits → rendezvous → debris avoidance. Each milestone is a chapter with its own visuals and commentary.

The site should feel like an interactive research log — part portfolio piece, part educational explainer, part tech demo.

---

## Audience & Goals

| Audience | What they care about | What they see |
|----------|---------------------|---------------|
| Hiring managers / recruiters | Execution, scope, polish | Clean site, clear narrative, impressive visuals |
| RL / aerospace engineers | Technical depth | Physics accuracy, reward design, training analysis |
| Curious generalists | "Cool factor" | Animated orbits, intuitive explanations, interactive controls |

**Design principle:** Lead with visuals, layer in technical depth for those who want it. Every section should be interesting at a glance and rewarding on closer inspection.

---

## Site Structure

```
/                       → Landing + hero animation
/journey                → Development timeline (all milestones)
/journey/fuel-shaping   → Phase 1 deep dive
/journey/elliptical     → Phase 2 deep dive
/journey/rendezvous     → Phase 3 deep dive
/run/:id                → Individual run explorer (trajectory replay + metrics)
/about                  → Physics primer + project motivation
```

Single-page app with client-side routing. All data is static (pre-baked JSON/binary files), no backend needed. Hostable on GitHub Pages, Netlify, Vercel, or Cloudflare Pages.

---

## Page Designs

### Landing Page

**Hero section:** A looping animation of a trained agent performing an orbital transfer with debris avoidance. Dark background (space), Earth at center, satellite trajectory drawn in real-time with a glowing trail. Burns shown as thrust flares. Debris orbits faintly visible. No controls needed — just an ambient, attention-grabbing loop.

Below the hero:

- One-line project description: *"Teaching a reinforcement learning agent to fly spacecraft using orbital mechanics simulated in C."*
- Three stat cards: `100% success rate` · `737K steps/sec` · `9 actions, 24 observations`
- "Explore the Journey →" CTA linking to `/journey`

### Development Journey Page (`/journey`)

A vertical timeline with milestone cards. Each card has:

- **Title** — e.g., "Phase 1: Learning to Transfer"
- **Key visual** — a small embedded trajectory animation or static plot (auto-plays on scroll)
- **Metrics summary** — success rate, fuel efficiency, training steps
- **One-paragraph narrative** — what changed, what the agent learned, what surprised us
- **"Deep Dive →"** link to the full phase page

Timeline milestones (grows as project progresses):

| # | Milestone | Key Visual |
|---|-----------|-----------|
| 1 | First successful orbit transfer | Side-by-side: random agent vs first convergence |
| 2 | Debris avoidance | Trajectory weaving between debris keepout zones |
| 3 | Fuel efficiency (v2 Phase 1) | Overlay: old spiral strategy vs new 2-impulse |
| 4 | Elliptical orbits (v2 Phase 2) | Transfer to a visibly eccentric target orbit |
| 5 | Rendezvous (v2 Phase 3) | Phasing orbit + intercept animation |

Each milestone is a self-contained story beat. The page works even if someone only reads the cards without clicking through.

### Phase Deep Dive Pages (`/journey/:phase`)

Each phase page has three sections:

**1. The Problem**
- What the agent needs to learn and why it's hard
- Diagram or animation showing the challenge (e.g., for rendezvous: "to catch something ahead, you slow down first")
- Keep it accessible — assume the reader knows what an orbit is but not what a Hohmann transfer is

**2. Training Progress**
- **Training curves:** Interactive chart (reward, success rate, fuel efficiency over training steps). Hover for values, toggle metrics on/off.
- **Trajectory montage:** Grid of trajectory snapshots at different training checkpoints (e.g., 0%, 25%, 50%, 75%, 100% of training). Shows the agent's strategy evolving.
- **Before/after animation:** Side-by-side trajectory replay — early checkpoint (chaotic) vs final checkpoint (clean). Synchronized playback so you can see the same scenario handled differently.

**3. Results & Analysis**
- Final eval metrics (success rate, fuel efficiency, conjunction distances)
- Comparison to analytical optimal (Hohmann Δv, phasing orbit Δv)
- Interesting observations (e.g., "the agent discovered that waiting for a debris window is cheaper than routing around")
- Static trajectory plots (the matplotlib ones you already generate, embedded as images or recreated in canvas)

### Run Explorer (`/run/:id`)

A detailed view of a single evaluation episode. This is the "cool interactive demo" page.

**Main panel — Trajectory Replay:**
- 2D orbital view: Earth at center, orbits drawn as ellipses, satellite animated along its trajectory
- Playback controls: play/pause, speed (1×, 2×, 5×, 10×), scrub bar, step forward/back
- Visual elements:
  - Satellite: small triangle oriented along velocity vector
  - Trajectory trail: color gradient (green → red by time, or colored by fuel remaining)
  - Burns: thrust flare effect + small arrow showing Δv direction
  - Target orbit: dashed line (cyan for circular, yellow for elliptical)
  - Debris: small dots with faint keepout radius circles, orbiting in real-time
  - Earth: blue circle with atmosphere glow, to scale
  - Conjunction warnings: pulse effect when satellite passes near debris

**Side panel — Telemetry:**
- Real-time readouts updating as the replay progresses:
  - Semi-major axis (with target shown as reference line)
  - Eccentricity (with target)
  - Fuel remaining (bar)
  - Current action (text label)
  - Distance to nearest debris
  - Episode step / sim time
- Small inline charts:
  - Semi-major axis vs time (a converging to a_target)
  - Fuel vs time (monotonically decreasing staircase)
  - Min debris distance vs time

**Data source:** Pre-exported JSON files converted from the `.npz` trajectory logs. Each run is a single JSON file (~50-200 KB depending on episode length).

---

## Data Pipeline

The visualization consumes pre-processed static data. No live training connection needed.

### Export Flow

```
Training / Eval (Python)
    │
    ▼
.npz trajectory logs (existing)
    │
    ▼
export_web_data.py (new script)
    │
    ├── runs/*.json          — individual episode replays
    ├── training_curves/*.json — reward/success over training steps
    ├── milestones.json      — metadata for journey timeline
    └── summary.json         — global stats for landing page
    │
    ▼
Static site build (Vite)
    │
    ▼
dist/ → deploy to hosting
```

### `export_web_data.py`

New script in `scripts/orbital/` that:

1. Reads `.npz` trajectory files from eval runs
2. Downsamples if needed (skip every N steps for long episodes to keep JSON small)
3. Converts to JSON with named fields:

```json
{
  "episode_id": 42,
  "metadata": {
    "checkpoint": "model_000153.pt",
    "phase": "debris",
    "success": true,
    "total_dv": 282.3,
    "hohmann_dv": 90.1,
    "fuel_efficiency": 0.319,
    "min_conjunction_km": 58.5,
    "num_steps": 43
  },
  "initial": {
    "sat_a_km": 6771,
    "target_a_km": 7171,
    "target_e": 0.0,
    "num_debris": 6
  },
  "steps": [
    {
      "t": 60,
      "x": 6771000, "y": 0,
      "a": 6771000, "e": 0.001, "theta": 0.05,
      "fuel": 0.98,
      "action": 1,
      "dv": 10,
      "min_debris_dist": 450000
    }
  ],
  "bodies": [
    {"type": "earth", "x": 0, "y": 0, "hard_r": 6371000, "keepout_r": 6571000},
    {"type": "debris", "a": 6900000, "e": 0.02, "M0": 1.2, "hard_r": 1, "keepout_r": 5000}
  ],
  "target": {
    "a": 7171000, "e": 0.0, "omega": 0
  }
}
```

Bodies that orbit (debris) are stored as orbital elements so the frontend can propagate them client-side for smooth animation (rather than storing x/y per step per body, which bloats the file).

### Training Curves

Extracted from TensorBoard logs or PufferLib's own logging:

```json
{
  "phase": "debris",
  "metrics": {
    "reward": { "steps": [0, 1000, 2000, ...], "values": [-8.2, -5.1, 2.3, ...] },
    "success_rate": { "steps": [...], "values": [...] },
    "mean_dv": { "steps": [...], "values": [...] },
    "mean_episode_length": { "steps": [...], "values": [...] }
  }
}
```

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | React (Vite) | Fast builds, good ecosystem, JSX artifacts if we prototype in Claude |
| Rendering | HTML Canvas 2D | Sufficient for 2D orbital viz, simpler than WebGL/Three.js, fast enough for smooth replay |
| Charts | Recharts or Chart.js | Training curves, telemetry panels |
| Styling | Tailwind CSS | Rapid iteration, dark theme is trivial |
| Routing | React Router | Client-side routing for SPA |
| Hosting | GitHub Pages or Netlify | Free, static, custom domain support |
| Build | Vite | Fast dev server, optimized production builds |

### Why Canvas 2D over Three.js

The simulation is 2D (coplanar orbits). Canvas 2D gives us:
- Smooth ellipse drawing with `ctx.ellipse()`
- Efficient particle trails with `ctx.globalAlpha` fade
- Simple coordinate transforms (translate + scale for zoom/pan)
- No WebGL context overhead, works everywhere

If we add 3D orbits later (inclination changes), we'd switch the main panel to Three.js. But for v1 visuals covering everything through rendezvous, 2D Canvas is the right call.

### Canvas Rendering Approach

```
Coordinate system:
- Origin at Earth center
- 1 pixel = variable scale (zoom level)
- Default view: Earth fills ~15% of viewport width

Render loop (requestAnimationFrame):
1. Clear canvas
2. Draw star field (static, cached to offscreen canvas)
3. Draw Earth (filled circle + atmosphere glow gradient)
4. Draw target orbit (dashed ellipse)
5. Draw debris orbits (faint gray ellipses) + current debris positions (dots)
6. Draw keepout zones (semi-transparent circles around debris)
7. Draw trajectory trail (polyline from step 0 to current step, colored by fuel or time)
8. Draw satellite (triangle at current position, oriented along velocity)
9. Draw burn effects (expanding circle + arrow if current step has dv > 0)
10. Draw conjunction warning (pulsing ring if near debris)
11. Draw HUD overlay (step counter, sim time, fuel bar)
```

Frame budget: at 60 fps with < 20 objects, Canvas 2D will be well under 1ms per frame. No performance concerns.

---

## Component Breakdown

### Shared Components

**`<OrbitalCanvas>`** — The core 2D orbital visualization. Used on every page.

Props:
- `episode` — full episode data (steps, bodies, target)
- `currentStep` — which step to render (controlled externally)
- `showTrail` — whether to draw trajectory history
- `showDebris` — toggle debris visibility
- `showKeepout` — toggle keepout zone visibility
- `showBurns` — toggle burn markers
- `zoom` / `center` — camera controls
- `highlightConjunctions` — pulse when near debris
- `staticMode` — render a single frame with full trail (for thumbnail/snapshot use)

**`<PlaybackControls>`** — Play/pause, speed, scrub bar, step buttons.

Props:
- `totalSteps`, `currentStep`, `onStepChange`
- `playbackSpeed`, `onSpeedChange`
- `isPlaying`, `onTogglePlay`

**`<TelemetryPanel>`** — Side panel with live readouts and mini-charts.

Props:
- `episode`, `currentStep`
- Shows: a, e, fuel, action, debris distance, step/time
- Mini line charts for a-vs-time and fuel-vs-time

**`<TrainingCurveChart>`** — Interactive line chart for training metrics.

Props:
- `data` — training curve JSON
- `metrics` — which metrics to show (togglable)
- `xRange` — zoom into a training step range

**`<MilestoneCard>`** — Card component for the journey timeline.

Props:
- `title`, `description`, `metrics`, `visual` (thumbnail or mini animation)

### Page Components

**`<LandingPage>`** — Hero animation (ambient `<OrbitalCanvas>` loop) + stat cards + CTA

**`<JourneyPage>`** — Vertical timeline of `<MilestoneCard>` components

**`<PhaseDeepDive>`** — Problem description + `<TrainingCurveChart>` + trajectory montage + before/after `<OrbitalCanvas>` pair

**`<RunExplorer>`** — Full `<OrbitalCanvas>` with `<PlaybackControls>` + `<TelemetryPanel>`

---

## Visual Design

### Color Palette (Dark Space Theme)

| Element | Color | Hex |
|---------|-------|-----|
| Background | Near-black | `#0a0e17` |
| Earth | Soft blue | `#4a9eed` |
| Earth atmosphere | Blue glow | `#4a9eed` at 20% opacity |
| Satellite | White/bright | `#ffffff` |
| Trajectory trail (start) | Green | `#22c55e` |
| Trajectory trail (end) | Amber | `#f59e0b` |
| Target orbit | Cyan dashed | `#06b6d4` |
| Initial orbit | Green dashed | `#22c55e` at 50% opacity |
| Debris | Orange dot | `#f97316` |
| Keepout zone | Orange | `#f97316` at 15% opacity |
| Burn flare | Yellow-white | `#fef08a` |
| Conjunction warning | Red pulse | `#ef4444` |
| Text / labels | Light gray | `#94a3b8` |
| Cards / panels | Dark slate | `#1e293b` |
| Accent | Indigo | `#6366f1` |

### Typography

- Headings: Inter or Space Grotesk (technical, clean)
- Body: Inter
- Monospace (code/metrics): JetBrains Mono or Fira Code
- Keep it simple — two fonts max

### Responsive Behavior

- Desktop (> 1024px): Full layout — canvas + side panel
- Tablet (768–1024px): Canvas full-width, telemetry collapses to bottom drawer
- Mobile (< 768px): Canvas full-width (square), telemetry as expandable accordion below. Playback controls simplified (play/pause + speed only, no scrub). Journey page works well since it's a vertical timeline already.

---

## Animation Details

### Trajectory Replay Timing

The replay maps simulation steps to real-time:

```
At 1× speed: 1 sim step = 200ms real time
At 2× speed: 1 sim step = 100ms real time
At 5× speed: 1 sim step = 40ms real time
At 10× speed: 1 sim step = 20ms real time
```

A typical 50-step episode plays in 10 seconds at 1× speed. Debris positions are interpolated between steps for smooth motion (Kepler propagation in JS — same math as the C code, just for visual smoothness).

### Client-Side Kepler Propagation (for debris animation)

To animate debris smoothly between discrete steps, implement a lightweight Kepler solver in JavaScript:

```javascript
function propagateOrbit(orbit, dt) {
  const n = Math.sqrt(MU / (orbit.a ** 3));
  const M = orbit.M0 + n * dt;
  const E = solveKepler(M, orbit.e);  // Newton-Raphson, 5 iterations
  const theta = 2 * Math.atan2(
    Math.sqrt(1 + orbit.e) * Math.sin(E / 2),
    Math.sqrt(1 - orbit.e) * Math.cos(E / 2)
  );
  const r = orbit.a * (1 - orbit.e * Math.cos(E));
  return {
    x: r * Math.cos(theta + orbit.omega),
    y: r * Math.sin(theta + orbit.omega)
  };
}
```

This is ~30 lines of JS and lets us animate all debris at 60fps without storing per-step positions.

### Burn Effects

When the current step has `dv > 0`:
1. Small expanding circle at satellite position (fades over 300ms)
2. Arrow pointing in burn direction (prograde/retrograde/radial), length proportional to Δv
3. Brief trail brightening (the trajectory line pulses brighter momentarily)

### Conjunction Warning

When `min_debris_dist` drops below 2× keepout radius (10 km):
1. Pulsing red ring around the nearby debris object
2. Dashed line connecting satellite to debris
3. Distance label on the connecting line

---

## Build Phases

### Web Phase 1: Core Replay (~3 days)

Build the `<OrbitalCanvas>` and `<RunExplorer>` page. This is the foundation everything else builds on.

Deliverables:
- [ ] `export_web_data.py` script — convert .npz to JSON
- [ ] `<OrbitalCanvas>` component — Earth, orbits, satellite, debris, trail, burns
- [ ] `<PlaybackControls>` — play/pause, speed, scrub
- [ ] `<TelemetryPanel>` — live readouts + mini charts
- [ ] `<RunExplorer>` page — canvas + controls + telemetry
- [ ] Client-side Kepler propagation for smooth debris animation
- [ ] Load episode JSON and replay it end-to-end
- [ ] Test with 3-5 exported episodes from current checkpoints

### Web Phase 2: Landing + Journey (~2 days)

Build the narrative layer.

Deliverables:
- [ ] `<LandingPage>` — hero animation loop + stats + CTA
- [ ] `<MilestoneCard>` component with thumbnail animations
- [ ] `<JourneyPage>` — vertical timeline with milestone cards
- [ ] `<TrainingCurveChart>` — interactive training metrics chart
- [ ] Export training curves from PufferLib logs
- [ ] Write narrative copy for existing milestones (v1 no-debris, v1 debris)

### Web Phase 3: Deep Dive Pages (~2 days)

Build the phase-level analysis pages. These grow as we complete v2 phases.

Deliverables:
- [ ] `<PhaseDeepDive>` page template
- [ ] Before/after trajectory comparison (side-by-side `<OrbitalCanvas>`)
- [ ] Trajectory montage (grid of static snapshots at training checkpoints)
- [ ] Write deep-dive content for v1 phases
- [ ] Wire up routing: landing → journey → phase → run explorer

### Web Phase 4: Polish & Deploy (~1 day)

- [ ] Responsive layout (tablet/mobile)
- [ ] Loading states and error handling
- [ ] Favicon, meta tags, Open Graph image (screenshot of trajectory for link previews)
- [ ] Deploy to GitHub Pages or Netlify
- [ ] README for the web repo with screenshot and link

---

## File Structure

```
orbital-web/
├── public/
│   ├── data/
│   │   ├── runs/                    # Episode JSON files
│   │   │   ├── v1_nodebris_001.json
│   │   │   ├── v1_debris_001.json
│   │   │   └── ...
│   │   ├── curves/                  # Training curve JSON files
│   │   │   ├── v1_nodebris.json
│   │   │   └── v1_debris.json
│   │   ├── milestones.json          # Journey timeline metadata
│   │   └── summary.json             # Landing page stats
│   └── og-image.png                 # Open Graph preview image
│
├── src/
│   ├── components/
│   │   ├── canvas/
│   │   │   ├── OrbitalCanvas.jsx    # Core 2D orbital renderer
│   │   │   ├── renderEarth.js       # Earth drawing logic
│   │   │   ├── renderOrbits.js      # Orbit ellipse drawing
│   │   │   ├── renderSatellite.js   # Satellite + trail + burns
│   │   │   ├── renderDebris.js      # Debris + keepout zones
│   │   │   └── kepler.js            # Client-side Kepler propagation
│   │   ├── PlaybackControls.jsx
│   │   ├── TelemetryPanel.jsx
│   │   ├── TrainingCurveChart.jsx
│   │   └── MilestoneCard.jsx
│   │
│   ├── pages/
│   │   ├── LandingPage.jsx
│   │   ├── JourneyPage.jsx
│   │   ├── PhaseDeepDive.jsx
│   │   └── RunExplorer.jsx
│   │
│   ├── hooks/
│   │   ├── usePlayback.js           # Playback state machine (play/pause/speed/step)
│   │   └── useEpisodeData.js        # Fetch + parse episode JSON
│   │
│   ├── lib/
│   │   ├── orbital-math.js          # Kepler solver, coordinate transforms
│   │   └── constants.js             # MU, R_EARTH, color palette, etc.
│   │
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css                    # Tailwind imports + custom dark theme
│
├── scripts/
│   └── export_web_data.py           # .npz → JSON conversion
│
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── README.md
```

---

## Content Strategy

Each milestone needs three things prepared:

1. **Curated episodes** — 3-5 representative eval runs exported as JSON. Pick a mix: one clean textbook-like transfer, one with an interesting debris avoidance maneuver, one failure case (if any). For before/after comparisons, export an early-training episode and a late-training episode from the same scenario.

2. **Training curves** — Export the full training run metrics. Mark key moments (reward spike, strategy shift, curriculum stage change) so we can annotate them on the chart.

3. **Narrative copy** — 2-3 paragraphs per milestone explaining: what changed in the environment, what the agent learned, what was surprising or interesting. Write in first person ("I noticed the agent developed a spiraling strategy..."). This is what turns a demo into a story.

### Copy Outline for Existing Milestones

**Milestone 1 — First Transfer (v1, no debris):**
> The first goal was simple: get from orbit A to orbit B. The agent started with completely random burns — sometimes escaping Earth's gravity entirely, sometimes crashing back into the atmosphere. After ~5M steps, it discovered that prograde burns raise your orbit. But instead of learning the textbook two-impulse Hohmann transfer, it developed a spiral strategy — dozens of tiny burns spread over many orbits. It works, but it uses 60% more fuel than optimal. Still, 96% success rate from a blank slate.

**Milestone 2 — Debris Avoidance (v1, with debris):**
> Adding debris changed the agent's personality. Warm-started from the no-debris checkpoint, it quickly learned to avoid all obstacles — minimum distance 58 km, well above the 5 km keepout zone. But it also became more aggressive: shorter episodes, more fuel burned, a "get there fast" mentality. The debris made it impatient. 100% success rate, but fuel efficiency dropped to 32% of optimal.

---

## Future Visual Upgrades (after v2 environment phases)

| Feature | When | Effort |
|---------|------|--------|
| Elliptical orbit rendering | After env Phase 2 | Low — `ctx.ellipse()` already supports this |
| ω visualization (periapsis marker) | After env Phase 2 | Low — small marker on orbit path |
| Phasing orbit animation | After env Phase 3 | Medium — need to show the "drop down, coast, come back up" pattern clearly |
| Target body animation | After env Phase 3 | Low — another dot on the target orbit |
| Rendezvous approach inset | After env Phase 3 | Medium — zoomed-in view of final approach |
| 3D view (Three.js) | After env adds inclination | High — full renderer rewrite |
| WASM live demo | Stretch goal | High — compile C env to WASM, run in browser |

---

*Spec written: 2026-04-13*
