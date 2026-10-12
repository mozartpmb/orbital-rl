import sys
p = sys.argv[1]; s = open(p).read()
R = [
(r"\textbf{Budgets} \; $\sim$480\,m/s of \dv\ (15\,\% propellant fraction, Isp 300\,s, Tsiolkovsky) and a wall clock measured in sub-steps of 60\,s",
 r"\textbf{Budgets} \; 478\,m/s of \dv\ (15\,\% propellant, Isp 300\,s, Tsiolkovsky), about 2.2$\times$ a Hohmann transfer across the band; a wall clock counted in 60\,s sub-steps"),
(r"Fuel ~480 m/s versus ~110 m/s for a textbook Hohmann 400->800 km, so the agent cannot brute-force.", r"Fuel 478 m/s versus 217 m/s for a textbook Hohmann 400->800 km (2.2x), so the agent cannot brute-force."),
("box/.style={draw,rounded corners,minimum height=1.05cm,text width=2.35cm,align=center,font=\\small,fill=white},",
 "box/.style={draw,rounded corners,minimum height=1.0cm,text width=2.05cm,align=center,font=\\footnotesize,fill=white},"),
("right=0.55cm of sim] (sen)","right=0.4cm of sim] (sen)"),("right=0.55cm of sen] (nav)","right=0.4cm of sen] (nav)"),
("right=0.55cm of nav,fill=accent!10] (pol)","right=0.4cm of nav,fill=accent!10] (pol)"),("right=0.55cm of pol] (act)","right=0.4cm of pol] (act)"),
(r"\item The same loop runs in pure C on the embedded port (no heap, bounded execution, bit-exact to PyTorch).",
 r"\item The same loop runs in allocation-free C99 on the embedded-style port: bounded execution, action stream identical to PyTorch over 200 episodes."),
(r"{\scriptsize\textcolor{grey}{Earth termination at the surface radius; keep-out floor at 200\,km altitude is an observation feature, not a terminal.}}",
 r"{\scriptsize\textcolor{grey}{Two-body plus secular J2 only; the state is the mean element set. Earth termination at the surface radius.}}"),
(r"""  \foreach \y/\lab in {2.5/prograde $\pm$, 2.0/radial $\pm$, 1.5/normal $\pm$}{
    \node[anchor=east,font=\scriptsize] at (1.6,\y) {\lab};
    \foreach \x/\m in {2.0/1, 2.6/2, 3.2/5, 3.8/10, 4.4/25}{
      \fill[accent!70] (\x,\y) circle (0.11);
    }
  }""",
 r"""  \node[anchor=east,font=\scriptsize] at (1.6,2.5) {prograde $\pm$};
  \foreach \x in {2.0,2.6,3.2,3.8,4.4}{ \fill[accent!70] (\x,2.5) circle (0.11); }
  \node[anchor=east,font=\scriptsize] at (1.6,2.0) {radial $\pm$};
  \foreach \x in {2.0,3.8}{ \fill[accent!70] (\x,2.0) circle (0.11); }
  \node[anchor=east,font=\scriptsize] at (1.6,1.5) {normal $\pm$};
  \foreach \x in {2.0,3.8,4.4}{ \fill[accent!70] (\x,1.5) circle (0.11); }
  \node[anchor=east,font=\scriptsize] at (1.6,1.0) {combined};
  \node[font=\scriptsize,accent!80!black] at (3.2,1.0) {$\pm25$ tangential with $\pm25$ normal};"""),
(r"\foreach \x/\m in {2.0/1, 2.6/2, 3.2/5, 3.8/10, 4.4/25}{ \node[font=\scriptsize] at (\x,1.0) {\m}; }",
 r"\foreach \x/\m in {2.0/1, 2.6/2, 3.2/5, 3.8/10, 4.4/25}{ \node[font=\scriptsize] at (\x,0.55) {\m}; }"),
(r"\node[font=\scriptsize,grey] at (3.2,0.6) {\dv\ magnitude (m/s); normal-axis grades are a subset};",
 r"\node[font=\scriptsize,grey] at (3.2,0.2) {\dv\ magnitude (m/s), 31 actions in all};"),
(r"\textbf{PPO} \; \nEnvs\ envs, $\gamma=0.995$, clipped surrogate $\pm0.2$, GAE, entropy bonus 0.01, 50\,M steps per curriculum rung",
 r"\textbf{PPO} \; \nEnvs\ envs, $\gamma=0.995$, clipped surrogate $\pm0.2$, GAE $\lambda=0.9$, entropy 0.01, Muon optimizer, 50\,M steps per curriculum rung (about 2\,h each)"),
(r"\includegraphics[width=0.92\textwidth]{traj_rendezvous.pdf}", r"\includegraphics[width=0.74\textwidth]{traj_rendezvous.pdf}"),
(r"  \item Re-checked today on a fresh evaluation seed: 200/200.", r"  \item Re-run for this deck on a fresh evaluation seed: 200/200."),
(r"{\scriptsize\textcolor{grey}{The policy did not learn to \emph{seek} information (measured null); it learned to wait for acquisition and spend nothing while blind (\dv\ before acquisition: median 75$\to$0\,m/s).}}",
 r"{\scriptsize\textcolor{grey}{It did not learn to \emph{probe} (measured null); it learned to wait for acquisition and burn nothing while blind (median \dv\ before acquisition 75$\to$0\,m/s).}}"),
(r"\includegraphics[width=\textwidth]{err_3sigma_crop.png}\\", r"\includegraphics[width=0.9\textwidth]{err_3sigma_crop.png}\\"),
(r"""\small
\begin{tabular}{@{}llllllll@{}}
\toprule
skill & LEO & $e\le0.10$ & $e\le0.20$ & $e\le0.30$ & long range & tight box & plane change\\
\midrule
what changes & baseline & shape & to 3,500\,km & to 8,000\,km & 200\,h clock & \tightR,\,\tightV & $\Delta i$ via J2 drift\\
\bottomrule
\end{tabular}
\\[1mm]{\scriptsize\textcolor{grey}{All cells bearings-only, 200 held-out episodes each, single training seed; tight box reaches 92.5\,\% at its training fuel floor. Six single-skill specialists score 0--35\,\% off their home cell, which is why a generalist was worth building.}}""",
 r"""\vspace{-1mm}{\scriptsize\textcolor{grey}{All cells: 3-D, J2, bearings-only navigation, fuel budget sampled per episode, 200 held-out episodes each, one training seed. Tight box reaches 92.5\,\% at its training fuel floor. Single-skill specialists average 6--42\,\% off their home cell, which is why one policy was worth building. Cell definitions in A5.}}"""),
(r"\includegraphics[width=\textwidth]{generalist_cells.pdf}", r"\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{generalist_cells.pdf}"),
(r"\IfFileExists{figs/traj3d_rendezvous.pdf}{\includegraphics[width=\textwidth]{traj3d_rendezvous.pdf}}{\includegraphics[width=\textwidth]{traj_overlay.pdf}}",
 r"\IfFileExists{figs/traj3d_rendezvous.pdf}{\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{traj3d_rendezvous.pdf}}{\includegraphics[width=0.85\textwidth]{traj_overlay.pdf}}"),
(r"  \item A sign error in the true-to-mean anomaly conversion let every burn teleport phase by $\sim\!25^\circ$ per episode. The old headline fell from 97.5\,\% to 13\,\% under correct dynamics; every number here is post-fix.",
 r"  \item A swapped half-angle factor in the true-to-mean anomaly conversion let every burn teleport phase (median $25^\circ$ per episode). The old headline fell from 97.5\,\% to 13\,\% under correct dynamics; every number here is post-fix."),
(r"  \item Nine metric-versus-implementation mismatches, two by hand.", r"  \item A ledger of seven metric-versus-implementation bugs, each with a before/after number."),
(r"  \item Two-body plus secular J2 only; no drag, no third body.", r"  \item Arrival is certified, not station-keeping; two-body plus secular J2 only."),
(r"  \item Rendezvous with a maneuvering target; continuous low thrust.", r"  \item A matched MPC comparator; a maneuvering target."),
(r"{\scriptsize\textcolor{grey}{2-D range/bearing EKF on the headline policy (\texttt{plots/relnav\_t3}); success flat to 10$\times$ nominal noise because the converged filter error (tens of metres) is far inside the 30\,km box. Beyond 30$\times$ the limit is filter divergence, not guidance.}}",
 r"{\scriptsize\textcolor{grey}{2-D range/bearing EKF ($\sigma_\rho$ 50\,m, $\sigma_\beta$ 1\,mrad) on the headline policy; flat to 10$\times$ noise because converged error (tens of metres) is far inside the 30\,km box; beyond 30$\times$ the limit is filter divergence.}}"),
(r"\includegraphics[width=\textwidth]{noise_sweep.pdf}" + "\n" + r"\small", r"\includegraphics[width=\textwidth]{noise_sweep.pdf}" + "\n" + r"\footnotesize"),
(r"  \item Parity: 100\,\% action agreement over 7,048 observation vectors; 200/200 closed-loop episodes bit-identical to PyTorch.",
 r"  \item Parity: 100\,\% action agreement over 7,048 observation vectors; 200/200 closed-loop episodes with an identical action stream to PyTorch."),
(r"  \item Deterministic and reentrant; AddressSanitizer and UBSan clean.", r"  \item Deterministic, reentrant; ASan and UBSan clean."),
(r"\begin{column}{0.5\textwidth}\small" + "\n" + r"\begin{itemize}" + "\n" + r"  \item No heap:", r"\begin{column}{0.5\textwidth}\footnotesize" + "\n" + r"\begin{itemize}" + "\n" + r"  \item No heap:"),
(r"\textbf{What} \; the true-anomaly $\to$ mean-anomaly conversion after a burn had its sign inverted. Each burn silently rotated the chaser's phase; over an episode that was $\sim\!25^\circ$ of free phasing.",
 r"\textbf{What} \; the true-anomaly $\to$ mean-anomaly conversion, called only after a burn, had its $\sqrt{1\pm e}$ factors swapped. Each burn off an apse silently moved the chaser along-track (up to 700\,km at $e=0.05$); a median $25^\circ$ of free phasing per episode. Burns at apsides were unaffected, so every unit test passed."),
(r"\textbf{How it was found} \; a Lambert audit: learned \dv\ was \emph{below} the two-impulse optimum on many episodes (ratio $<0.5$), which is impossible. The teleport diagnostic then showed an $86.5^\circ$ true-anomaly jump for a 25\,m/s burn at $e=0$ against a 1\,km displacement.",
 r"\textbf{How it was found} \; twice on the same day: by formula inspection while porting the dynamics into the navigation filter, and by the Lambert audit, where 15\,\% of episodes beat half the time-matched two-impulse optimum. A diagnostic then showed an $86.5^\circ$ true-anomaly jump for a 25\,m/s burn against a 1\,km displacement."),
(r"for zero failures the rule of three gives $p<1.5$\,\%. I report intervals, and I do not claim 100\,\% means zero.",
 r"for 0/200 the two-sided Clopper--Pearson lower bound is 98.2\,\%. I report intervals, and I do not say 100\,\% means zero."),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print("patched", len(R))
