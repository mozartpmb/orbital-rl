import sys
p = sys.argv[1]; s = open(p).read()
R = [
# slide 5 combined label
(r"\node[font=\scriptsize,accent!80!black] at (3.2,1.0) {$\pm25$ tangential with $\pm25$ normal};",
 r"\node[font=\scriptsize,accent!80!black,anchor=west] at (1.75,1.0) {$\pm25$ tangential $+$ $\pm25$ normal};"),
# slide 7 shrink
(r"\includegraphics[width=0.66\textwidth]{traj_rendezvous.pdf}", r"\includegraphics[width=0.6\textwidth]{traj_rendezvous.pdf}"),
(r"\begin{column}{0.62\textwidth}\small" + "\n" + r"\begin{itemize}" + "\n" + r"  \item \key{200/200}",
 r"\begin{column}{0.62\textwidth}\footnotesize" + "\n" + r"\begin{itemize}" + "\n" + r"  \item \key{200/200}"),
# slide 10 handled inline
(r"""a 3-degree plane change by burning would exceed the whole tank.}
\end{frame}""",
 r"""a 3-degree plane change by burning would exceed the whole tank.}
\end{frame}

% =====================================================================
\begin{frame}{The generalist flying: 3-D, J2, bearings-only, one of the 200 held-out episodes}
\centering
\includegraphics[width=0.92\textwidth]{traj3d_rendezvous.pdf}
\\[-1mm]{\scriptsize\textcolor{grey}{Cell E1\_j2, T15e checkpoint. A $158^\circ$ gap and a $0.9^\circ$ plane difference closed in 22.5\,h with 306\,m/s of 471 and 23 decisions, flying on the filter's estimate. Panel (b) is the phasing loop seen from the target; (b2) the cross-track oscillation the plane burns null at the end; (c) the closure.}}
\note{Point at (b): the chaser parks above the target so the target catches up, then descends. Panel (b2): out-of-plane is left alone until the end, then nulled with three normal burns. Everything here is what an EO sensor on the chaser would see turned into a decision.}
\end{frame}"""),
# A2 shrink
(r"\includegraphics[width=\textwidth]{traj_closing.pdf}", r"\includegraphics[width=0.95\textwidth]{traj_closing.pdf}"),
(r"\textbf{Plane change by J2} \; $\dot\Omega = -\tfrac32 J_2 n \left(\tfrac{R_E}{p}\right)^2 \cos i$; the rate depends on $a$ and $i$, so two orbits at different altitudes drift apart in $\Omega$ at a controllable relative rate. The seventh skill dips low, waits with day-long coasts, and lets the node drift close the plane gap.",
 r"\textbf{Plane change by J2} \; $\dot\Omega = -\tfrac32 J_2 n (R_E/p)^2 \cos i$ depends on $a$ and $i$, so two orbits at different altitudes drift apart in $\Omega$ at a controllable rate. The seventh skill dips low, coasts a day at a time, and lets differential nodal precession close the plane gap (about half the alignment bought this way in the specialist)."),
# A6 shrink
(r"\includegraphics[width=0.92\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=0.92\textwidth]{action_hist.pdf}", r"\includegraphics[width=0.8\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=0.8\textwidth]{action_hist.pdf}"),
# A4: add nav3d estimate slide after A4
(r"""\begin{frame}{A5 \; The seven cells of the generalist mixture}""",
 r"""\begin{frame}{A4b \; Bearings-only filter in the loop, 3-D: acquisition then convergence}
\centering
\includegraphics[width=0.8\textwidth]{nav3d_estimate.pdf}
\\[-1mm]{\scriptsize\textcolor{grey}{Same episode as the 3-D rendezvous slide. The policy is blind until the batch solver accepts a solution, then the recursive filter's error collapses as range closes; the $3\sigma$ envelope is the filter's own claim.}}
\end{frame}

\begin{frame}{A5 \; The seven cells of the generalist mixture}"""),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print("patched", len(R))
