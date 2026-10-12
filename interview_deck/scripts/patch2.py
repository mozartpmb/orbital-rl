import sys
p = sys.argv[1]; s = open(p).read()
R = [
# slide 4: tighten
(r"\begin{column}{0.55\textwidth}\small" + "\n" + r"\textbf{State}", r"\begin{column}{0.55\textwidth}\footnotesize" + "\n" + r"\textbf{State}"),
# slide 5: scale tikz
(r"\begin{tikzpicture}[x=1cm,y=1cm]", r"\begin{tikzpicture}[x=0.93cm,y=0.93cm]"),
# slide 7
(r"\includegraphics[width=0.74\textwidth]{traj_rendezvous.pdf}", r"\includegraphics[width=0.66\textwidth]{traj_rendezvous.pdf}"),
(r"""\begin{column}{0.36\textwidth}\scriptsize\textcolor{grey}{
Shown: a $91^\circ$ gap closed in 8.1\,h with 210\,m/s. The policy sits low and behind, lets the lower orbit catch up, then raises in three steps. Source: \texttt{models/t3/seed42\_L2\_headline.pt}, \texttt{T3\_RECOVERY\_CAMPAIGN.md} \S7.}""",
 r"""\begin{column}{0.36\textwidth}\scriptsize\textcolor{grey}{
Shown: a $91^\circ$ gap closed in 8.1\,h with 210\,m/s; low and behind, catch up, then raise. \texttt{models/t3/seed42\_L2\_headline.pt}.}"""),
# slide 8
(r"\begin{column}{0.44\textwidth}\small" + "\n" + r"\textbf{Comparator}", r"\begin{column}{0.44\textwidth}\footnotesize" + "\n" + r"\textbf{Comparator}"),
# slide 10
("height=0.72\\textheight,keepaspectratio]{generalist_cells.pdf}", "height=0.64\\textheight,keepaspectratio]{generalist_cells.pdf}"),
("height=0.72\\textheight,keepaspectratio]{traj3d_rendezvous.pdf}", "height=0.64\\textheight,keepaspectratio]{traj3d_rendezvous.pdf}"),
# A1 table
(r"\begin{frame}{A1 \; Observation vector (\nObs\ floats, all roughly in $[-1,1]$)}" + "\n" + r"\scriptsize",
 r"\begin{frame}{A1 \; Observation vector (\nObs\ floats, all roughly in $[-1,1]$)}" + "\n" + r"\tiny"),
# A2
(r"\begin{column}{0.5\textwidth}\small" + "\n" + r"\textbf{Phasing}", r"\begin{column}{0.5\textwidth}\footnotesize" + "\n" + r"\textbf{Phasing}"),
# A5 table
(r"\begin{frame}{A5 \; The seven cells of the generalist mixture}" + "\n" + r"\small", r"\begin{frame}{A5 \; The seven cells of the generalist mixture}" + "\n" + r"\scriptsize"),
(r"TIGHT\_5k1 & 300--800 & 0.05 & \tightR\,/\,\tightV & 3,000 & 0.10 & 86.0\,\% (92.5 at floor)\\", r"TIGHT\_5k1 & 300--800 & 0.05 & \tightR\,/\,\tightV & 3,000 & 0.10 & 86.0\,\%$^\dagger$\\"),
(r"All cells: 3-D, J2 on, bearings-only navigation with real batch acquisition, Discrete(31) head, fuel budget sampled per episode so the policy is fuel-aware.",
 r"$^\dagger$92.5\,\% at the training fuel floor. LONGRANGE is a $\sim$500\,km transfer under a 12,000-sub-step clock. All cells: 3-D, J2 on, bearings-only navigation with real batch acquisition, Discrete(31) head, fuel budget sampled per episode (353--656\,m/s) so the policy is fuel-aware."),
# A4 / A6 figures
(r"\includegraphics[width=\textwidth]{nees.png}\\" + "\n" + r"\includegraphics[width=\textwidth]{nis.png}", r"\includegraphics[width=0.9\textwidth]{nees.png}\\" + "\n" + r"\includegraphics[width=0.9\textwidth]{nis.png}"),
(r"\includegraphics[width=\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=\textwidth]{action_hist.pdf}", r"\includegraphics[width=0.92\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=0.92\textwidth]{action_hist.pdf}"),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print("patched", len(R))
