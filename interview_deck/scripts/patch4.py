import sys
p = sys.argv[1]; s = open(p).read()
R = [
(r"\includegraphics[width=0.88\textwidth]{generalist_cells.pdf}", r"\includegraphics[width=0.8\textwidth,height=0.6\textheight,keepaspectratio]{generalist_cells.pdf}"),
(r"\includegraphics[width=0.92\textwidth]{traj3d_rendezvous.pdf}", r"\includegraphics[width=0.92\textwidth,height=0.74\textheight,keepaspectratio]{traj3d_rendezvous.pdf}"),
(r"\begin{column}{0.56\textwidth}" + "\n" + r"\includegraphics[width=\textwidth]{dv_ratio.pdf}", r"\begin{column}{0.56\textwidth}" + "\n" + r"\includegraphics[width=0.96\textwidth]{dv_ratio.pdf}"),
# verification slide smaller
(r"\begin{frame}{Verification is the story: what I caught, and what is still open}" + "\n" + r"\begin{columns}[T]" + "\n" + r"\begin{column}{0.5\textwidth}\small",
 r"\begin{frame}{Verification is the story: what I caught, and what is still open}" + "\n" + r"\begin{columns}[T]" + "\n" + r"\begin{column}{0.5\textwidth}\footnotesize"),
(r"\end{column}" + "\n" + r"\begin{column}{0.5\textwidth}\small" + "\n" + r"\textbf{Still open}", r"\end{column}" + "\n" + r"\begin{column}{0.5\textwidth}\footnotesize" + "\n" + r"\textbf{Still open}"),
(r"\includegraphics[width=0.95\textwidth]{traj_closing.pdf}", r"\includegraphics[width=0.85\textwidth]{traj_closing.pdf}"),
(r"\includegraphics[width=0.9\textwidth]{nees.png}\\" + "\n" + r"\includegraphics[width=0.9\textwidth]{nis.png}", r"\includegraphics[width=0.84\textwidth]{nees.png}\\" + "\n" + r"\includegraphics[width=0.84\textwidth]{nis.png}"),
(r"\includegraphics[width=0.8\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=0.8\textwidth]{action_hist.pdf}", r"\includegraphics[width=0.7\textwidth]{cport_timing.pdf}" + "\n" + r"\includegraphics[width=0.7\textwidth]{action_hist.pdf}"),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print("patched", len(R))
