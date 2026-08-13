# Core-Norm method and derivation

Core-Norm is a feature-wise fitted transform. For every original numeric feature it stores four quantities: the median `m`, lower side scale `s-`, upper side scale `s+`, and transition threshold `tau`.

## Forward map

For a value `x`, define the asymmetric standardized coordinate

\[
u = \begin{cases}
(x-m)/(s^-+\varepsilon), & x<m,\\
(x-m)/(s^++\varepsilon), & x\ge m.
\end{cases}
\]

The fitted threshold is

\[
\tau=\operatorname{clip}(Q_q(|u|),\tau_{min},\tau_{max}).
\]

The central coordinate is

\[
C=\operatorname{clip}(u/\tau,-1,1).
\]

Let

\[
d=\max(|u|-\tau,0), \qquad a=\log(1+d).
\]

The residual coordinate is

\[
R=\operatorname{sign}(u)\frac{a}{1+a}.
\]

Core-Norm returns `[C,R]` for each feature.

## Inverse map

When `R=0`, the point is inside the retained central interval and

\[
u=\tau C.
\]

When `R != 0`, let `e=|R|`. Since `e=a/(1+a)`,

\[
a=\frac{e}{1-e},
\qquad
d=\exp(a)-1.
\]

Therefore

\[
u=\operatorname{sign}(R)\left[\tau+\exp\left(\frac{|R|}{1-|R|}\right)-1\right].
\]

Finally,

\[
x=\begin{cases}
m+u(s^-+\varepsilon), & u<0,\\
m+u(s^++\varepsilon), & u\ge0.
\end{cases}
\]

The `+ epsilon` term must appear in both the forward and inverse maps. The implementation and tests enforce this exact correspondence.

## Property 1 — bounded encoding

By construction, `C` lies in `[-1,1]`. Since `a >= 0`, `a/(1+a)` lies in `[0,1)`, hence `R` lies in `(-1,1)`. Arbitrarily large finite raw values therefore cannot create arbitrarily large encoded coordinates.

## Property 2 — invertibility of valid encodings

The residual mapping is strictly monotone with tail distance and has the closed-form inverse above. Together with the stored side-specific scale and median, the complete valid Core-Norm encoding reconstructs the original finite value up to floating-point error.

## Property 3 — positive affine invariance

For `x' = ax+b` with `a>0`, the median and both quartile scales transform as `m'=am+b` and `s'=as`. The standardized coordinate `u` is therefore unchanged (up to the fixed numerical stabilizer), and so are the Core-Norm coordinates.

## Property 4 — diminishing tail sensitivity

For tail distance `d`, the unsigned residual is

\[
r(d)=\frac{\log(1+d)}{1+\log(1+d)}.
\]

Its derivative is

\[
r'(d)=\frac{1}{(1+d)[1+\log(1+d)]^2}>0,
\]

while `r'(d)` tends to zero as `d` grows. Tail ordering is retained, but each additional unit of extreme magnitude has progressively less leverage in the encoded space.

## Statistical interpretation

Core-Norm should not be described as a universal replacement for every scaler. It is designed for the specific case where numeric extremes may be either contamination or informative events, and where destructive clipping is undesirable. It is also not a distribution-shift correction method.
