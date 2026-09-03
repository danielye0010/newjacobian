# Unified Methodology Framework V2.2

## 1. Inverse-Response Formulation

Geometric compensation is an inverse-response problem: geometry specifies the correction to be achieved, while equilibrium determines the reference-geometry modification required to realize it. Full-response-consistent compensation can be recovered through objective- and task-directed response actions without identifying the complete response operator. Concentrated task-weighted spectral structure can make a few such actions sufficient.

The methodology computes one compensated reference geometry offline before its subsequent physical realization. Define the compensation coefficient vector \(a\in\mathbb R^M\) and the reference-geometry modification by

\[
\delta X=\Phi a,\qquad X_c(a)=X_0+\Phi a.
\]

The derivation requires a fixed reduced compensation basis \(\Phi\); modal coordinates provide the realization used here. The equilibrium displacement \(u(a)\) and surface geometry satisfy

\[
G(u(a),a)=0,\qquad y(a)=S[X_c(a)+u(a)],
\]

where \(S\) is fixed surface interpolation. With fixed positive quadrature weights,

\[
W=\operatorname{blockdiag}_p\!\left(\sqrt{w_p/A_\Gamma}\,I_3\right),
\qquad A_\Gamma=\sum_p w_p,
\]

the norm \(\|Wv\|_2\) is the normalized area-weighted surface RMS. At the uncompensated equilibrium,

\[
\bar A=S\Phi,\qquad A=W\bar A,\qquad
\bar H=\left.\frac{dy}{da}\right|_0,\qquad
H=W\bar H=A+WSu_{,a}|_0.
\]

Thus \(A\) gives the surface change implied directly by the reference-geometry modification, whereas \(H\) gives its local response after equilibrium. The affine surrogate \(y(0)+\bar H a\) supplies the physical map needed by the inverse.

## 2. Physical Response Departure

The first exact reduction separates the physical response within and outside the geometric representation:

\[
\boxed{H=AJ+H_\perp},\qquad J=A^+H,\qquad
H_\perp=(I-AA^+)H,
\]

\[
\boxed{H-A=A(J-I)+H_\perp}.
\]

The Moore-Penrose pseudoinverse uses the weighted output coordinates. The term \(A(J-I)\) describes in-representation gain and coupling; \(H_\perp\) is orthogonal to the retained geometric span. Direct Compensation uses \(A\), Reduced Response uses \(AJ\), and Explicit Full Response uses \(H\).

Direct has full local output equivalence when \(H=A\). For a fixed linearized objective operator \(C\), the weaker condition \(CH=CA\) gives objective-response equivalence with a common residual offset. A physical discrepancy invisible after objective projection therefore leaves that objective's response unchanged. This motivates identifying the response that the compensation objective actually observes.

## 3. Objective-Visible Response

Each surface point retains its nominal association with a finite target feature \(\mathcal F_p\), including its boundaries. Exact closest-point projection defines

\[
\rho(y)=W[y-\Pi_{\mathcal F}(y)],\qquad
\Psi(y)=\tfrac12\|\rho(y)\|_2^2.
\]

Within a verified region where projections remain unique and strictly inside the same planar faces, their unit normals give the constant residual projector

\[
C=\operatorname{blockdiag}_p(n_pn_p^T),\qquad C=C^T=C^2.
\]

Tangential changes within each face leave the distance unchanged. Since \(CW=WC\), the frozen affine surrogates have objective-visible responses

\[
\boxed{T_F=CH,\qquad T_R=CAJ},\qquad
E=T_F-T_R=C(H-AJ)=CH_\perp.
\]

In the common fixed-face region, both residuals share the geometric offset \(b\). Outside it, their quadratic continuation and exact finite-feature reprojection are evaluated separately.

Objective projection can induce a reduced remapping \(J_C=(CA)^+(CH)\) different from \(J\); an objective-aware reduced ablation isolates this explanatory mechanism. The central reduction follows \(T_F\): mechanics contains more response information than the compensation objective sees.

## 4. Inverse-Sufficient Information

For \(T\in\{T_R,T_F\}\), write the fixed-feature quadratic as

\[
f(a)=\tfrac12\|b+Ta\|_2^2
=\tfrac12a^TQa+g^Ta+\tfrac12\|b\|_2^2,
\qquad Q=T^TT,\qquad g=T^Tb.
\]

For a fixed residual offset and admissible compensation set, two operators inducing the same \((Q,g)\) define the same quadratic inverse problem, even if their output response fields differ. Hence

\[
\underbrace{H}_{\text{what mechanics contains}}
\longrightarrow
\underbrace{CH}_{\text{what the objective sees}}
\longrightarrow
\underbrace{(Q,g)}_{\text{what the inverse depends on}}.
\]

Substituting \(T_F=T_R+E\) gives the exact connection to physical response departure:

\[
\boxed{Q_F-Q_R=T_R^TE+E^TT_R+E^TE},\qquad
\boxed{g_F-g_R=E^Tb}.
\]

The inverse effect of the visible defect is therefore measured relative to the Reduced response and the current residual offset.

## 5. Reduced-Relative Full Inverse

The Reduced response model is an available reference inverse model. We ask how much additional Full-response information recovers the Full-response-consistent compensation coefficients. Let \(a_R\) solve the unconstrained Reduced quadratic, \(Q_Ra_R+g_R=0\), and assume \(Q_R,Q_F\) are positive definite on the declared active compensation space. Define

\[
x=Q_R^{1/2}(a-a_R),\qquad
K=Q_R^{-1/2}(Q_F-Q_R)Q_R^{-1/2},
\]

\[
h=Q_R^{-1/2}\nabla f_F(a_R)
=Q_R^{-1/2}\big[(Q_F-Q_R)a_R+(g_F-g_R)\big].
\]

Then

\[
f_F(a_R+Q_R^{-1/2}x)
=f_F(a_R)+h^Tx+\tfrac12x^T(I+K)x,
\]

so Full stationarity reduces to

\[
\boxed{(I+K)x_F=-h},\qquad a_F=a_R+Q_R^{-1/2}x_F.
\]

The Reduced model supplies the reference inverse geometry. Relative to it, the remaining Full mechanics enters through the inverse-relative curvature correction \(K\) and task vector \(h\). For rank-deficient \(Q_R\), the construction applies on its positive active space, with Full-visible Reduced-null directions included explicitly when present.

## 6. Inverse-Visible Response Acquisition

One Full forward/adjoint pair forms the Full quadratic residual \(b+CHa_R\) and its gradient,

\[
\nabla f_F(a_R)=H^TC^T(b+CHa_R).
\]

Combining this exact Full task vector with the available Reduced inverse geometry gives the first inverse-visible correction:

\[
\boxed{x_{\mathrm{IV1}}=-h},\qquad
a_{\mathrm{IV1}}=a_R-Q_R^{-1}\nabla f_F(a_R).
\]

The Full correction equation immediately yields

\[
\boxed{x_{\mathrm{IV1}}-x_F=Kx_F}.
\]

Thus one-interrogation accuracy depends on the relative defect acting on the correction needed by the task. Its spectral structure makes this dependence explicit. For an orthonormal eigendecomposition,

\[
I+K=U\Lambda U^T,\qquad
h=\sum_i\beta_i u_i,\qquad
x_F=-\sum_i\frac{\beta_i}{\lambda_i}u_i,\qquad \lambda_i>0.
\]

Only eigen-directions excited by \(h\) contribute to the Full correction. Refinement minimizes the correction quadratic over progressively richer task-directed spaces satisfying

\[
\mathcal K_m(I+K,h)=\mathcal K_m(K,h)
=\operatorname{span}\{h,Kh,\ldots,K^{m-1}h\}.
\]

If \(h\) has support on only \(s\) distinct eigenvalues of \(I+K\), Krylov refinement from \(x=0\) recovers \(x_F\) in at most \(s\) iterations in exact arithmetic. Few-action difficulty is governed by the task-weighted spectral structure of \((K,h)\), rather than the dimension of the complete response operator.

PCG with Reduced preconditioning realizes this refinement through curvature actions

\[
Q_Fv=H^TC^TCHv.
\]

IV-1 uses one Full forward/adjoint pair; PCG-3 uses three cumulative pairs: initial task-vector acquisition and two curvature refinements. These operations solve for compensation coefficients through task-directed response actions. The resulting reference-geometry modification is applied once, and fresh nonlinear equilibrium with exact finite-feature reprojection evaluates its realization.

## 7. Native FEM Response Actions

At a converged equilibrium, let \(K_T=G_u\), \(B=G_a\), and let \(L\) map free displacement increments to weighted surface coordinates. Differentiating equilibrium gives the forward action; transposing the resulting response gives the adjoint:

\[
K_Ts_v=-Bv,\qquad Hv=Av+Ls_v,
\]

\[
K_T^T\boldsymbol\lambda=L^Tw,\qquad
H^Tw=A^Tw-B^T\boldsymbol\lambda.
\]

Each action uses one tangent solve. The implemented symmetric tangent permits factorization reuse. The design-derivative basis \(B\) is materialized, while \(H\) is evaluated through these actions without assembly. IV-1 uses 2 tangent solves and PCG-3 uses 6, compared with 32 for explicit 32-column Full-response acquisition.

## 8. Method Hierarchy and Applicability

| Method | Role |
|---|---|
| Direct Compensation | Geometric identity-response baseline. |
| Reduced Response | Response-aware reduced model and available reference inverse geometry. |
| Explicit Full Response | Physics-complete local reference endpoint. |
| Inverse-Visible Compensation | Proposed methodology approaching that endpoint through task-directed Full response actions. |
| Objective-Aware Reduced Ablation | Explanatory mechanism. |
| PCG | Numerical refinement mechanism. |

The response formulation is local to a differentiable equilibrium branch with a nonsingular tangent. The inverse identities are exact for the fixed-feature quadratic on the declared active compensation space; agreement with the finite-feature objective holds within the verified projection region. The resulting compensated reference geometry is subsequently evaluated through fresh nonlinear equilibrium and exact finite-feature reprojection.

## 9. Minimal Equation Backbone

\[
H=AJ+H_\perp,\qquad J=A^+H.
\]

\[
T=CH.
\]

\[
Q=T^TT,\qquad g=T^Tb.
\]

\[
K=Q_R^{-1/2}(Q_F-Q_R)Q_R^{-1/2},\qquad (I+K)x_F=-h.
\]

\[
x_{\mathrm{IV1}}=-h,\qquad x_{\mathrm{IV1}}-x_F=Kx_F,\qquad
\mathcal K_m(I+K,h)=\mathcal K_m(K,h).
\]
