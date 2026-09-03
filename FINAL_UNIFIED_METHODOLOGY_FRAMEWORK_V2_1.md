# Unified Methodology Framework V2.1

## 1. Inverse-Response View of Geometric Compensation

Geometric compensation is an inverse-response problem: geometry specifies the desired correction, while equilibrium determines the command that realizes it. The Full-response-consistent inverse can be solved through objective- and task-directed response actions without identifying the complete response operator. When the resulting inverse-visible correction is concentrated, a few such actions can suffice.

Let a fixed basis \(\Phi\) parameterize the commanded reference geometry,

\[
X_c(a)=X_0+\Phi a,\qquad
y(a)=S[X_c(a)+u(a)],\qquad G(u(a),a)=0,
\]

where \(u(a)\) is the equilibrium displacement and \(S\) interpolates nodal coordinates to fixed surface quadrature points. The target, interpolation, and area weights are common to all response models. Define

\[
W=\operatorname{blockdiag}_p\!\left(\sqrt{w_p/A_\Gamma}\,I_3\right),
\qquad A_\Gamma=\sum_p w_p.
\]

The norm \(\|Wv\|_2\) is the normalized area-weighted RMS of a physical surface field \(v\), evaluated over the complete exterior surface. At the uncompensated state,

\[
\bar A=S\Phi,\qquad A=W\bar A,
\qquad
\bar H=\left.\frac{dy}{da}\right|_0,\qquad
H=W\bar H=A+WSu_{,a}|_0.
\]

The geometric action \(A\) describes the commanded surface change; the physical response \(H\) includes its equilibrium-induced modification. Both act on the same command coordinates and use the same physical output metric. The basis specifies a change in reference geometry, and the equilibrium displacement changes in response to that command. The frozen local model \(y(a)\simeq y(0)+\bar H a\) therefore separates the desired geometric correction from the command required to produce it. Understanding that distinction begins with the departure of \(H\) from \(A\).

## 2. Physical Response Departure

The physical response decomposes into a remapping within the geometric representation and a component outside it:

\[
\boxed{H=AJ+H_\perp},\qquad
J=A^+H,\qquad H_\perp=(I-AA^+)H,
\]

\[
\boxed{H-A=A(J-I)+H_\perp}.
\]

Here \(A^+\) is the Moore-Penrose pseudoinverse in the weighted output coordinates. The term \(A(J-I)\) captures in-representation gain and coupling, while \(H_\perp\), with \(A^TH_\perp=0\), captures response outside the geometric span. Diagonal entries of \(J\) describe modal gains; off-diagonal entries describe how one commanded mode contributes to other represented output directions.

Direct uses \(A\) and assumes identity response. Reduced Response uses \(AJ\) to account for the physical remapping within the retained representation. Explicit Full Response uses \(H\) and defines the physics-complete local reference. This hierarchy explains what each response model retains; the compensation objective determines which differences influence the command.

## 3. Objective-Visible Response

The objective measures geometric error through exact closest-point projection onto finite target features. Each surface point retains its nominal macro-feature association \(\mathcal F_p\), including the finite faces and their boundaries. The weighted residual and objective are

\[
\rho(y)=W[y-\Pi_{\mathcal F}(y)],\qquad
\Psi(y)=\tfrac12\|\rho(y)\|_2^2.
\]

Within a verified region where every projection is unique and remains strictly inside the same planar face with unit normal \(n_p\), the residual projector is constant:

\[
C=\operatorname{blockdiag}_p(n_pn_p^T),\qquad C=C^T=C^2.
\]

Within a face interior, tangential changes leave the closest-point distance unchanged, and \(C\) extracts the normal change. This explains how a mechanical response component can disappear from the objective. Scalar weight blocks give \(CW=WC\). The frozen response surrogates then induce exactly affine residuals, with objective-visible responses

\[
T_F=CH,\qquad T_R=CAJ,\qquad
\boxed{E=T_F-T_R=C(H-AJ)=CH_\perp}.
\]

Within a common verified fixed-face region, the Reduced and Full affine surrogates share the same geometric residual offset \(b\) and differ only in their response operators. The offset belongs to that region; evaluation beyond it uses the quadratic continuation, while the finite-feature objective follows exact reprojection.

The desired representable correction remains distinct from its realization. If \(\beta_*\) is a stationary solution of the A-space geometric problem \(\Psi(y(0)+\bar A\beta)\), Direct applies \(a_I=\beta_*\), whereas Reduced Response realizes the same predicted correction through \(Ja_R=\beta_*\), for nonsingular \(J\) and a command domain permitting this reparameterization.

Objective projection can also induce a reduced remapping \(J_C=(CA)^+(CH)\) different from the physical \(J\). Objective-JA isolates this mechanism as an explanatory ablation. The central progression follows \(T_F\): mechanics contains more response information than the compensation objective sees, and the inverse depends on a further reduction of that visible response.

## 4. Inverse-Sufficient Information

For either response \(T\in\{T_R,T_F\}\), the fixed-feature quadratic is

\[
f(a)=\tfrac12\|b+Ta\|_2^2
=\tfrac12a^TQa+g^Ta+\tfrac12\|b\|_2^2,
\qquad Q=T^TT,\qquad g=T^Tb.
\]

With the command domain and null-space convention fixed, \((Q,g)\) determines the inverse. The constant term fixes the absolute objective value. Two response operators producing the same \((Q,g)\) give the same quadratic command problem, even when their output fields differ. Thus the information needed to select a command follows the exact structural reduction

\[
\underbrace{H}_{\text{what mechanics contains}}
\longrightarrow
\underbrace{CH}_{\text{what the objective sees}}
\longrightarrow
\underbrace{(Q,g)}_{\text{what the inverse depends on}}.
\]

The objective-visible defect connects the physical response hierarchy directly to the inverse:

\[
\boxed{Q_F-Q_R=T_R^TE+E^TT_R+E^TE},
\qquad
\boxed{g_F-g_R=E^Tb}.
\]

These identities show how the response difference interacts with both the Reduced model and the residual offset. They motivate expressing the Full inverse in the geometry supplied by the Reduced inverse.

## 5. Reduced-Relative Full Inverse

We take an available Reduced response model as the reference inverse model and ask how much additional Full-response information is required to recover the Full-consistent command. Let \(a_R\) solve the unconstrained Reduced quadratic, \(Q_Ra_R+g_R=0\), and assume \(Q_R\) and \(Q_F\) are positive definite on the declared active command space.

Measure the command correction in Reduced inverse coordinates:

\[
x=Q_R^{1/2}(a-a_R),\qquad
K=Q_R^{-1/2}(Q_F-Q_R)Q_R^{-1/2},
\]

\[
h=Q_R^{-1/2}\nabla f_F(a_R)
=Q_R^{-1/2}\big[(Q_F-Q_R)a_R+(g_F-g_R)\big].
\]

The Full quadratic becomes

\[
f_F(a_R+Q_R^{-1/2}x)
=f_F(a_R)+h^Tx+\tfrac12x^T(I+K)x.
\]

Its stationarity condition is therefore

\[
\boxed{(I+K)x_F=-h},\qquad
a_F=a_R+Q_R^{-1/2}x_F.
\]

The Reduced model supplies the reference inverse geometry. In that geometry, Full mechanics appears only through the relative curvature correction \(K\) and task vector \(h\). The identity term describes the Reduced quadratic curvature, while \(h\) measures the Full stationarity mismatch at \(a_R\). Both the remaining curvature and this mismatch follow from the objective-visible defect \(E\). If \(Q_R\) is rank deficient, the construction applies on its positive active space; Full-visible Reduced-null directions require explicit augmentation.

The Full-consistent command can now be sought by acquiring the actions needed to resolve this relative correction.

## 6. Inverse-Visible Response Acquisition

One Full forward/adjoint pair supplies the task vector: the forward action \(Ha_R\) forms the Full quadratic residual \(b+CHa_R\), and the adjoint gives

\[
\nabla f_F(a_R)=H^TC^T(b+CHa_R).
\]

Retaining the identity curvature in the relative equation while using the exact Full task vector gives the first inverse-visible correction,

\[
\boxed{x_{\mathrm{IV1}}=-h},\qquad
a_{\mathrm{IV1}}=a_R-Q_R^{-1}\nabla f_F(a_R).
\]

The Full correction equation yields the exact error relation

\[
\boxed{x_{\mathrm{IV1}}-x_F=Kx_F}.
\]

One-interrogation accuracy is controlled by the action of the relative defect on the correction required by the task. When this action is small, the Reduced inverse geometry already captures most of the Full correction. The relevant difficulty is therefore the interaction between \(K\) and \(x_F\), rather than the size of the complete mechanical response alone.

Further refinement minimizes the same quadratic over progressively richer task-directed spaces,

\[
\operatorname{span}\{h,Kh,K^2h,\ldots\}.
\]

The task vector generates these spaces, and successive actions reveal the additional directions needed by its correction. PCG with Reduced preconditioning is the numerical realization of this refinement. Its curvature actions use

\[
Q_Fv=H^TC^TCHv,
\]

so Full response is interrogated through directions exposed by the inverse task. These actions support convergence to the Explicit Full endpoint without assembling the complete response operator. IV-1 uses one Full forward/adjoint pair; PCG-3 uses three cumulative pairs, comprising the initial gradient and two PCG updates.

The resulting command is applied once at unit amplitude. Fresh nonlinear equilibrium and exact finite-feature reprojection evaluate its physical realization. The remaining implementation requirement is to obtain the forward and adjoint actions from the FEM equations.

## 7. Native FEM Actions

At a converged equilibrium, let \(K_T=G_u\), \(B=G_a\), and let \(L\) map free displacement increments to weighted surface coordinates. The forward and adjoint actions are

\[
K_Ts_v=-Bv,\qquad Hv=Av+Ls_v,
\]

\[
K_T^T\lambda=L^Tw,\qquad H^Tw=A^Tw-B^T\lambda.
\]

Each action uses one tangent solve and retains both the direct geometric and equilibrium contributions. The adjoint is the transpose of the weighted response map under Euclidean inner products. The validated static, contact-free realization has a symmetric tangent, permitting reuse of one factorization for forward and transpose solves.

The Class-B realization materializes the design-derivative basis \(B\) and evaluates Full response through tangent actions without assembling \(H\). IV-1 uses 2 tangent solves and PCG-3 uses 6, compared with 32 solves for explicit 32-column Full-response acquisition. The remaining \(B\) assembly cost makes the reduction in response solves stronger than the reduction in total wall-clock time.

## 8. Method Hierarchy and Empirical Realization

| Method | Scientific role |
|---|---|
| Direct / Direct-I | Geometric identity-response baseline, using \(A\). |
| Reduced Response / Full-J | Response-aware reduced model \(AJ\) and available reference inverse geometry. |
| Explicit Full Response / Complete-H | Physics-complete local reference endpoint, defined by \(H\). |
| Inverse-Visible Response | Proposed method approaching the same Full-consistent endpoint through task-directed Full forward/adjoint actions. |
| Objective-JA | Explanatory mechanism ablation. |
| PCG | Numerical refinement mechanism for the inverse-visible formulation. |

In the M32 L-bracket, the 95% spectral-energy effective ranks of \(H_\perp\), \(E\), and \(K\) decrease as \(17\to13\to7\), an observed property of this instance. The associated commands give the following realization; linear gap closure is \([f_F(a_R)-f_F(a)]/[f_F(a_R)-f_F(a_F)]\), and nonlinear RMS is the forward finite-feature, area-weighted surface metric.

| Method | Additional Full F/A pairs | Linear gap closure | Nonlinear RMS (mm) |
|---|---:|---:|---:|
| Reduced Response | 0 | 0 | 0.023973586635 |
| Explicit Full Response | Full operator available | 1 | 0.011236157536 |
| IV-1 | 1 | 0.981673331 | 0.011230518468 |
| PCG-3 | 3 | 0.999929997 | 0.011233842072 |

The near-equal nonlinear endpoints demonstrate transfer of the local inverse correction to this nonlinear structural problem. The Explicit Full endpoint is defined by the frozen quadratic, while nonlinear RMS evaluates each command after re-equilibration.

## 9. Scope

The formulation is local to a differentiable equilibrium branch with a nonsingular tangent and is exact for the unregularized fixed-feature quadratic inverse on the declared active command space. Its agreement with the finite-feature objective holds within the verified projection region. Commands are evaluated through fresh nonlinear equilibrium and exact finite-feature reprojection. The present realization uses the prescribed-eigenstrain nonlinear M32 L-bracket and a Class-B native tangent implementation; physical additive-manufacturing validation provides the next external validation layer.

## 10. Minimal Equation Backbone

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
K=Q_R^{-1/2}(Q_F-Q_R)Q_R^{-1/2},
\qquad (I+K)x_F=-h.
\]

\[
x_{\mathrm{IV1}}=-h,\qquad x_{\mathrm{IV1}}-x_F=Kx_F,
\qquad \operatorname{span}\{h,Kh,K^2h,\ldots\}.
\]
