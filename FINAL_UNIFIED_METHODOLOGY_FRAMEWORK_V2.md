# Unified Methodology Framework V2

## 1. Central Thesis

**Geometric compensation is an inverse-response problem: geometry specifies the correction sought, equilibrium determines the command required to realize it, and the inverse ultimately needs only a task-visible fraction of the full mechanical response.**

Geometric representation → physical equilibrium response → response departure → objective-visible response → inverse-sufficient information → inverse-visible Full-response acquisition → Full-consistent one-shot compensation → native FEM realization.

The first question is therefore how a geometric command differs from the output it produces after equilibrium.

## 2. Geometric Representation and Physical Response

Let a fixed basis \(\Phi\in\mathbb R^{3n\times M}\) parameterize the commanded reference geometry,

\[
X_c(a)=X_0+\Phi a,\qquad
y(a)=S[X_c(a)+u(a)],\qquad G(u(a),a)=0,
\]

where \(u(a)\) is the equilibrium displacement and \(S\) interpolates nodal coordinates to \(N_\Gamma\) fixed surface quadrature points. The target, interpolation, and quadrature weights are fixed. Define

\[
W=\operatorname{blockdiag}_{p}\!\left(\sqrt{w_p/A_\Gamma}\,I_3\right),
\qquad A_\Gamma=\sum_p w_p,\qquad \mathcal S=WS.
\]

Thus \(\|Wv\|_2\) is the normalized area-weighted RMS of a physical surface field \(v\). The complete exterior surface, including the clamped face, is retained.

At the uncompensated state, the geometric action and physical response are

\[
\bar A=S\Phi,\qquad A=W\bar A,
\qquad
\bar H=\left.\frac{dy}{da}\right|_0,\qquad
H=W\bar H=A+\mathcal S u_{,a}|_0.
\]

Both \(A,H\in\mathbb R^{3N_\Gamma\times M}\) map command coordinates to weighted surface changes. Equivalently, \(H=\partial_a r|_0\) for the fixed-correspondence physical residual \(r(a)=W[y(a)-y_t]\), with \(y_t=SX_t\). This residual defines the response metric; the compensation objective is introduced in Section 4.

Geometry determines which corrections can be commanded. Equilibrium determines what those commands do. The local model \(y(a)\simeq y(0)+\bar H a\) therefore supplies information that the geometric representation alone cannot provide. The next question is which parts of that information Direct compensation omits.

## 3. Response Departure and the Reduced/Full Hierarchy

Projecting the physical response onto the geometric action space gives

\[
J=A^+H,\qquad P_A=AA^+,\qquad H_\perp=(I-P_A)H,
\]

and the exact identities

\[
\boxed{H=AJ+H_\perp},\qquad
\boxed{H-A=A(J-I)+H_\perp}.
\]

Here \(J\in\mathbb R^{M\times M}\) is dimensionless. Its diagonal and off-diagonal entries describe in-representation gains and coupling; \(H_\perp\) describes physical response outside the retained geometric span. Orthogonality, \(A^T H_\perp=0\), is with respect to the already weighted output coordinates. The pseudoinverse is Moore–Penrose on the declared retained space; the numerical response convention uses a relative singular-value cutoff of \(10^{-10}\).

This decomposition defines a hierarchy of physical models:

- **Direct:** \(A\), the identity-response assumption.
- **Reduced Response:** \(AJ\), the response-aware remapping within the geometric representation.
- **Explicit Full Response:** \(H\), the complete local physical response and gold-standard reference.

Direct assumes that the desired geometric action passes through equilibrium unchanged. Reduced Response corrects the in-representation remapping. Full Response also retains the out-of-representation response. These are levels of response fidelity; \(\operatorname{range}(AJ)\) and \(\operatorname{range}(H)\) need not be nested. If \(J\) is nonsingular, \(\operatorname{range}(AJ)=\operatorname{range}(A)\).

For any command \(v\),

\[
\|(H-A)v\|_2^2
=\|A(J-I)v\|_2^2+\|H_\perp v\|_2^2.
\]

The coordinate-invariant departure measure \(\epsilon_D=\sup_{v\ne0}\|(H-A)v\|_2/\|Av\|_2\), restricted to the active geometric space, quantifies the identity-response mismatch. It does not by itself determine compensation performance: the objective may see only part of that mismatch.

## 4. Objective-Visible Response

The geometric objective measures distance to finite target features. Each surface point has a fixed nominal macro-feature association \(\mathcal F_p\), represented by its closed finite face union and boundaries. With exact closest-point projection,

\[
\rho(y)=W[y-\Pi_{\mathcal F}(y)],\qquad
\Psi(y)=\tfrac12\|\rho(y)\|_2^2.
\]

This is distinct from material point-to-point error. Projection classes and face, edge, or vertex identities can change as the geometry changes.

Inside a verified region in which every projection is unique and remains strictly inside the same planar face with unit normal \(n_p\), let

\[
C=\operatorname{blockdiag}_p(n_pn_p^T).
\]

Then \(C=C^T=C^2\) is constant. Since each weight block is scalar, \(CW=WC\). For a frozen affine response surrogate, the residual is exactly affine within this region. Consequently,

\[
T_F=CH,\qquad T_R=CAJ,\qquad
\boxed{E=T_F-T_R=C(H-AJ)=CH_\perp}.
\]

The common residual offset must be recovered from a valid region anchor: if \(\rho_*\) is the residual at command \(a_*\) for that surrogate, set \(b=\rho_*-Ta_*\). A common face assignment and baseline geometry give the same \(b\) for Reduced and Full models. It need not equal the exact baseline residual \(\rho(y(0))\), whose active features may differ.

The finite-feature construction also preserves the meaning of the desired geometric correction. If \(\beta_*\) is a stationary A-space solution of \(\Psi(y(0)+\bar A\beta)\), Direct applies \(a_I=\beta_*\); Reduced Response realizes the same predicted correction through \(Ja_R=\beta_*\), when \(J\) is nonsingular and the command domain permits that reparameterization.

As a supporting mechanism, \(J_C=(CA)^+(CH)\) may differ from \(J\): objective projection can remap response that was physically orthogonal to \(A\) back into \(\operatorname{range}(CA)\). Objective-JA isolates this effect as an ablation. The main reduction proceeds through \(T_F\) itself: mechanics contains more response information than the geometric objective can see, and even that visible operator may contain more than the inverse needs.

## 5. Inverse-Sufficient Information

For the fixed-quadratic problem, write

\[
f_F(a)=\tfrac12\|b+T_Fa\|_2^2,
\qquad
f_R(a)=\tfrac12\|b+T_Ra\|_2^2.
\]

For either response \(T\), define

\[
Q=T^TT,\qquad g=T^Tb,
\qquad
f(a)=\tfrac12a^TQa+g^Ta+\tfrac12\|b\|_2^2.
\]

The inverse is completely determined by \((Q,g)\), together with a fixed command domain and any declared null-space selection. The constant term affects absolute objective values but not the optimizer. This yields the central information hierarchy:

\[
\underbrace{H}_{\text{what mechanics contains}}
\ \longrightarrow\
\underbrace{CH}_{\text{what the objective sees}}
\ \longrightarrow\
\underbrace{(Q,g)}_{\text{what the inverse needs}}.
\]

The response departure connects exactly to these inverse quantities:

\[
Q_F-Q_R=T_R^TE+E^TT_R+E^TE,
\qquad
g_F-g_R=E^Tb.
\]

Thus the inverse effect depends on how the visible defect interacts with both the Reduced response and the task residual. The hierarchy is an information reduction, not a theorem that numerical or effective ranks must decrease. It now becomes natural to express the Full inverse relative to the Reduced inverse already available.

## 6. Reduced-Relative Full Inverse

Let \(a_R\) solve the unconstrained Reduced quadratic, so \(Q_Ra_R+g_R=0\). First assume \(Q_R\succ0\) and \(Q_F\succ0\). The Reduced model supplies both a response-aware command and a metric for measuring its correction:

\[
x=Q_R^{1/2}(a-a_R),\qquad
P=Q_R^{-1/2}Q_FQ_R^{-1/2}=I+K,
\]

\[
h=Q_R^{-1/2}\nabla f_F(a_R)
=Q_R^{-1/2}\big[(Q_F-Q_R)a_R+(g_F-g_R)\big].
\]

In these coordinates,

\[
f_F(a_R+Q_R^{-1/2}x)
=f_F(a_R)+h^Tx+\tfrac12x^T(I+K)x.
\]

The Full stationarity condition therefore reduces exactly to

\[
\boxed{(I+K)x_F=-h},\qquad
a_F=a_R+Q_R^{-1/2}x_F.
\]

Once the Full inverse is expressed in the geometry supplied by the Reduced model, the remaining mechanics appears only as a relative inverse-visible curvature defect \(K\) and a task vector \(h\). The large response operator has disappeared from the correction equation.

**Rank qualification.** Full rank is verified for the M32 instance, not assumed universally. If \(Q_R=U\Lambda U^T\) on its positive active space, use \(a=a_R+U\Lambda^{-1/2}x\), \(P=\Lambda^{-1/2}U^TQ_FU\Lambda^{-1/2}\), and \(h=\Lambda^{-1/2}U^T\nabla f_F(a_R)\). This solves the Full quadratic restricted to that affine space. It represents an unrestricted Full optimum if the excluded Reduced-null directions are also Full-null, \(T_F\ker Q_R=\{0\}\), with a fixed null-coordinate convention. Otherwise Full can activate those directions and their coupled equations must be included; simply replacing inverses by pseudoinverses loses that information. PCG further requires positive Full curvature on the solve space. Numerical rank thresholds must be declared consistently with the retained model.

The remaining question is how much Full information is needed to resolve this relative correction for the present task.

## 7. Inverse-Visible Response Acquisition

Given the Reduced inverse and the common quadratic objective, the first Full interrogation acquires the task vector. It uses one forward action and one adjoint action:

\[
Ha_R\ \longrightarrow\ b+CHa_R
\ \longrightarrow\ H^TC^T(b+CHa_R)=\nabla f_F(a_R).
\]

With no additional Full curvature information, the Reduced-relative formulation gives

\[
\boxed{x_{\mathrm{IV1}}=-h},\qquad
a_{\mathrm{IV1}}=a_R-Q_R^{-1}\nabla f_F(a_R).
\]

IV-1 is the minimum-information Full-physics correction within this formulation: it uses the exact Full task vector and the available Reduced inverse geometry. This description makes no information-theoretic optimality claim over all possible acquisition methods.

The exact Full equation immediately gives

\[
\boxed{x_{\mathrm{IV1}}-x_F=Kx_F}.
\]

One-interrogation accuracy is therefore controlled by the action of the relative defect on the required correction. With \(\|z\|_P^2=z^TPz\),

\[
f_F(a_{\mathrm{IV1}})-f_F(a_F)=\tfrac12\|Kx_F\|_P^2,
\qquad
\frac{f_F(a_{\mathrm{IV1}})-f_F(a_F)}{f_F(a_R)-f_F(a_F)}
\le\|K\|_2^2
\]

when the denominator is positive. The bound follows because symmetric \(K\) commutes with \(P=I+K\). A small task-directed defect can make IV-1 accurate even without exact low rank; a large response dimension alone does not make the inverse difficult.

When more information is needed, minimize the same correction quadratic over progressively richer spaces

\[
\mathcal V_k=\operatorname{span}\{h,Kh,\ldots,K^{k-1}h\}
=\operatorname{span}\{h,Ph,\ldots,P^{k-1}h\}.
\]

CG in whitened coordinates, equivalently PCG with Reduced preconditioning in command coordinates, realizes this refinement. Each new curvature action is obtained as

\[
Q_Fv=H^TC^TCHv,
\]

using one Full forward/adjoint pair. Neither \(H\), \(Q_F\), nor \(K\) needs to be assembled. The methodological choice is to acquire the Full-response actions exposed by the current inverse task; PCG is the numerical realization.

The initial gradient costs one pair before any PCG update. Thus **PCG-3 means three cumulative pairs: initial gradient acquisition plus two PCG updates**. IV-1 and this PCG sequence use the same initial task information; PCG-3 is not three extra iterations applied after IV-1. Any additional Full action used for a stopping certificate must also be counted.

After this offline inverse calculation, form \(X_c(a)\), check reference-geometry admissibility, and apply the command once at unit amplitude. Exact finite-feature projection and physical admissibility are assessed in the separate nonlinear equilibrium evaluation. Any extra Full action for a surrogate-region check also lies beyond the stated inverse-acquisition budget and must be counted. Internal Krylov iterations do not constitute additional physical compensation cycles. Full consistency means convergence to the Explicit Full quadratic endpoint; finite action budgets approximate that endpoint, and nonlinear equilibrium tests its physical transfer. The required actions must therefore be realizable directly from the mechanical tangent system.

## 8. Native FEM Realization

At a converged equilibrium, let \(K_T=G_u\), \(B=G_a\), and let \(L\) map free equilibrium displacement increments to the weighted surface coordinates, including the fixed DOF recovery. Then

\[
K_Ts_v=-Bv,\qquad Hv=Av+Ls_v,
\]

\[
K_T^T\lambda=L^Tw,\qquad H^Tw=A^Tw-B^T\lambda.
\]

These actions retain both the direct geometric contribution and the equilibrium contribution. Each uses one tangent solve; \(H^T\) is the adjoint of the weighted response under Euclidean inner products. The validated static, mechanical, contact-free implementation uses a symmetric tangent and reuses one factorization for forward and transpose solves.

The **Class-B realization is partially matrix-free**: it materializes the full 32-column design-derivative basis \(B\), but does not materialize the Full response matrix or compute its 32 sensitivity columns. IV-1 uses **2 tangent solves**; PCG-3 uses **6**. The Reduced response and fixed-face objective are supplied inputs, so these counts describe the additional Full information, not the independent cost of constructing the Reduced model.

Relative to 32-column Full acquisition, IV-1 uses **16× fewer tangent solves**. Charging all 32 design-derivative assemblies, the reported first-use incremental speedup is approximately **1.50×**, and the end-to-end planning speedup including the baseline equilibrium is approximately **1.11×**. These wall-clock ratios combine historical SPOOLES timings with new SuperLU action timings and exclude common, cacheable initialization; they are a mixed-solver audit, not a matched-solver or complete manufacturing-cycle benchmark. The established computational gain is the reduction in Full information and tangent actions.

Direct design JVP/VJP actions could remove the remaining \(B\)-materialization cost, but that extension is outside the validated method. The implementation therefore preserves the same method hierarchy as the mathematical formulation.

## 9. Final Method Hierarchy

| Method | Response and role |
|---|---|
| Direct / Direct-I | \(A\): geometric identity-response baseline; applies the desired representable correction as the command. |
| Reduced Response / historical Full-J | \(AJ\): response-aware reduced model and the reference inverse geometry for the Full problem. |
| Explicit Full Response / historical Complete-H | \(H\): physics-complete local reference defining the Full-consistent endpoint when the operator is explicitly available. |
| Inverse-Visible Response | Proposed information-efficient method approaching that same endpoint through task-directed Full forward and adjoint actions; IV-1 is its first correction and few-action refinement increases fidelity. |
| Objective-JA | Mechanism ablation explaining how objective projection changes the reduced remapping. |
| PCG | Numerical refinement mechanism for the inverse-visible equation. |

The progression concerns what information is used to choose one command. Its general algebra and its demonstrated computational behavior require different kinds of evidence.

## 10. Theory-to-Evidence Map

| Theoretical object | What it asks us to inspect | L-bracket evidence |
|---|---|---|
| \(H=AJ+H_\perp\), then \(E=CH_\perp\) | Which physical differences survive the objective? | Common fixed-face closures and the objective-visible bridge verify the projection structure while identifying feature transitions at other endpoints. |
| \((Q,g)\), then \(P=I+K\) | Is the remaining inverse correction concentrated and supported by the Reduced space? | In M32, the 95% spectral-energy ranks of \(H_\perp,E,K\) are \(17\to13\to7\); all three numerical ranks remain 32. \(Q_R\) has rank 32 and \(\|K\|_2\approx0.15985\). |
| \(x_{\mathrm{IV1}}-x_F=Kx_F\), then action refinement | Can a small action budget recover the Full inverse? | The one-pair and three-pair quadratic closures below establish task-specific recovery. |
| Tangent forward/adjoint actions, then one-shot equilibrium | Does action acquisition reproduce the reference and transfer to nonlinear mechanics? | Relative native action errors are approximately \(10^{-12}\); fresh nonlinear evaluations reproduce the candidate RMS values below. |

For clarity, linear gap closure is \([f_F(a_R)-f_F(a)]/[f_F(a_R)-f_F(a_F)]\). The nonlinear metric is the **forward finite-feature, area-weighted surface RMS**, not symmetric RMS or material point-to-point RMS.

| Method | Additional Full F/A pairs | Linear objective-gap closure | Nonlinear RMS (mm) |
|---|---:|---:|---:|
| Reduced Response | 0 | 0 | 0.023973586635 |
| Explicit Full Response reference | Full operator available | 1 | 0.011236157536 |
| IV-1 | 1 | 0.981673331 | 0.011230518468 |
| PCG-3 | 3 | 0.999929997 | 0.011233842072 |

The favorable effective-rank sequence is an observation about this instance, not a universal compression law. The near-equal nonlinear RMS values demonstrate transfer of the inverse-visible command to the studied nonlinear problem. Their small differences do not establish general nonlinear superiority: Explicit Full minimizes the frozen quadratic, not the nonlinear equilibrium objective.

Evidence: [objective-visible bridge](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/objective_visible_response_bridge_audit/OBJECTIVE_VISIBLE_RESPONSE_BRIDGE_AUDIT.md>), [fixed-face closure](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/response_jacobian_fixed_face_closure_audit/FIXED_FACE_QUADRATIC_CLOSURE_AUDIT.md>), [inverse structure](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/EXP11_inverse_visible/11A_structure_audit/EXP11A_STRUCTURE_REPORT.md>), [counted actions](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/EXP11_inverse_visible/11B_oracle_efficiency/EXP11B_ORACLE_REPORT.md>), [nonlinear closure](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/EXP11_inverse_visible/11C_nonlinear_closure/EXP11C_NONLINEAR_REPORT.md>), [native closure](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/EXP12_native_inverse_visible/EXP12_NATIVE_CLOSURE_REPORT.md>), and [authoritative nonlinear endpoints](<C:/Users/31746/OneDrive - Iowa State University/newjacobian/native_M32C_R2_end_to_end/NATIVE_M32C_R2_END_TO_END_COMPENSATION_VALIDATION.md>).

These results establish a concrete realization within the following boundaries.

## 11. Scope

The response is local to a differentiable equilibrium branch with a nonsingular tangent. The exact inverse theory concerns an unregularized fixed quadratic, with positive curvature on the solve space. Its equivalence to the finite-feature objective requires the verified unchanged projection region; outside that region it is a quadratic continuation, and exact reprojection must be evaluated separately. In particular, the Reduced own-model optimum and Full optimum share the verified face assignment, while the Reduced command evaluated through the Full response surrogate can leave it. No global affine observation, global nonlinear optimum, or unrestricted active-set convergence is asserted.

The demonstrated realization is the prescribed-eigenstrain, nonlinear M32 L-bracket with its fixed basis, metric, and admissibility checks. Transfer to other response regimes and physical additive-manufacturing validation remain external validation layers. Native acquisition still materializes \(B\) and takes the Reduced inverse as available.

Within this scope, the methodology is one continuous reduction from mechanical response to the information required by the compensation inverse.

## 12. Minimal Equation Backbone

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
(I+K)x=-h.
\]

\[
x_{\mathrm{IV1}}=-h,\qquad x_{\mathrm{IV1}}-x_F=Kx_F.
\]

### Notation reconciliation

Bars denote unweighted physical surface maps; \(A,H\) include \(W\), and \(\mathcal S=WS\) preserves the original weighting convention. \(N_\Gamma\) denotes the surface-point count so \(P\) can denote Reduced-relative Full curvature. \(Q_R,Q_F\) replace the normal matrices called \(H_R,H_F\) in the inverse-visible reports; \(K\) is reserved for the relative inverse defect, \(K_T\) for the FEM tangent, and \(B\) for the equilibrium design derivative. The surface-dimensional \(b\) is the affine objective offset; \(\beta_*\) is the command-dimensional desired geometric correction. Neither is a replacement for \(\rho(y)\).

Under the retained volume-only modal normalization, \(a,\beta_*\) have units \(\mathrm{mm}^{5/2}\), and \(A,H,T\) have units \(\mathrm{mm}^{-3/2}\); their products are physical lengths. Thus \(Q\) has units \(\mathrm{mm}^{-3}\), \(g\) has units \(\mathrm{mm}^{-1/2}\), \(b,x,h\) have units mm, and \(J,C,P,K\) are dimensionless. \(Q\in\mathbb R^{M\times M}\), \(g,x,h\in\mathbb R^M\), and \(b\in\mathbb R^{3N_\Gamma}\), with the active-space dimensions replacing \(M\) where required.

### Source reconciliation issue

The repository index names a corrected manuscript, but that file is absent from the available workspace; the named `11420_Pure_Relative_Matrix_App.pdf` was also unavailable and could not be identified by exact-title web search. Consequently, direct reconciliation against those two documents remains pending. The framework follows the available canonical response ledger, final storyline, fixed-face and objective-visible reports, native nonlinear endpoints, and inverse-visible results; the presentation principles supplied in the task were applied without importing matrix-approximation theory.

One source-level scope ambiguity is explicit here: the inverse-visible reports call \(f_F\) the exact common objective, whereas the objective-visible bridge records feature transitions at the Full-response evaluation of \(a_R\). Sections 4 and 11 distinguish the exact fixed-face quadratic continuation used for the inverse identities and gap closures from the exact finite-feature objective outside that region. The reported commands and numerical closure values are retained.
