# Theory: two-site DMRG, derived step by step

This document derives the algorithm implemented in `src/core/` from the ground
up. It is the shared-notation reference for the package: the
[implementation notes](implementation.md) and future feature documents build on
the definitions established here.

Every formula is tied to the code that realizes it, using the exact stored
index orders:

| Object | Stored as | Index order |
| --- | --- | --- |
| MPS site tensor | `Array{ComplexF64,3}` | `A[left, physical, right]` |
| MPO site tensor | `Array{ComplexF64,4}` | `W[left, physical_out, physical_in, right]` |
| Environment | `Array{ComplexF64,3}` | `[bra bond, MPO bond, ket bond]` |

Throughout, `N` is the number of sites, `d` the physical dimension (`d = 2`
for spin-1/2), and an overbar $\overline{(\cdot)}$ denotes complex conjugation.
Repeated indices inside a sum are contracted.

---

## 1. Matrix-product states

### 1.1 Definition

A matrix-product state (MPS) stores the $d^N$ amplitudes of a state as a chain
of small tensors. Write the site tensor at site $i$ as a set of $d$ matrices,
one per physical value $s$,

```math
\bigl(A^{(i)}_{s}\bigr)_{l,r} \;=\; A^{(i)}[l, s, r],
\qquad l = 1,\dots,D_{i-1},\quad r = 1,\dots,D_i ,
```

where $D_i$ is the bond dimension on the link between sites $i$ and $i+1$. The
state is

```math
|\psi\rangle \;=\; \sum_{s_1,\dots,s_N}
      A^{(1)}_{s_1}\, A^{(2)}_{s_2}\cdots A^{(N)}_{s_N}\;
      |s_1 s_2 \dots s_N\rangle .
```

Open boundaries mean $D_0 = D_N = 1$, so the matrix product
$A^{(1)}_{s_1}\cdots A^{(N)}_{s_N}$ is $1\times D_1$ times $\cdots$ times
$D_{N-1}\times 1$, i.e. a scalar amplitude

```math
c_{s_1\dots s_N} \;=\; A^{(1)}_{s_1}\, A^{(2)}_{s_2}\cdots A^{(N)}_{s_N}.
```

The constructor `MPS` (`src/core/MPS.jl`) enforces exactly these invariants:
`size(first,1) == 1`, `size(last,3) == 1`, a single shared `d`, and matching
neighbour bonds `size(tensors[i],3) == size(tensors[i+1],1)`.

### 1.2 The dense state vector

`dense(psi)` materializes the coefficient tensor by contracting the shared
bonds left to right. After absorbing site $i$ into a running block `state`,

```math
\text{joined}[l, s_1, s_2, r] \;=\; \sum_{m} \text{state}[l, s_1, m]\; A^{(i)}[m, s_2, r],
```

and reshaping merges $(s_1,s_2)$ into one physical leg. The final length-$d^N$
vector is $c_{s_1\dots s_N}$ flattened. This is exponential in $N$ and is used
only for small-system validation.

### 1.3 Inner products and norm

The overlap $\langle\phi|\psi\rangle$ contracts the two chains one site at a
time. Starting from the $1\times1$ scalar $E^{(0)} = 1$, `overlap`
(`src/core/MPS.jl`) applies

```math
E^{(i)}[r_a, r_b] \;=\; \sum_{l_a,l_b,s}
      \overline{A^{(i)}[l_a, s, r_a]}\; E^{(i-1)}[l_a, l_b]\; B^{(i)}[l_b, s, r_b],
```

where $A$ are the bra tensors and $B$ the ket tensors. Because the boundaries
are one-dimensional, $E^{(N)}$ is the scalar $\langle\phi|\psi\rangle$. The norm
follows from the same contraction with $\phi = \psi$,

```math
\lVert\psi\rVert \;=\; \sqrt{\max\!\bigl(0,\ \operatorname{Re}\langle\psi|\psi\rangle\bigr)},
```

and `normalize!` divides the first tensor by $\lVert\psi\rVert$ (rescaling one
site rescales the whole product).

---

## 2. Matrix-product operators

### 2.1 Definition

A matrix-product operator (MPO) stores an operator as a chain of rank-4
tensors. With the operator-valued matrices
$\bigl(W^{(i)}_{s' s}\bigr)_{a,b} = W^{(i)}[a, s', s, b]$ (output index $s'$ on
the bra, input index $s$ on the ket),

```math
H \;=\; \sum_{\{s'\},\{s\}}
      W^{(1)}_{s'_1 s_1}\cdots W^{(N)}_{s'_N s_N}\;
      |s'_1\dots s'_N\rangle\langle s_1 \dots s_N| .
```

The `MPO` constructor (`src/core/MPO.jl`) enforces unit boundary MPO bonds and
matching neighbour bonds, mirroring the MPS. `dense(H)` contracts the chain into
the $d^N \times d^N$ matrix $\langle s'_1\dots s'_N| H | s_1 \dots s_N\rangle$;
note it permutes the merged legs to `(l, o_1, o_2, i_1, i_2, r)` so that the row
index collects all outputs and the column index all inputs.

### 2.2 Expectation values

Combining §1.3 and §2.1, the (unnormalized) expectation value
$\langle\psi|H|\psi\rangle$ is a three-layer contraction — bra MPS, MPO, ket MPS
— evaluated site by site by `overlap_with_operator` (`src/core/dmrg.jl`):

```math
\Xi^{(i)}[r_a, b, r_k] \;=\; \sum_{l_a,\,a,\,l_k,\,s',\,s}
      \overline{A^{(i)}[l_a, s', r_a]}\; \Xi^{(i-1)}[l_a, a, l_k]\;
      W^{(i)}[a, s', s, b]\; A^{(i)}[l_k, s, r_k],
```

starting from $\Xi^{(0)} = 1$; the scalar $\Xi^{(N)} = \langle\psi|H|\psi\rangle$.

The normalized variational energy returned by `compute_energy(H, psi)` is

```math
E[\psi] \;=\; \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}.
```

The code warns if the imaginary part is not negligible — for a Hermitian $H$ and
a genuine state it must vanish up to rounding.

### 2.3 Worked example: the open Heisenberg chain

`heisenberg_mpo(N; J, hz)` builds the spin-1/2 Hamiltonian

```math
H \;=\; J\sum_{n=1}^{N-1}
      \bigl(S^x_n S^x_{n+1} + S^y_n S^y_{n+1} + S^z_n S^z_{n+1}\bigr)
      \;+\; h_z\sum_{n=1}^{N} S^z_n
```

as a bond-dimension-5 MPO. The construction is a finite-state machine: MPO bond
value $5$ means "no term started yet", value $1$ means "term completed", and
values $2,3,4$ carry a half-finished two-site term. The bulk tensor $W$ has the
nonzero operator-valued entries

```math
\begin{aligned}
W_{5,5} &= \mathbb{1}, & W_{1,1} &= \mathbb{1}, & W_{5,1} &= h_z\, S^z, \\
W_{5,2} &= \tfrac{J}{2} S^+, & W_{5,3} &= \tfrac{J}{2} S^-, & W_{5,4} &= J\, S^z, \\
W_{2,1} &= S^-, & W_{3,1} &= S^+, & W_{4,1} &= S^z, &
\end{aligned}
```

with $S^z = \operatorname{diag}(\tfrac12,-\tfrac12)$ and $S^\pm$ the raising and
lowering operators. The left boundary selects the start row ($a=5$) and the
right boundary the completed column ($b=1$). A path from start to finish either
takes the on-site step $5\!\to\!1$ (contributing $h_z S^z$) or a two-step path
$5\!\to\!k\!\to\!1$ that lays down a bond term. Summing the two-step paths,

```math
\tfrac{J}{2}\bigl(S^+_n S^-_{n+1} + S^-_n S^+_{n+1}\bigr) + J\,S^z_n S^z_{n+1}
      \;=\; J\bigl(S^x_n S^x_{n+1} + S^y_n S^y_{n+1} + S^z_n S^z_{n+1}\bigr),
```

using $S^x S^x + S^y S^y = \tfrac12(S^+S^- + S^-S^+)$. This reproduces $H$ exactly.

### 2.4 General nearest-neighbor MPOs from local terms

The finite-state machine of §2.3 is not special to the Heisenberg chain — it is a
recipe for *any* translation-invariant Hamiltonian built from on-site and
nearest-neighbor terms,

```math
H \;=\; \sum_{i}\ \sum_{(c,\hat O)\,\in\,\text{onsite}} c\,\hat O_i
    \;+\; \sum_{i}\ \sum_{(c,\hat L,\hat R)\,\in\,\text{bond}} c\,\hat L_i \hat R_{i+1}.
```

Label the automaton states so that state $w$ means "no term started", state $1$
means "term completed", and states $2,\dots,K+1$ are one intermediate channel per
bond term, where $K = |\text{bond}|$. The MPO bond dimension is then $w = K + 2$,
and the bulk tensor has the operator-valued entries

```math
\begin{aligned}
W_{w,w} &= \mathbb{1}, & W_{1,1} &= \mathbb{1}, &
W_{w,1} &= \sum_{(c,\hat O)} c\,\hat O, \\
W_{w,\,1+k} &= c_k\,\hat L_k, & W_{1+k,\,1} &= \hat R_k, & &
\end{aligned}
```

for the $k$-th bond term $(c_k, \hat L_k, \hat R_k)$. The left boundary selects
the start row $w$ and the right boundary the completed column $1$, exactly as in
§2.3. A path $w \to 1$ deposits an on-site term; a path $w \to 1+k \to 1$
deposits $c_k\,\hat L_k \hat R_{k+1}$ on one bond; and $W_{w,w}$/$W_{1,1}$ carry
the identity before and after.

`nearest_neighbor_mpo(N, d; onsite, bond)` (`src/core/mpo_builder.jl`) implements
this directly, and `tfim_mpo` is the transverse-field Ising special case
$\text{onsite} = \{(-h, S^x)\}$, $\text{bond} = \{(-J, S^z, S^z)\}$. The tests
confirm that feeding the Heisenberg terms
$\{(\tfrac{J}{2}, S^+, S^-), (\tfrac{J}{2}, S^-, S^+), (J, S^z, S^z)\}$ into the
builder reproduces `heisenberg_mpo` to machine precision.

---

## 3. Canonical forms

Canonical (gauge-fixed) tensors are what make the two-site update a *variational*
problem. There is gauge freedom in an MPS: inserting $G G^{-1}$ on any bond
leaves $|\psi\rangle$ unchanged. We fix it so that blocks of tensors act as
isometries.

### 3.1 Left-canonical tensors

Reshape a site tensor into the matrix $M_{(l s),\, r} = A[l, s, r]$. The tensor
is **left-canonical** when its columns are orthonormal,

```math
\sum_{l, s} \overline{A[l, s, r']}\; A[l, s, r] \;=\; \delta_{r r'}
\qquad\Longleftrightarrow\qquad
M^\dagger M = \mathbb{1}.
```

`left_canonicalize!` (`src/core/MPS.jl`) obtains this with a thin QR
factorization $M = Q R$: $Q$ has orthonormal columns, so it replaces $A$, while
$R$ is pushed into the next site,

```math
A^{(i)} \leftarrow Q, \qquad A^{(i+1)} \leftarrow R\, A^{(i+1)} ,
```

which leaves the product $A^{(i)} A^{(i+1)}$ — and therefore $|\psi\rangle$ —
invariant.

### 3.2 Right-canonical tensors

Symmetrically, reshape $M'_{l,\, (s r)} = A[l, s, r]$. The tensor is
**right-canonical** when its rows are orthonormal,

```math
\sum_{s, r} A[l, s, r]\; \overline{A[l', s, r]} \;=\; \delta_{l l'}
\qquad\Longleftrightarrow\qquad
M' M'^\dagger = \mathbb{1}.
```

`right_canonicalize!` achieves this by QR-factorizing the adjoint
$M'^\dagger = Q R$, storing $Q^\dagger$ at the current site and pushing $R^\dagger$
into the previous site. `dmrg!` starts every run from a normalized,
fully right-canonical state so the first left-to-right sweep already has a clean
orthogonality structure.

### 3.3 Mixed-canonical form and the orthogonality center

A state is in **mixed-canonical form with center on the bond $(i, i{+}1)$** when

```math
\underbrace{A^{(1)},\dots,A^{(i-1)}}_{\text{left-canonical}}, \quad
\underbrace{A^{(i)},A^{(i+1)}}_{\text{center}}, \quad
\underbrace{A^{(i+2)},\dots,A^{(N)}}_{\text{right-canonical}} .
```

All of the state's weight then sits in the two center tensors. The next section
shows why this is the ideal place to optimize.

### 3.4 Why canonicalize? Why normalize?

These gauge and scale choices are not cosmetic — they are what make two-site
DMRG *correct, optimal, and stable*. Each point below is tied to where its
payoff is used.

**Why normalize.** A quantum state is defined only up to scale: $|\psi\rangle$
and $c|\psi\rangle$ describe the same physics. Every observable is a *ratio*,

```math
E[\psi] = \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle},
\qquad
\langle \hat O\rangle = \frac{\langle\psi|\hat O|\psi\rangle}{\langle\psi|\psi\rangle}.
```

Fixing $\langle\psi|\psi\rangle = 1$ removes the denominator, so the numerator
alone is the answer. Three concrete consequences:

1. *The variational principle reads off directly.* With
   $\lVert\psi\rVert = 1$, $E[\psi] = \langle\psi|H|\psi\rangle \ge E_0$, and the
   local eigenvalue $\lambda_{\min}(H_{\mathrm{eff}})$ is immediately an energy
   (§5.5).
2. *The discarded weight becomes a relative error.* The local eigenvector
   $\Theta$ returned by Lanczos is normalized, so its Schmidt values satisfy
   $\sum_k \sigma_k^2 = 1$. The truncation error $\epsilon = \sum_{k>\chi}\sigma_k^2$
   (§6.2) is then a pure *fraction* of the state discarded, not a value in
   arbitrary units.
3. *Numerical range control.* An MPS amplitude is a product of $N$ matrices;
   without rescaling $\lVert\psi\rVert$ drifts exponentially in $N$ and the
   tensors under/overflow. Normalizing keeps entries $\mathcal{O}(1)$.

`normalize!` divides one site tensor by $\lVert\psi\rVert$ (rescaling one factor
rescales the whole product); `dmrg!` normalizes both the initial and final
states.

**Why canonicalize.** An MPS carries an enormous *gauge freedom*: on any bond one
may insert an invertible $G$ and its inverse,

```math
A^{(i)} \to A^{(i)} G, \qquad A^{(i+1)} \to G^{-1} A^{(i+1)},
```

leaving $|\psi\rangle$ unchanged. Canonical form fixes this freedom by demanding
that each tensor be an isometry (§3.1–3.2). Why this particular gauge is the
right one:

1. **Orthonormal block bases $\Rightarrow$ isometric embedding.**
   Left-canonical tensors make the left-block states $\{|\Lambda_l\rangle\}$
   orthonormal, and right-canonical tensors make $\{|P_r\rangle\}$ orthonormal
   (§5.2). Hence the two-site embedding $P:\Theta\mapsto|\psi\rangle$ satisfies
   $P^\dagger P = \mathbb{1}$ (§5.3). The whole method rests on this.

2. **A standard eigenproblem instead of a generalized one.** In a *general*
   gauge the block states are not orthonormal. The local energy is still
   $\Theta^\dagger H_{\mathrm{eff}}\Theta$ with $H_{\mathrm{eff}} = P^\dagger H P$,
   but the norm becomes $\Theta^\dagger (P^\dagger P)\Theta = \Theta^\dagger M\Theta$
   with a non-trivial metric $M = P^\dagger P \ne \mathbb{1}$. Minimizing the
   Rayleigh quotient then means solving the *generalized* eigenproblem

   ```math
   H_{\mathrm{eff}}\,\Theta \;=\; \lambda\, M\,\Theta ,
   ```

   which is costlier, less stable, and ill-conditioned when $M$ is near-singular.
   Canonical form sets $M = \mathbb{1}$, collapsing this to the ordinary
   Hermitian eigenproblem $H_{\mathrm{eff}}\Theta = \lambda\Theta$ that
   `KrylovKit.eigsolve(ishermitian=true)` solves (§5.5). It also guarantees the
   variational bound, because every trial $P\Theta$ is then genuinely normalized.

3. **Optimal, controlled truncation.** Because the surrounding blocks are
   orthonormal, the SVD of the local tensor $\Theta$ *is* the Schmidt
   decomposition of the global state across that bond (§6.1). By the
   Eckart–Young theorem, keeping the largest $\chi$ singular values is the
   provably optimal rank-$\chi$ approximation of $|\psi\rangle$, and the
   discarded weight equals the exact loss of state norm. In a non-canonical
   gauge the local SVD has no such global meaning and truncation is uncontrolled.

4. **The singular values are physical.** In canonical form the reduced density
   matrix at the bond is diagonal with eigenvalues $\sigma_k^2$; the
   $\{\sigma_k^2\}$ are the entanglement spectrum and
   $-\sum_k \sigma_k^2 \log \sigma_k^2$ the entanglement entropy. Truncating by
   $\sigma_k^2$ discards the least-entangled — least important — directions.

5. **Cheap local observables.** With the orthogonality center at site $i$, the
   left and right environments of an expectation value collapse to identities, so
   $\langle\hat O_i\rangle$ reduces to a single local trace $\operatorname{tr}(O\,\rho_i)$
   (§8). Measurements cost $\mathcal{O}(1)$ instead of $\mathcal{O}(N)$.

6. **Incremental maintenance.** The chain is never fully re-canonicalized inside a
   sweep. The SVD split reassigns $U$ or $V^\dagger$ so the one site just updated
   becomes canonical, walking the center by a single bond (§6.3). Canonical form
   uses only QR/SVD, which are backward-stable.

`right_canonicalize!` puts the whole chain in right-canonical form once at the
start — so the first left-to-right sweep begins with the center at the left edge
— after which the sweep maintains the mixed-canonical form for free.

---

## 4. Environments

### 4.1 Definition

An environment is a partially contracted $\langle\psi|H|\psi\rangle$ network with
the bonds at one cut left open. The **left environment after site $i$** contracts
sites $1,\dots,i$,

```math
L^{(i)}[r_b, b, r_k] \;=\!\!
\sum_{\text{sites } 1..i}\!\!
      \Bigl(\textstyle\prod_{j\le i}\overline{A^{(j)}}\Bigr)
      \Bigl(\textstyle\prod_{j\le i} W^{(j)}\Bigr)
      \Bigl(\textstyle\prod_{j\le i} A^{(j)}\Bigr),
```

with free indices $r_b$ (bra bond), $b$ (MPO bond), $r_k$ (ket bond) on the cut.
The **right environment before site $i$**, $R^{(i)}$, is the mirror image over
sites $i,\dots,N$.

### 4.2 Recurrences

The environments obey one-site recurrences. Starting from the $1\times1\times1$
boundaries $L^{(0)} = R^{(N+1)} = 1$, `absorb_left`/`absorb_right`
(`src/core/dmrg.jl`) apply

```math
L^{(i)}[r_b, b, r_k] \;=\; \sum_{l_b,a,l_k,\,s',s}
      \overline{A^{(i)}[l_b, s', r_b]}\; L^{(i-1)}[l_b, a, l_k]\;
      W^{(i)}[a, s', s, b]\; A^{(i)}[l_k, s, r_k],
```

```math
R^{(i)}[l_b, a, l_k] \;=\; \sum_{r_b,b,r_k,\,s',s}
      \overline{A^{(i)}[l_b, s', r_b]}\; W^{(i)}[a, s', s, b]\;
      A^{(i)}[l_k, s, r_k]\; R^{(i+1)}[r_b, b, r_k].
```

The bra leg contracts the MPO **output** index $s'$ and the ket leg the **input**
index $s$, matching the MPO convention of §2.1.

> **Code indexing.** `EnvironmentCache` stores these in 1-based arrays with a
> one-slot offset: `cache.left[i+1]` holds $L^{(i)}$ (so `cache.left[1]` is
> $L^{(0)}$), and `cache.right[i]` holds $R^{(i)}$ (so `cache.right[N+1]` is
> $R^{(N+1)}$). Keep this offset in mind when reading the contractions below.

### 4.3 Energy from environments

Closing all the way to a boundary contracts the entire network, so both extreme
environments equal the expectation value:

```math
L^{(N)} \;=\; R^{(1)} \;=\; \langle\psi|H|\psi\rangle
\qquad(\text{a } 1\times1\times1 \text{ scalar}).
```

The primitives test checks exactly this against the dense energy
$\langle\psi|H|\psi\rangle = \psi^\dagger\, \mathrm{dense}(H)\,\psi$.

---

## 5. The two-site optimization

### 5.1 The two-site tensor

At bond $(i, i{+}1)$ the two center tensors are merged into

```math
\Theta[l, s_1, s_2, r] \;=\; \sum_{m} A^{(i)}[l, s_1, m]\; A^{(i+1)}[m, s_2, r]
```

(`two_site_tensor`). This four-leg object holds the variational parameters we
optimize at this step.

### 5.2 The block bases and why the center is special

Assume mixed-canonical form (§3.3). The left-canonical tensors
$A^{(1)},\dots,A^{(i-1)}$ define left-block states, and the right-canonical
$A^{(i+2)},\dots,A^{(N)}$ define right-block states,

```math
|\Lambda_l\rangle \;=\!\!\sum_{s_1\dots s_{i-1}}\!\!
      \bigl(A^{(1)}_{s_1}\cdots A^{(i-1)}_{s_{i-1}}\bigr)_{l}\,
      |s_1\dots s_{i-1}\rangle,
\qquad
|P_r\rangle \;=\!\!\sum_{s_{i+2}\dots s_N}\!\!
      \bigl(A^{(i+2)}_{s_{i+2}}\cdots A^{(N)}_{s_N}\bigr)_{r}\,
      |s_{i+2}\dots s_N\rangle .
```

The canonical conditions of §3.1–3.2 are precisely the statements that these
blocks are **orthonormal**,

```math
\langle\Lambda_{l'}|\Lambda_l\rangle = \delta_{l l'},
\qquad
\langle P_{r'}|P_r\rangle = \delta_{r r'} .
```

In this basis the full state is carried entirely by $\Theta$,

```math
|\psi\rangle \;=\; \sum_{l, s_1, s_2, r}
      \Theta[l, s_1, s_2, r]\; |\Lambda_l\rangle\,|s_1\rangle\,|s_2\rangle\,|P_r\rangle .
```

### 5.3 The effective Hamiltonian as a projection

Define the linear embedding $P$ that maps a two-site tensor to the full space,
$P:\Theta \mapsto |\psi\rangle$ via the formula above. Because the four families
$\{|\Lambda_l\rangle\}, \{|s_1\rangle\}, \{|s_2\rangle\}, \{|P_r\rangle\}$ are
orthonormal, $P$ is an **isometry**:

```math
P^\dagger P = \mathbb{1}
\qquad\Longrightarrow\qquad
\langle\psi|\psi\rangle = \sum_{l,s_1,s_2,r} |\Theta[l,s_1,s_2,r]|^2 = \lVert\Theta\rVert^2 .
```

The energy is therefore an ordinary quadratic form in $\Theta$,

```math
\langle\psi|H|\psi\rangle \;=\; \Theta^\dagger\, H_{\mathrm{eff}}\, \Theta,
\qquad
H_{\mathrm{eff}} \;=\; P^\dagger H P .
```

$H_{\mathrm{eff}}$ is exactly the surrounding tensor network with the two center
tensors removed: the left block collapses to $L^{(i-1)}$, the right block to
$R^{(i+2)}$, and the two bare MPO tensors $W^{(i)}, W^{(i+1)}$ remain.

### 5.4 Matrix-free action

We never build $H_{\mathrm{eff}}$ as a dense matrix. `effective_action`
(`src/core/dmrg.jl`) applies it to a trial $\Theta$ by contraction:

```math
(H_{\mathrm{eff}}\Theta)[l, s'_1, s'_2, r]
\;=\!\!\sum_{\substack{l_k,\,a,\,m,\,b,\,r_k \\ s_1,\,s_2}}\!\!
      L^{(i-1)}[l, a, l_k]\;
      W^{(i)}[a, s'_1, s_1, m]\;
      W^{(i+1)}[m, s'_2, s_2, b]\;
      R^{(i+2)}[r, b, r_k]\;
      \Theta[l_k, s_1, s_2, r_k].
```

Here $L^{(i-1)}$ is `cache.left[i]` and $R^{(i+2)}$ is `cache.right[i+2]` in code
indexing (§4.2). The primitives test validates this against the dense form
$P^\dagger\, \mathrm{dense}(H)\, P$ for every bond.

### 5.5 The local eigenproblem and the variational bound

Because $H$ is Hermitian and the same $P$ appears on both sides,
$H_{\mathrm{eff}} = P^\dagger H P$ is Hermitian. Minimizing the Rayleigh quotient

```math
\min_{\Theta \ne 0} \frac{\Theta^\dagger H_{\mathrm{eff}} \Theta}{\Theta^\dagger \Theta}
\;=\; \lambda_{\min}(H_{\mathrm{eff}})
```

is a standard Hermitian eigenproblem, solved for the lowest algebraic eigenpair
by `lowest_local_state` via `KrylovKit.eigsolve(:SR; ishermitian=true)` (Lanczos),
warm-started from the current $\Theta$. Since $P$ is an isometry, every trial
$P\Theta$ is a normalized physical state, so

```math
\lambda_{\min}(H_{\mathrm{eff}}) \;=\; \min_{\lVert\Theta\rVert=1}\langle P\Theta|H|P\Theta\rangle
\;\ge\; E_0(H).
```

**Each local solve yields a variational upper bound on the ground-state energy.**
This is the property that makes the sweep well-behaved.

---

## 6. Truncation

### 6.1 SVD of the two-site tensor

After the local solve, $\Theta$ generally has a larger bond between sites $i$ and
$i+1$ than we wish to keep. Reshape it into the matrix
$\Theta_{(l s_1),\,(s_2 r)}$ and take its singular value decomposition
(`split_two_site!`),

```math
\Theta_{(l s_1),\,(s_2 r)} \;=\; \sum_{k} U_{(l s_1),\,k}\; \sigma_k\; V^\dagger_{k,\,(s_2 r)},
\qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge 0 .
```

The singular values $\sigma_k$ are the Schmidt coefficients of the bipartition at
this bond; $\sigma_k^2$ are the weights of the reduced density matrix.

### 6.2 Kept dimension, cutoff, and discarded weight

`kept_dimension` chooses how many Schmidt values to keep. It retains the largest
$\chi = \min(\text{maxdim}, \text{rank})$ subject to the discarded tail staying
within a relative cutoff,

```math
\epsilon(\chi) \;=\; \sum_{k > \chi} \sigma_k^2
\;\le\; \text{cutoff}\cdot\sum_{k} \sigma_k^2 .
```

The **discarded weight** $\epsilon(\chi)$ is the squared error introduced in the
state at this split; `split_two_site!` returns it and the sweep sums it as a
truncation diagnostic. Since the local eigenvector is normalized,
$\sum_k \sigma_k^2 = \lVert\Theta\rVert^2 = 1$, so $\epsilon$ is directly the
relative truncation error.

### 6.3 Moving the orthogonality center

How the factors are reattached decides which way the orthogonality center moves
and keeps the mixed-canonical form intact for the *next* bond:

```math
\begin{aligned}
\textbf{right sweep:}\quad & A^{(i)} \leftarrow U, & A^{(i+1)} &\leftarrow \Sigma V^\dagger , \\
\textbf{left sweep:}\quad  & A^{(i)} \leftarrow U \Sigma, & A^{(i+1)} &\leftarrow V^\dagger ,
\end{aligned}
```

with $\Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_\chi)$. On a right sweep
$U$ has orthonormal columns, so site $i$ becomes left-canonical and the center
moves to $i+1$; on a left sweep $V^\dagger$ has orthonormal rows, so site $i+1$
becomes right-canonical and the center moves to $i-1$. Either way the isometry
argument of §5.3 holds again at the next bond — the variational bound is
preserved as the center walks along the chain.

---

## 7. The sweep algorithm

### 7.1 One half-sweep

A right-moving half-sweep (`sweep!` with `direction = :right`) visits bonds
$i = 1, 2, \dots, N-1$. At each bond it:

1. forms $\Theta$ from the current center tensors (§5.1);
2. solves the local eigenproblem using $L^{(i-1)}$ and $R^{(i+2)}$ (§5.4–5.5);
3. truncates via SVD and stores $U$ / $\Sigma V^\dagger$ (§6.3);
4. updates the left environment $L^{(i)}$ from the new $A^{(i)}$ with `absorb_left!`.

The left-moving half-sweep is the mirror image over $i = N-1, \dots, 1$, updating
right environments instead.

### 7.2 A full sweep

`dmrg!` performs, per sweep: a right half-sweep, then a left half-sweep, then a
single normalized energy evaluation $E^{(t)} = E[\psi]$ (§2.2). Only the right
environments actually consumed by the upcoming right half-sweep are precomputed
(`right_sweep_cache!`), and each local step refreshes exactly the one
environment the next step needs — so no full environment rebuild happens inside a
sweep.

### 7.3 Monotonicity and convergence

Within a sweep, each local eigen-minimization can only lower or preserve the
energy (§5.5). The single step that can *raise* it is truncation, and that
increase is bounded by the discarded weight $\epsilon$ of §6.2. Hence, for a
sufficiently small `cutoff`, the sweep energies form a non-increasing sequence
bounded below by $E_0(H)$:

```math
E^{(1)} \ge E^{(2)} \ge \cdots \ge E_0(H) .
```

A monotone bounded sequence converges. `dmrg!` therefore stops as soon as the
change between complete sweeps falls within tolerance,

```math
\bigl| E^{(t)} - E^{(t-1)} \bigr| \le \text{tol},
```

reporting `converged = true` with `stopping_reason = :energy_tolerance`; if the
sweep budget `nsweeps` is exhausted first it returns
`stopping_reason = :maximum_sweeps`. Reaching the budget is *not* proof of
convergence — inspect the final $|E^{(t)} - E^{(t-1)}|$.

### 7.4 Sweep schedules

Each of `maxdim`, `cutoff`, `tol`, and `eig_tol` may be a scalar or a per-sweep
schedule (tuple/vector). `schedule_value` returns entry
$\min(t, \text{length})$ on sweep $t$, so the final entry is reused for any
further sweeps. Gradually raising `maxdim` — a small bond early, larger later —
is often more robust and cheaper than starting at the final maximum.

### 7.5 Single-site DMRG with subspace expansion

The two-site update grows bonds for free through the SVD of the merged $\Theta$
(§6). A cheaper alternative optimizes **one** site tensor at a time. Its effective
Hamiltonian keeps a single MPO tensor,

```math
(H^{1}_{\mathrm{eff}} A)[l, s', r]
\;=\; \sum_{l_k,\,a,\,b,\,r_k,\,s}
      L^{(i-1)}[l, a, l_k]\, W^{(i)}[a, s', s, b]\, R^{(i+1)}[r, b, r_k]\, A[l_k, s, r_k],
```

solved for its lowest eigenpair exactly as in §5.5 (`effective_action_1site`,
`lowest_local_state_1site` in `src/core/single_site_dmrg.jl`). The center is then
moved by a QR/SVD of the single tensor.

**The problem.** A single-site update cannot, by itself, change a bond dimension:
the SVD of one site tensor produces at most its current rank. Started from a
too-small bond it is trapped there and converges to the wrong state.

**Subspace expansion (a "DMRG3S"-style fix).** Before moving the center rightward
from site $i$, enlarge the active bond with an $H$-informed perturbation built
from the environment and MPO,

```math
P[l, s', (b, r)] \;=\; \sum_{l_k,\,a,\,s}
      L^{(i-1)}[l, a, l_k]\, W^{(i)}[a, s', s, b]\, A^{(i)}[l_k, s, r] ,
```

which shares the left index $l$ and physical index $s'$ of the optimized tensor
$M = A^{(i)}$ but carries a new right index $(b, r)$ of size $w\cdot r$. Append it
to $M$ and pad the *next* site with a matching block of zeros:

```math
M_{\mathrm{enl}} = \bigl[\,M \;\big|\; \alpha P\,\bigr], \qquad
B_{\mathrm{enl}} = \begin{bmatrix} B \\ 0 \end{bmatrix}.
```

Because the appended columns of $M_{\mathrm{enl}}$ multiply the zero rows of
$B_{\mathrm{enl}}$, the product — and hence $|\psi\rangle$ — is **unchanged**:
$M_{\mathrm{enl}} B_{\mathrm{enl}} = M B$. Now SVD $M_{\mathrm{enl}}$ and truncate
to `maxdim` (§6.2): the perturbation has opened $H$-informed directions the plain
bond could not represent, so the retained left-canonical basis can *grow* toward
`maxdim`, and later sweeps populate the new directions. The left move is the
mirror image, enlarging the left bond from $R^{(i+1)}$, $W^{(i)}$, and $M$.

The mixing factor $\alpha$ controls how strongly the perturbation enters. It must
be small (so the truncation does not discard real directions in favor of noise)
and is **decayed to zero** over the sweeps; once $\alpha = 0$ the method is exact
single-site DMRG and the run is allowed to declare convergence. `single_site_dmrg`
takes an `alpha` schedule for exactly this. Starting from a bond-2 state, it grows
the four-site Heisenberg chain to the exact bonds $[2,4,2]$ and recovers the
ground-state energy to machine precision (see the tests).

---

## 8. Local observables and correlation functions

A converged state is only useful once we can measure it. For a normalized state
the expectation value of a local operator $\hat O_i$ — the $d\times d$ matrix
$O[s', s] = \langle s'|\hat O|s\rangle$ acting on site $i$, the identity
elsewhere — is

```math
\langle \hat O_i\rangle
      \;=\; \frac{\langle\psi|\hat O_i|\psi\rangle}{\langle\psi|\psi\rangle}.
```

### 8.1 Single-site expectation

Evaluate the numerator with the same MPS–MPS contraction as the overlap (§1.3),
inserting $O$ at site $i$. Sweep a two-index environment $E[\text{bra},\text{ket}]$
from the boundary $E^{(0)} = 1$:

```math
E^{(k)}[r_a, r_b] = \begin{cases}
\displaystyle\sum_{l_a,l_b,s',s}
      \overline{A^{(k)}[l_a, s', r_a]}\; E^{(k-1)}[l_a, l_b]\; O[s', s]\; A^{(k)}[l_b, s, r_b],
      & k = i, \\[10pt]
\displaystyle\sum_{l_a,l_b,s}
      \overline{A^{(k)}[l_a, s, r_a]}\; E^{(k-1)}[l_a, l_b]\; A^{(k)}[l_b, s, r_b],
      & k \ne i.
\end{cases}
```

The scalar $E^{(N)}$ is $\langle\psi|\hat O_i|\psi\rangle$; dividing by
$\langle\psi|\psi\rangle$ gives $\langle\hat O_i\rangle$. This is
`expect(psi, op, site)` (`src/core/observables.jl`); the per-site step
`_absorb_site` is exactly the two cases above.

As noted in §3.4(5), if the orthogonality center sits at site $i$ every
environment but the center collapses to the identity and this reduces to the
single local trace $\langle\hat O_i\rangle = \operatorname{tr}(O\,\rho_i)$. The
implementation performs the full sweep, so it is correct in any gauge.

### 8.2 Two-point correlations

For $\hat O^1_i$ and $\hat O^2_j$ the same sweep inserts *both* operators — $O^1$
at $k=i$ and $O^2$ at $k=j$ — and returns

```math
\langle \hat O^1_i\, \hat O^2_j\rangle \;=\; \frac{E^{(N)}}{\langle\psi|\psi\rangle}.
```

Operators on distinct sites act on different tensor factors and therefore
commute, so the value does not depend on whether $i<j$ or $i>j$; the on-site case
$i=j$ inserts the product $O^1 O^2$. These are `correlation(psi, op1, op2, i, j)`
and, assembled over all pairs, `correlation_matrix` returning
$C[i,j] = \langle\hat O^1_i \hat O^2_j\rangle$.

Values are `ComplexF64`: a Hermitian operator gives a real result up to rounding,
while a genuinely non-Hermitian correlation such as
$\langle S^+_i S^-_j\rangle$ is complex.

### 8.3 Example: spin-1/2

`spin_half_operators()` returns $S^x, S^y, S^z, S^+, S^-, \mathbb{1}$ as
$2\times2$ matrices. Useful diagnostics for the Heisenberg ground state are the
connected correlation

```math
C^{zz}_{ij} \;=\; \langle S^z_i S^z_j\rangle - \langle S^z_i\rangle\langle S^z_j\rangle,
```

and two exact identities for the singlet ground state on an even chain:
$\langle (S^z_i)^2\rangle = \tfrac14$ on the diagonal, and
$\sum_{ij}\langle S^z_i S^z_j\rangle = \langle(\textstyle\sum_i S^z_i)^2\rangle = 0$
because the state lies in the total-$S^z = 0$ sector. Both are checked in the
tests.

### 8.4 Entanglement entropy and the Schmidt spectrum

The Schmidt decomposition of $|\psi\rangle$ across the cut between sites $b$ and
$b+1$ writes it as

```math
|\psi\rangle \;=\; \sum_{k} \sigma_k\, |L_k\rangle\,|R_k\rangle,
\qquad \sum_k \sigma_k^2 = 1,
```

with $\{|L_k\rangle\}$ and $\{|R_k\rangle\}$ orthonormal states of the two halves.
As §3.4(4) noted, the $\sigma_k^2$ are the eigenvalues of the reduced density
matrix $\rho_{\le b} = \operatorname{tr}_{>b}|\psi\rangle\langle\psi|$ — the
*entanglement spectrum* — and the von Neumann entanglement entropy is

```math
S(b) \;=\; -\sum_k \sigma_k^2 \log \sigma_k^2 .
```

Canonical form turns this into a single SVD. Put the orthogonality center at site
$b$ — sites $1..b{-}1$ left-canonical, $b{+}1..N$ right-canonical — so that
$|\psi\rangle = \sum_{l,s,r} C[l,s,r]\,|\Lambda_l\rangle|s\rangle|P_r\rangle$ with
both block families orthonormal (§5.2). The combined index $(l,s)$ labels the
left half $\{1..b\}$ and $r$ the right half $\{b{+}1..N\}$, both orthonormally, so
the singular values of the reshaped center $C_{(l s),\,r}$ *are* the Schmidt
values $\sigma_k$.

`schmidt_values(psi, bond)` (`src/core/entanglement.jl`) does exactly this on a
copy: it normalizes, right-canonicalizes, walks the center to site `bond` with
QR factorizations, and returns $\operatorname{svdvals}$ of the reshaped center.
`entanglement_entropy(psi, bond; base)` forms $S(b)$ from those values (`base=2`
for bits), and `entanglement_entropy(psi)` returns the profile over all bonds. A
product state gives $\sigma = (1)$ and $S = 0$; a singlet gives
$\sigma = (\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2})$ and $S = \log 2$ — both checked
against a dense SVD in the tests.

---

## 9. Cost and memory

Let $\chi$ be the typical MPS bond dimension and $w$ the MPO bond dimension
($w = 5$ for the Heisenberg chain). The dominant cost is the repeated
application of $H_{\mathrm{eff}}$ during each local Lanczos solve. Contracting
the network of §5.4 in a good order costs on the order of

```math
\mathcal{O}\!\left(\chi^3 d^2 w \;+\; \chi^2 d^3 w^2\right)
```

per matrix–vector product, with the exact figure set by the contraction
optimizer. Memory is polynomial: the environments occupy
$\mathcal{O}(N\chi^2 w)$ and each $\Theta$ is $\mathcal{O}(\chi^2 d^2)$. This is
the whole point of the matrix-free formulation — nothing here scales like the
dense objects `dense(psi)` ($d^N$) or `dense(H)` ($d^N \times d^N$), which exist
only for small-system checks.

---

## 10. What the tests verify

The derivation above is pinned down, piece by piece, by the test suite:

- **`test/naive_primitives_test.jl`** — the environment recurrences of §4.2
  against an independent dense reference; the effective action of §5.4 against
  $P^\dagger\,\mathrm{dense}(H)\,P$ at every bond; Hermiticity of
  $H_{\mathrm{eff}}$ (§5.5); and the minimal right-sweep environment cache.
- **`test/naive_dmrg_test.jl`** — the four-site Heisenberg ground-state energy
  $E_0 = -1.6160254037844386$ from exact diagonalization (§1.2, §2.1); agreement
  of the sweep result (§7); the variational bound and its improvement as `maxdim`
  increases (§5.5, §7.4); and agreement with the ITensor reference DMRG.
- **`test/single_site_dmrg_test.jl`** — single-site DMRG with subspace expansion
  (§7.5) growing a bond-restricted state to the exact bonds and energy, agreeing
  with two-site DMRG, and solving the TFIM ground state.
- **`test/mpo_builder_test.jl`** — the builder of §2.4 reproducing `heisenberg_mpo`
  to machine precision, and `tfim_mpo` against a dense Kronecker Hamiltonian and
  exact-diagonalization ground state.
- **`test/observables_test.jl`** — `expect` and `correlation` (§8) against dense
  reference operators built by Kronecker products, exact values on a product
  state, and the singlet identities $\langle(S^z_i)^2\rangle = \tfrac14$ and
  $\sum_{ij}\langle S^z_i S^z_j\rangle = 0$.
- **`test/entanglement_test.jl`** — `schmidt_values`/`entanglement_entropy` (§8.4)
  against a dense reduced density matrix, with $S=0$ for a product state and
  $S=\log 2$ for a singlet.
- **`test/reference_test.jl`** — keeps the `NaiveDMRG.Reference` baseline healthy.

Run them with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Notation summary

| Symbol | Meaning |
| --- | --- |
| $N,\ d$ | number of sites, physical dimension |
| $D_i,\ \chi$ | bond dimension on link $i$; typical/maximum bond dimension |
| $w$ | MPO bond dimension |
| $A^{(i)}[l,s,r]$ | MPS site tensor (left, physical, right) |
| $W^{(i)}[a,s',s,b]$ | MPO site tensor (left, out, in, right) |
| $L^{(i)},\ R^{(i)}$ | left/right environments (bra, MPO, ket) |
| $\Theta[l,s_1,s_2,r]$ | two-site center tensor at bond $(i,i{+}1)$ |
| $P,\ H_{\mathrm{eff}}$ | two-site embedding isometry; $P^\dagger H P$ |
| $\sigma_k,\ \epsilon$ | Schmidt values; discarded weight $\sum_{k>\chi}\sigma_k^2$ |
| $E[\psi],\ E^{(t)}$ | normalized energy; energy after sweep $t$ |
