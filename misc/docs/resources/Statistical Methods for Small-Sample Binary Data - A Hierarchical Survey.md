# Statistical Methods for Small-Sample Binary Data: A Hierarchical Survey

**Author:** Thor Whalen
**Date:** March 2026

---

> **Abstract.** This survey organizes the principal statistical methods for exploring associations in small-sample binary datasets (10–50 cases, 10–30 binary variables) around a single conceptual trunk: **Bayesian posterior inference on 2×2 tables**. Every other technique—Bayes factors, exact frequentist tests, effect-size intervals, QCA, association-rule metrics, Naïve Bayes, log-linear models, and penalized regression—is shown to branch from, or connect back to, the same cell counts and the same posterior distributions. Explicit formulas, small-sample caveats, and Python implementation notes are provided throughout. The intended reader is a Python architect building an interactive data-exploration tool.

---

## Table of contents

1. [The Bayesian trunk: posterior inference on a 2×2 table](#1-the-bayesian-trunk-posterior-inference-on-a-2×2-table)
2. [Bayes factors for 2×2 tables](#2-bayes-factors-for-2×2-tables)
3. [Frequentist exact tests](#3-frequentist-exact-tests)
4. [Effect-size measures with exact confidence intervals](#4-effect-size-measures-with-exact-confidence-intervals)
5. [Connections across the 2×2 paradigm](#5-connections-across-the-2×2-paradigm)
6. [Set-theoretic and Boolean methods](#6-set-theoretic-and-boolean-methods)
7. [Extension to n-tuple conditions](#7-extension-to-n-tuple-conditions)
8. [Prior elicitation for Bayesian components](#8-prior-elicitation-for-bayesian-components)
9. [References](#references)

---

## 1. The Bayesian trunk: posterior inference on a 2×2 table

All methods in this survey operate on a single data structure—the **2×2 contingency table**:

|  | Y = 1 | Y = 0 | Row total |
|---|---|---|---|
| **X = 1** | a | b | n₁ = a + b |
| **X = 0** | c | d | n₀ = c + d |
| **Col total** | a + c | b + d | N = a + b + c + d |

Under the **independent-multinomial** (row-fixed) sampling scheme, each row is an independent Binomial:

$$a \sim \mathrm{Binomial}(n_1,\;\pi_1), \qquad c \sim \mathrm{Binomial}(n_0,\;\pi_0)$$

where π₁ = P(Y = 1 | X = 1) and π₀ = P(Y = 1 | X = 0).

### 1.1 Conjugate Beta-Binomial model

Assign independent conjugate priors:

$$\pi_1 \sim \mathrm{Beta}(\alpha_1, \beta_1), \qquad \pi_0 \sim \mathrm{Beta}(\alpha_0, \beta_0)$$

**Common defaults** (small-sample behavior differs):

| Prior | α, β | Character | Small-sample note |
|---|---|---|---|
| Uniform (Bayes–Laplace) | 1, 1 | Flat on [0, 1] | Adds 2 pseudo-observations; ~13% prior weight at n = 13 |
| Jeffreys | 0.5, 0.5 | Reference prior; invariant to reparameterization | Adds 1 pseudo-observation; ~7% prior weight at n = 13 |
| Haldane | ε → 0 | Improper; flat on logit(π) | Data-dominated; undefined if a = 0 or b = 0 |

By conjugacy the posteriors are available in closed form [1, 5]:

$$\pi_1 \mid \text{data} \sim \mathrm{Beta}(\alpha_1 + a,\;\beta_1 + b)$$

$$\pi_0 \mid \text{data} \sim \mathrm{Beta}(\alpha_0 + c,\;\beta_0 + d)$$

The posterior mean is a **shrinkage estimator**—a weighted average of the prior mean and the MLE:

$$E[\pi_1 \mid \text{data}] = \underbrace{\frac{n_1}{\alpha_1+\beta_1+n_1}}_{w}\;\frac{a}{n_1} \;+\; \underbrace{\frac{\alpha_1+\beta_1}{\alpha_1+\beta_1+n_1}}_{1-w}\;\frac{\alpha_1}{\alpha_1+\beta_1}$$

The posterior variance is:

$$\mathrm{Var}[\pi_1 \mid \text{data}] = \frac{(\alpha_1+a)(\beta_1+b)}{(\alpha_1+\beta_1+n_1)^2\,(\alpha_1+\beta_1+n_1+1)}$$

### 1.2 Derived quantities as functions of the posteriors

Three effect measures follow directly from the two proportion posteriors. **None has a tractable closed-form posterior**; all are computed by Monte Carlo sampling.

**Risk Difference (RD):**  RD = π₁ − π₀

**Relative Risk (RR):**  RR = π₁ / π₀

**Odds Ratio (OR):**  OR = [π₁/(1 − π₁)] / [π₀/(1 − π₀)]

### 1.3 Monte Carlo algorithm

```
ALGORITHM  Posterior-MC-2×2
───────────────────────────────────────────
INPUT   (a, b, c, d), priors (α₁,β₁,α₀,β₀), S draws
STEP 1  α₁* ← α₁+a,  β₁* ← β₁+b,  α₀* ← α₀+c,  β₀* ← β₀+d
STEP 2  FOR s = 1 … S :
          π₁⁽ˢ⁾ ~ Beta(α₁*, β₁*)
          π₀⁽ˢ⁾ ~ Beta(α₀*, β₀*)          (independent draw)
          RD⁽ˢ⁾ = π₁⁽ˢ⁾ − π₀⁽ˢ⁾
          RR⁽ˢ⁾ = π₁⁽ˢ⁾ / π₀⁽ˢ⁾
          OR⁽ˢ⁾ = [π₁⁽ˢ⁾/(1−π₁⁽ˢ⁾)] / [π₀⁽ˢ⁾/(1−π₀⁽ˢ⁾)]
STEP 3  Posterior summaries from {RD⁽ˢ⁾}, {RR⁽ˢ⁾}, {OR⁽ˢ⁾}:
          • mean, median
          • 95% equal-tail credible interval (2.5th and 97.5th percentiles)
          • P(θ > 0 | data)  or  P(θ > 1 | data)
OUTPUT  Posterior distributions and summaries for RD, RR, OR
```

**Small-sample note.** When π₀ is near zero (common with sparse data), RR and OR develop extreme right skew. Working on the **log scale** (log RR, log OR) produces more symmetric, better-behaved posteriors [5]. With S = 100 000 draws, Monte Carlo standard error is negligible.

**Python (NumPy/SciPy):**

```python
import numpy as np
from scipy import stats

def bayesian_2x2(a, b, c, d, alpha1=0.5, beta1=0.5,
                 alpha0=0.5, beta0=0.5, S=100_000):
    pi1 = np.random.beta(alpha1 + a, beta1 + b, S)
    pi0 = np.random.beta(alpha0 + c, beta0 + d, S)
    RD  = pi1 - pi0
    RR  = pi1 / pi0
    OR  = (pi1 / (1 - pi1)) / (pi0 / (1 - pi0))
    def summary(x, name, null=0):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return dict(name=name, mean=x.mean(), median=np.median(x),
                    ci95=(lo, hi), prob_gt_null=np.mean(x > null))
    return [summary(RD, "RD"), summary(RR, "RR", 1), summary(OR, "OR", 1)]
```

---

## 2. Bayes factors for 2×2 tables

While the posterior tells us *what we believe about the parameters*, the Bayes factor tells us *how much the data shift our belief between two hypotheses*. The Bayes factor compares model H₁ (association) against H₀ (independence) [32]:

$$\mathrm{BF}_{10} = \frac{P(\text{data} \mid \mathcal{H}_1)}{P(\text{data} \mid \mathcal{H}_0)} = \frac{\int P(\text{data} \mid \boldsymbol{\theta}, \mathcal{H}_1)\,\pi(\boldsymbol{\theta} \mid \mathcal{H}_1)\,d\boldsymbol{\theta}}{\int P(\text{data} \mid \boldsymbol{\gamma}, \mathcal{H}_0)\,\pi(\boldsymbol{\gamma} \mid \mathcal{H}_0)\,d\boldsymbol{\gamma}}$$

**Relation to posterior odds:**

$$\underbrace{\frac{P(\mathcal{H}_1 \mid \text{data})}{P(\mathcal{H}_0 \mid \text{data})}}_{\text{posterior odds}} = \underbrace{\frac{P(\mathcal{H}_1)}{P(\mathcal{H}_0)}}_{\text{prior odds}} \;\times\; \mathrm{BF}_{10}$$

Under prior equipoise (P(H₁) = P(H₀) = 0.5), the posterior probability of H₁ simplifies to BF₁₀ / (BF₁₀ + 1).

### 2.1 The Gunel–Dickey framework

Gunel and Dickey [2] derived closed-form Bayes factors for R × C tables under four sampling schemes, using Dirichlet-type priors with a single concentration parameter *a* (default a = 1). Jamil et al. [3] provided a modern exposition with corrected formulas and extensive simulation.

**Independent multinomial (row totals fixed)** — the most common experimental design:

$$\mathrm{BF}_{01}^{I} = \frac{\mathcal{D}(\mathbf{y}_{.*} + \boldsymbol{\xi}_{.*})}{\mathcal{D}(\boldsymbol{\xi}_{.*})} \cdot \frac{\mathcal{D}(\mathbf{y}_{*.} + \mathbf{a}_{*.})}{\mathcal{D}(\mathbf{a}_{*.})} \cdot \frac{\mathcal{D}(\mathbf{a}_{**})}{\mathcal{D}(\mathbf{y}_{**} + \mathbf{a}_{**})}$$

where D(·) denotes the Dirichlet function D(**a**) = ∏Γ(aᵢ) / Γ(Σaᵢ), dot subscripts denote marginal sums, and ξ terms are derived from the concentration parameter [3].

For the **2×2 case with a = 1** (uniform Dirichlet prior), this simplifies to:

$$\mathrm{BF}_{01}^{I} = \frac{\binom{n_{.1}}{a}\binom{n_{.2}}{b}}{\binom{n_{.1}+n_{.2}}{a+b}} \cdot \frac{(n_{.1}+1)(n_{.2}+1)}{n_{..}+1}$$

Three additional sampling schemes are available [2, 3]:

| Scheme | Fixed quantities | Evidence ordering |
|---|---|---|
| Poisson | Nothing | Most evidence for H₁ |
| Joint multinomial | Grand total N | |
| Independent multinomial | Row totals | |
| Hypergeometric | Both margins | Least evidence for H₁ |

**Ordering:** BF₁₀ᴾ > BF₁₀ᴹ > BF₁₀ᴵ > BF₁₀ᴴ. Evidence for association decreases with successive conditioning.

**Small-sample behavior.** Jamil et al. [3] showed via simulation that for small samples, Bayes factors often remain in the "anecdotal" range (BF₁₀ < 3), even when a true effect exists. This is a feature, not a bug: the Bayes factor correctly quantifies that small datasets cannot provide overwhelming evidence. For the 2×2 case with a = 1, **all Gunel–Dickey BFs reduce to expressions involving only factorials** — no MCMC is needed. Use `scipy.special.gammaln` for numerical stability.

**Interpretation scale** (Lee and Wagenmakers, adapted from Jeffreys [33]):

| BF₁₀ | Evidence category |
|---|---|
| > 100 | Extreme for H₁ |
| 30–100 | Very strong |
| 10–30 | Strong |
| 3–10 | Moderate |
| 1–3 | Anecdotal |
| 1/3–1 | Anecdotal for H₀ |
| < 1/10 | Strong for H₀ |

**Software.** R's `BayesFactor::contingencyTableBF()` implements all four schemes. JASP provides a GUI. No mature Python package exists; implement from the factorial formulas above.

---

## 3. Frequentist exact tests

### 3.1 Fisher's exact test

Under H₀ (independence), conditioning on **all four marginal totals** yields the hypergeometric distribution for cell *a* [6]:

$$P(X = a) = \frac{\binom{n_1}{a}\,\binom{n_0}{a+c-a}}{\binom{N}{a+c}} = \frac{(a+b)!\,(c+d)!\,(a+c)!\,(b+d)!}{a!\,b!\,c!\,d!\,N!}$$

**One-sided p-value (right tail):**

$$p_{\text{right}} = \sum_{x \geq a} \frac{\binom{n_1}{x}\,\binom{n_0}{m_1-x}}{\binom{N}{m_1}}$$

where m₁ = a + c.

**Two-sided p-value** (method of small tables, used by SciPy and R):

$$p_{\text{two}} = \sum_{\{x\,:\,P(X=x) \,\leq\, P(X=a)\}} P(X = x)$$

**Small-sample conservatism.** Fisher's test is conservative (actual Type I error < α) for two reasons [39]:

- **Discreteness.** The hypergeometric distribution produces only finitely many achievable significance levels. The gap between actual size and nominal α can be substantial when N < 50.
- **Conditioning on both margins.** In most designs only row totals are fixed. Conditioning on both margins discards information, shrinking the reference set and increasing conservatism.

**Python:** `scipy.stats.fisher_exact(table)`.

### 3.2 Barnard's exact test

Barnard's test [7] conditions only on row totals (unconditional on column totals). Under H₀ with common success probability π:

$$P(\mathbf{x} \mid \pi) = \binom{n_1}{x_1}\binom{n_2}{x_2}\,\pi^{x_1+x_2}(1-\pi)^{n_1+n_2-x_1-x_2}$$

The **nuisance parameter** π is eliminated by maximization:

$$p_B = \sup_{\pi \in (0,1)} \sum_{\{x\,:\,T(x) \geq T(x_{\text{obs}})\}} P(\mathbf{x} \mid \pi)$$

The test statistic T is typically the pooled score statistic. **Boschloo's test** uses Fisher's p-value itself as T within Barnard's framework, yielding a test that is **uniformly more powerful** than Fisher's [39].

**When to prefer Barnard's.** When only one margin is fixed (the typical experimental design) and N < 50, Barnard's test exploits the richer unconditional reference set to achieve finer p-value granularity and higher power. Mehta and Senchaudhuri [39] showed that for n₁ = n₂ = 15, Barnard's p-value can be nearly half Fisher's.

**Python:** `scipy.stats.barnard_exact(table)` and `scipy.stats.boschloo_exact(table)`.

### 3.3 The mid-p correction

The mid-p value halves the point mass at the observed outcome [8]:

$$p_{\text{mid}} = P(T > t_{\text{obs}}) + \tfrac{1}{2}\,P(T = t_{\text{obs}}) = p_{\text{exact}} - \tfrac{1}{2}\,P(T = t_{\text{obs}})$$

Under H₀, the mid-p value has E[p_mid] = 0.5 (the ideal for a continuous test), whereas E[p_exact] > 0.5 for discrete distributions. The mid-p is **no longer strictly exact** — it does not guarantee P(reject | H₀) ≤ α — but in practice it is well-calibrated and recommended by Agresti [1] and Lancaster [8] as a pragmatic compromise.

---

## 4. Effect-size measures with exact confidence intervals

### 4.1 Odds ratio: Cornfield's exact CI

The **conditional exact CI** inverts Fisher's exact test using the noncentral hypergeometric distribution [13]. Under the alternative H₀: OR = θ₀, the conditional distribution of *a* is:

$$P(X = x \mid \theta_0) = \frac{\binom{n_1}{x}\binom{n_0}{m_1 - x}\,\theta_0^x}{\sum_k \binom{n_1}{k}\binom{n_0}{m_1 - k}\,\theta_0^k}$$

The limits (θ_L, θ_U) are found by solving:

$$\sum_{x=a}^{\min(n_1,m_1)} P(X = x \mid \theta_L) = \alpha/2 \qquad \text{(lower limit)}$$

$$\sum_{x=0}^{a} P(X = x \mid \theta_U) = \alpha/2 \qquad \text{(upper limit)}$$

Each equation is solved numerically (bisection). This CI is **truly exact** and conservative.

**Python:** `scipy.stats.contingency.odds_ratio(table, kind='conditional').confidence_interval()`.

### 4.2 Odds ratio: Baptista–Pike exact CI

Baptista and Pike [11] form the acceptance region by ordering tables in decreasing probability (rather than using tail areas), yielding **shorter intervals** with the same coverage guarantee. Combining with the mid-p correction gives the **Baptista–Pike mid-p interval**, which Fagerland et al. [14] recommend as the best choice for small samples.

### 4.3 Relative risk: Koopman score CI

The Koopman [12] method inverts a score test for ϕ = p₁/p₂. Given the constraint p̃₁ = ϕ₀ p̃₂, the constrained MLE is:

$$\tilde{p}_2 = \frac{-(A\phi_0 - x_1 - x_2) + \sqrt{(A\phi_0 - x_1 - x_2)^2 + 4\phi_0\,A\,x_1}}{2\,A\,\phi_0}$$

where A = n₁ + n₂ϕ₀ and p̃₁ = ϕ₀ p̃₂. The score statistic is:

$$T(\phi_0) = \frac{\hat{p}_1 - \phi_0\,\hat{p}_2}{\sqrt{\dfrac{\tilde{p}_1(1-\tilde{p}_1)}{n_1} + \phi_0^2\,\dfrac{\tilde{p}_2(1-\tilde{p}_2)}{n_2}}}$$

Solve T(ϕ_L) = −z_{α/2} and T(ϕ_U) = z_{α/2} by root-finding. This is approximate (score-based) but **well-calibrated for n ≥ 10** [14].

### 4.4 Risk difference: Agresti–Caffo interval

Add one pseudo-success and one pseudo-failure to each group, then apply the Wald formula [9]:

$$\tilde{p}_i = \frac{x_i + 1}{n_i + 2}$$

$$\text{CI:}\; (\hat{p}_1 - \hat{p}_2) \;\pm\; z_{\alpha/2}\,\sqrt{\frac{\tilde{p}_1(1-\tilde{p}_1)}{n_1+2} + \frac{\tilde{p}_2(1-\tilde{p}_2)}{n_2+2}}$$

The point estimate remains the unadjusted (x₁/n₁ − x₂/n₂). Coverage stays close to nominal for n ≥ 10 per group.

### 4.5 Risk difference: Newcombe hybrid score interval (Method 10)

1. Compute Wilson score CIs (l₁, u₁) and (l₂, u₂) for each proportion separately [10].
2. Combine via the "square-and-add" rule:

$$L = (\hat{p}_1 - \hat{p}_2) - \sqrt{(\hat{p}_1 - l_1)^2 + (u_2 - \hat{p}_2)^2}$$

$$U = (\hat{p}_1 - \hat{p}_2) + \sqrt{(\hat{p}_2 - l_2)^2 + (u_1 - \hat{p}_1)^2}$$

### 4.6 Summary of CI methods

| Effect | Method | Type | Best for |
|---|---|---|---|
| OR | Cornfield exact | Truly exact (conditional) | Conservative, all n |
| OR | Baptista–Pike mid-p | Quasi-exact | **Best for small n** |
| RR | Koopman score | Approximate (score) | Well-calibrated, n ≥ 10 |
| RD | Agresti–Caffo | Approximate (adjusted Wald) | Simple, n ≥ 10 |
| RD | Newcombe Method 10 | Approximate (hybrid score) | n ≥ 20 |

---

## 5. Connections across the 2×2 paradigm

Every method in Sections 1–4 operates on the same four cell counts (a, b, c, d). They differ in **what question they answer** [4, 40]:

| Method | Question answered | Output |
|---|---|---|
| Bayesian posterior | "What is the full probability distribution over the true proportions?" | π(θ ∣ data) |
| Bayes factor | "How much more likely is association vs. independence?" | BF₁₀ ∈ (0, ∞) |
| Frequentist p-value | "How surprising is this data if there is no association?" | p ∈ [0, 1] |
| Effect size + CI | "How large is the association, and how precisely is it known?" | θ̂ ± interval |

**The posterior as trunk.** The Bayesian posterior subsumes the others in several senses:

- **Posterior credible intervals approximate frequentist CIs** when priors are weak relative to data. With a Jeffreys prior (α = β = 0.5) and n ≥ 30, the 95% equal-tail credible interval for a proportion is numerically close to the Wilson score CI [41].
- **The Bayes factor can be computed from the posterior.** The marginal likelihood under a model M is the normalizing constant of the posterior: P(data | M) = ∫ P(data | θ) π(θ) dθ, which is the Beta-Binomial probability for the Beta–Binomial model.
- **Tail posterior probabilities relate to p-values.** Under a flat prior, P(θ ≤ 0 | data) ≈ one-sided p-value. However, Berger and Sellke [40] showed that p-values systematically *overstate* evidence against H₀ compared to posterior probabilities.

**Key difference.** Bayesian: P(hypothesis | data). Frequentist: P(data | hypothesis). The posterior gives a direct probability statement about the parameter; the p-value gives a probability statement about the data under a hypothetical world.

---

## 6. Set-theoretic and Boolean methods

### 6.1 Crisp-set Qualitative Comparative Analysis (csQCA)

QCA, introduced by Ragin [15], treats the binary data matrix as a truth table and asks set-theoretic questions: *Is condition X sufficient for outcome Y?* (X ⊆ Y in set notation). *Is it necessary?* (Y ⊆ X).

**Core metrics** (using the 2×2 cell notation):

| Metric | Formula | Probabilistic equivalent |
|---|---|---|
| Consistency of sufficiency | a / (a + b) | P(Y ∣ X) = confidence(X → Y) |
| Coverage of sufficiency | a / (a + c) | P(X ∣ Y) = confidence(Y → X) |
| Consistency of necessity | a / (a + c) | P(X ∣ Y) |
| Coverage of necessity | a / (a + b) | P(Y ∣ X) |

Standard thresholds: consistency ≥ 0.80 (lenient) or ≥ 0.90 (strict) [17].

**Consistency IS conditional probability.** This is the critical bridge: QCA's consistency of sufficiency is mathematically identical to confidence in association-rule mining and to P(Y | X) from the Bayesian 2×2 analysis. What QCA adds is not a different number but a different *interpretation* and a powerful *Boolean minimization* step.

**Boolean minimization via the Quine–McCluskey algorithm** [15, 34]:

1. List all truth-table rows where Y = 1 (the "minterms").
2. Group by Hamming weight (number of 1-bits).
3. Merge adjacent-group pairs differing in exactly one bit; replace the differing position with "–" (don't-care).
4. Iterate until no further merges are possible. Unmerged terms are **prime implicants**.
5. Use a prime-implicant chart (with Petrick's method if needed) to find a minimum cover.

**Logical remainders.** With k conditions and small n, many of the 2^k configurations are unobserved. QCA handles these "limited diversity" cases via three solution types [16]:

- **Conservative (complex):** unobserved configurations assumed to NOT produce the outcome.
- **Parsimonious:** unobserved configurations assumed to produce whatever simplifies the formula most.
- **Intermediate:** uses theory-driven directional expectations.

**What QCA reveals beyond probabilistic measures:**

- **Equifinality.** The outcome may have multiple distinct "recipes": Y = (A · B · ¬C) + (D · E). Each disjunct is a separate sufficient path — something a single regression coefficient cannot express.
- **Conjunctural causation.** Individual conditions may matter only in combination.
- **Causal asymmetry.** The causes of Y = 1 may differ entirely from the causes of Y = 0.

**What QCA misses.** QCA has no built-in uncertainty quantification. With n = 13, a consistency of 0.85 (say, 6/7) is compatible with wide uncertainty. The Bayesian 2×2 posterior gives the missing uncertainty band.

**Python:** `scpQCA` (PyPI). R: `QCA` package [34].

### 6.2 Association rule mining

Association-rule metrics are all arithmetical functions of the same 2×2 cell counts. Below, P(X) = (a+b)/N, P(Y) = (a+c)/N, P(X∩Y) = a/N.

| Measure | Formula (cell counts) | Range | Null-invariant? | Symmetric? |
|---|---|---|---|---|
| **Support** | a / N | [0, 1] | No | Yes |
| **Confidence** | a / (a+b) | [0, 1] | Yes | No |
| **Lift** | N·a / [(a+b)(a+c)] | [0, ∞) | No | Yes |
| **Conviction** | (a+b)(c+d) / (N·b) | [0, ∞] | No | No |
| **Leverage** | a/N − (a+b)(a+c)/N² | [−0.25, 0.25] | No | Yes |
| **Zhang's** | leverage / max[a(b+d)/N², (a+b)c/N²] | [−1, 1] | No | No |
| **Kulczynski** | ½[a/(a+b) + a/(a+c)] | [0, 1] | Yes | Yes |
| **Imbalance ratio** | \|b−c\| / (a+b+c) | [0, 1] | Yes | Yes |
| **Cosine (Ochiai)** | a / √[(a+b)(a+c)] | [0, 1] | Yes | Yes |
| **Jaccard** | a / (a+b+c) | [0, 1] | Yes | Yes |
| **All-confidence** | a / max(a+b, a+c) | [0, 1] | Yes | Yes |

**Key relationships** [19, 36]:

- **Confidence = QCA consistency of sufficiency = P(Y | X)** — the most important bridge.
- **Lift = consistency / P(Y)** — how much sufficiency exceeds the base rate.
- **Cosine = geometric mean of the two directional confidences** = √[P(Y|X) · P(X|Y)].
- **Kulczynski = arithmetic mean of the two directional confidences.** When paired with the imbalance ratio, it is recommended by Tan, Kumar and Steinbach [19] as the most robust measure for pattern evaluation.
- **Null-invariant measures** (cosine, Jaccard, Kulczynski, all-confidence) are insensitive to the number of cases where neither X nor Y is present. These are preferable when *d* is large (the common case with rare conditions).

**Python:** `mlxtend.frequent_patterns.association_rules()` implements support, confidence, lift, leverage, conviction, and Zhang's metric.

---

## 7. Extension to n-tuple conditions

### 7.1 The sparsity wall

With k binary conditions, the full joint-configuration space has 2^k cells. The expected number of occupied cells with n observations is [26]:

$$E[\text{occupied}] = 2^k\left[1 - \left(1 - 2^{-k}\right)^n\right] \;\approx\; n \quad\text{when } n \ll 2^k$$

| k | 2^k | Occupied (n = 13) | Occupied (n = 50) |
|---|---|---|---|
| 5 | 32 | ~11 | ~29 |
| 10 | 1 024 | ~13 | ~49 |
| 15 | 32 768 | ~13 | ~50 |
| 20 | 1 048 576 | ~13 | ~50 |

**What can be estimated** reliably: marginal 2×2 tables (k of them, each using all n observations) and pairwise tables (k-choose-2 of them). **What cannot** without structural assumptions: any joint distribution of order 3 or higher when n < 50 and k > 10.

The only escape from the sparsity wall is **structural assumptions that reduce the effective parameter count** below n.

### 7.2 Naïve Bayes: conditional independence as a bridge

The strongest such assumption is **conditional independence** of features given the outcome [24, 25]:

$$P(X_1, \ldots, X_k \mid Z = z) = \prod_{i=1}^k P(X_i \mid Z = z)$$

This reduces the parameter count from 2^k − 1 per class to k per class. Define θ_{iz} = P(X_i = 1 | Z = z). Then:

$$P(X_i = x_i \mid Z = z) = \theta_{iz}^{x_i}\,(1 - \theta_{iz})^{1 - x_i}$$

Applying Bayes' theorem and taking the log-odds:

$$\log\frac{P(Z=1 \mid \mathbf{x})}{P(Z=0 \mid \mathbf{x})} = \underbrace{\log\frac{P(Z=1)}{P(Z=0)} + \sum_{i=1}^k \log\frac{1-\theta_{i1}}{1-\theta_{i0}}}_{\beta_0} + \sum_{i=1}^k x_i \underbrace{\log\frac{\theta_{i1}(1-\theta_{i0})}{\theta_{i0}(1-\theta_{i1})}}_{\beta_i}$$

**This is a linear logistic model.** Each coefficient β_i is the log conditional odds ratio from the marginal 2×2 table for (X_i, Z) — exactly the quantity analyzed in Section 1.

**Connection to the Bayesian 2×2 analysis.** Each θ_{iz} has a Beta posterior from the marginal table: θ_{iz} | data ~ Beta(α + n_{iz}, β + m_z − n_{iz}). Full posterior-predictive inference propagates uncertainty from all k marginal posteriors through the Naïve Bayes formula simultaneously.

**Laplace smoothing** — adding 1 to each count: θ̂_{iz} = (n_{iz} + 1)/(m_z + 2) — is the posterior mean under a uniform Beta(1, 1) prior. This is essential when n is small and some cells are zero.

**When conditional independence fails.** If features X₁ and X₂ are positively correlated given Z, Naïve Bayes double-counts their evidence. The resulting posterior probabilities are miscalibrated (too extreme), though the **ranking of cases by P(Z = 1 | x) is often still correct** [24]. With n = 13 and k = 20, this robustness to model misspecification is a major practical advantage.

### 7.3 Log-linear models for sparse tables

For a three-way binary table X₁ × X₂ × Z, the log-linear model for expected cell count m_{ijk} is [1, 26]:

$$\log m_{ijk} = \mu + \lambda_i^{X_1} + \lambda_j^{X_2} + \lambda_k^Z + \lambda_{ij}^{X_1X_2} + \lambda_{ik}^{X_1Z} + \lambda_{jk}^{X_2Z} + \lambda_{ijk}^{X_1X_2Z}$$

The nested model hierarchy:

| Model | Terms | Free params (2×2×2) | Interpretation |
|---|---|---|---|
| Complete independence | Main effects only | 4 | No associations |
| Conditional independence X₁⊥X₂ ∣ Z | + λ^{X₁Z} + λ^{X₂Z} | 6 | Naïve Bayes equivalent |
| Homogeneous association | + λ^{X₁X₂} | 7 | All pairwise, no 3-way |
| Saturated | + λ^{X₁X₂Z} | 8 | Perfect fit (0 df) |

**Connection to logistic regression.** Conditioning on the Z-margin of the conditional-independence log-linear model yields logistic regression of Z on (X₁, X₂) with no interaction — which is mathematically equivalent to Naïve Bayes.

**Small-sample problems.** ML estimates may not exist when cells contain zeros (the MLE lies on the boundary). The G² and χ² goodness-of-fit statistics are poorly approximated by chi-squared when expected counts fall below 5 [1]. Solutions include MCMC exact tests (Diaconis and Sturmfels) and Bayesian log-linear models with proper priors.

### 7.4 Penalized and regularized approaches

When k > n, some form of regularization is mandatory.

**LASSO (L1 penalty)** [20]:

$$\hat{\beta} = \arg\min_\beta \left\{\frac{1}{2n}\|Y - X\beta\|_2^2 + \lambda\|\beta\|_1\right\}$$

With n = 13 and k = 20, LASSO selects at most min(n, k) = 13 non-zero coefficients. Cross-validation is unreliable with so few cases; information criteria (BIC) or **stability selection** are preferable.

**Stability selection** [23]:

1. Resample half the data B times (e.g., B = 100).
2. Run LASSO on each subsample.
3. Compute selection probability Π̂_j = (1/B) Σ I[β̂_j^(b) ≠ 0].
4. Select variables with Π̂_j > π_thr (e.g., 0.6).

Error bound: E[|false selections|] ≤ q²/[(2π_thr − 1)p], where q is the average number selected per subsample.

**Bayesian shrinkage priors** provide full uncertainty quantification, which frequentist penalization does not.

*Horseshoe prior* [21]:

$$\beta_j \mid \lambda_j, \tau \sim N(0,\;\lambda_j^2\tau^2), \qquad \lambda_j \sim C^+(0,1), \qquad \tau \sim C^+(0,1)$$

The marginal prior on β_j has an infinitely tall spike at zero (aggressively shrinks noise) and Cauchy-like tails (preserves large signals). The posterior mean is E[β_j | y] = (1 − κ̂_j) y_j where κ̂_j ∈ [0, 1] is a data-adaptive shrinkage factor.

*Spike-and-slab prior* [22]:

$$\beta_j \mid \gamma_j \sim (1-\gamma_j)\,\delta_0 + \gamma_j\,N(0,\sigma_{\text{slab}}^2), \qquad \gamma_j \sim \text{Bernoulli}(\pi)$$

This directly models variable inclusion/exclusion and produces **posterior inclusion probabilities** P(γ_j = 1 | data) — the Bayesian analog of a variable-importance ranking.

**Practical guidance by sample size:**

| n | Recommended approach |
|---|---|
| 13–20 | Marginal 2×2 Bayesian analysis for each feature. Naïve Bayes as a classifier. Bayesian shrinkage (horseshoe/spike-and-slab) if joint modeling is needed. |
| 20–30 | Add LASSO/elastic net for exploratory selection; use stability selection. Consider pairwise interactions only with strong prior justification. |
| 30–50 | Penalized logistic regression becomes more viable. Low-order log-linear models for k ≤ 5–6 conditions. |

**Python:** `scikit-learn` (BernoulliNB, LogisticRegression with L1/L2), `PyMC` (horseshoe, spike-and-slab).

---

## 8. Prior elicitation for Bayesian components

### 8.1 Converting user inputs to Beta(α, β) parameters

**Method 1 — Moment matching (mean μ and variance σ²):**

$$\nu = \frac{\mu(1-\mu)}{\sigma^2} - 1, \qquad \alpha = \mu\,\nu, \qquad \beta = (1-\mu)\,\nu$$

Constraint: σ² < μ(1 − μ); otherwise no Beta solution exists.

**Method 1b — Mean μ and equivalent sample size n₀:**

$$\alpha = \mu \cdot n_0, \qquad \beta = (1 - \mu) \cdot n_0$$

**Method 1c — Mode ω and concentration κ = α + β** (requires κ > 2):

$$\alpha = \omega(\kappa - 2) + 1, \qquad \beta = (1 - \omega)(\kappa - 2) + 1$$

**Method 2 — Quantile matching** (user specifies a 90% credible interval [L, U]):

Solve the system F_Beta(L; α, β) = 0.05 and F_Beta(U; α, β) = 0.95 numerically:

```python
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize
import numpy as np

def beta_from_quantiles(q1, p1, q2, p2):
    """Find Beta(a,b) matching P(θ<q1)=p1, P(θ<q2)=p2."""
    def objective(log_ab):
        a, b = np.exp(log_ab)
        r1 = beta_dist.cdf(q1, a, b) - p1
        r2 = beta_dist.cdf(q2, a, b) - p2
        return r1**2 + r2**2
    res = minimize(objective, [np.log(2), np.log(2)], method='Nelder-Mead')
    return np.exp(res.x)  # (alpha, beta)
```

**Method 3 — Maximum-entropy matching** (implemented in PreliZ [38]):

Maximize the Beta entropy H = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β) subject to a constraint on the quantile range. This yields the "least opinionated" prior consistent with the user's stated interval.

### 8.2 The "prior sample size" interpretation

For Beta(α, β), the sum **n₀ = α + β is the equivalent number of prior observations**: α "prior successes" and β "prior failures." After observing s successes and f failures in n trials:

$$E[\theta \mid \text{data}] = \underbrace{\frac{n_0}{n_0 + n}}_{w_{\text{prior}}}\;\frac{\alpha}{\alpha+\beta} + \underbrace{\frac{n}{n_0 + n}}_{w_{\text{data}}}\;\frac{s}{n}$$

The prior weight w = n₀/(n₀ + n) governs how much the prior influences the posterior.

**Concrete examples at n = 13:**

| Prior | n₀ | Prior weight | Interpretation |
|---|---|---|---|
| Jeffreys Beta(0.5, 0.5) | 1 | 7% | Near-negligible |
| Uniform Beta(1, 1) | 2 | 13% | Mild influence |
| Informative Beta(5, 5) | 10 | 43% | Substantial |
| Strong Beta(20, 20) | 40 | 75% | Prior dominates |

**Intuition for non-statisticians.** "Think of the prior as imaginary data. Beta(2, 8) is like having already seen 10 imaginary patients — 2 responders and 8 non-responders — before any real data arrives. With 13 real patients, you combine 10 imaginary + 13 real = 23 total 'observations.' The more imaginary data, the harder it is for real data to change your mind."

**Observations needed to overwhelm the prior.** For the prior weight to drop below ε:

$$n^* = n_0 \cdot \frac{1 - \varepsilon}{\varepsilon}$$

At ε = 0.05: n* = 19 · n₀. For Beta(5, 5): need n > 190.

### 8.3 Sensitivity analysis

**Multi-prior comparison.** The simplest and most effective approach for an interactive tool: compute the posterior under a battery of priors (Haldane, Jeffreys, uniform, weakly informative, skeptical, enthusiastic, user-elicited) and overlay the resulting posterior PDFs. If all posteriors nearly coincide, the conclusion is robust to prior choice. If they diverge, the user's data is too thin to overcome prior assumptions.

**Prior-data conflict detection.** Evans and Moshonov [29] propose checking whether the observed data is "surprising" under the prior predictive distribution. For the Beta–Binomial model, the prior predictive is:

$$P(S = s \mid n, \alpha, \beta) = \binom{n}{s}\frac{B(\alpha+s,\;\beta+n-s)}{B(\alpha,\beta)}$$

Compute the tail probability: conflict = P(m(S) ≤ m(s_obs)), where m(·) is the prior predictive PMF. A small value (< 0.05) signals that the prior and the data disagree strongly — the user should reconsider their prior or their data.

**Visualization strategies for an interactive tool [31]:**

- **Prior vs. posterior overlay.** Plot both PDFs on the same axes with shaded 95% credible intervals. The visual gap between them shows data influence.
- **Prior-strength slider.** Fix the prior mean; let the user drag a slider for n₀ = α + β from 0.1 to 100. Show real-time posterior updating and a "prior weight" gauge.
- **Sensitivity tornado plot.** Bar chart showing how the posterior mean shifts when each prior assumption is varied. Longest bars = most influential assumptions.
- **Prior predictive check.** Simulate M datasets from the prior predictive; plot histogram of the summary statistic; mark the observed value. If it falls in the tails, flag the conflict.
- **"How many more observations?" calculator.** Display n* = n₀(1−ε)/ε as a concrete number the user can act on.

---

## References

[1] [Agresti A. *Categorical Data Analysis*. 3rd ed. Wiley; 2013.](https://onlinelibrary.wiley.com/doi/book/10.1002/0471249688)

[2] [Gunel E, Dickey J. Bayes factors for independence in contingency tables. *Biometrika*. 1974;61(3):545–557.](https://doi.org/10.1093/biomet/61.3.545)

[3] [Jamil T, Ly A, Morey RD, Love J, Marsman M, Wagenmakers E-J. Default "Gunel and Dickey" Bayes factors for contingency tables. *Behav Res Methods*. 2017;49(2):638–652.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5405059/)

[4] [Howard JV. The 2×2 table: a discussion from a Bayesian viewpoint. *Stat Sci*. 1998;13(4):351–367.](https://projecteuclid.org/journals/statistical-science/volume-13/issue-4/The-2times2-table-a-discussion-from-a-Bayesian-viewpoint/10.1214/ss/1028905830.full)

[5] [Agresti A, Hitchcock DB. Bayesian inference for categorical data analysis. *Stat Methods Appl*. 2005;14:297–330.](https://users.stat.ufl.edu/~aa/cda/bayes.pdf)

[6] Fisher RA. *The Design of Experiments*. Edinburgh: Oliver and Boyd; 1935.

[7] [Barnard GA. A new test for 2×2 tables. *Nature*. 1945;156:177.](https://www.nature.com/articles/156177a0)

[8] Lancaster HO. Significance tests in discrete distributions. *J Am Stat Assoc*. 1961;56(294):223–234.

[9] [Agresti A, Caffo B. Simple and effective confidence intervals for proportions and differences of proportions result from adding two successes and two failures. *Am Stat*. 2000;54(4):280–288.](https://users.stat.ufl.edu/~aa/articles/agresti_caffo_2000.pdf)

[10] [Newcombe RG. Interval estimation for the difference between independent proportions: comparison of eleven methods. *Stat Med*. 1998;17(8):873–890.](https://pubmed.ncbi.nlm.nih.gov/9595617/)

[11] [Baptista J, Pike MC. Exact two-sided confidence limits for the odds ratio in a 2×2 table. *J R Stat Soc Ser C*. 1977;26(2):214–220.](https://www.jstor.org/stable/2347041)

[12] Koopman PAR. Confidence intervals for the ratio of two binomial proportions. *Biometrics*. 1984;40(2):513–517.

[13] Cornfield J. A statistical problem arising from retrospective studies. *Proc Third Berkeley Symp Math Stat Probab*. 1956;4:135–148.

[14] [Fagerland MW, Lydersen S, Laake P. Recommended confidence intervals for two independent binomial proportions. *Stat Methods Med Res*. 2015;24(2):224–254.](https://journals.sagepub.com/doi/10.1177/0962280211415469)

[15] [Ragin CC. *The Comparative Method*. Berkeley: University of California Press; 1987.](https://www.ucpress.edu/books/the-comparative-method/paper)

[16] [Ragin CC. *Redesigning Social Inquiry*. Chicago: University of Chicago Press; 2008.](https://press.uchicago.edu/ucp/books/book/chicago/R/bo5973952.html)

[17] [Schneider CQ, Wagemann C. *Set-Theoretic Methods for the Social Sciences*. Cambridge: Cambridge University Press; 2012.](https://www.cambridge.org/core/books/settheoretic-methods-for-the-social-sciences/)

[18] [Agrawal R, Imielinski T, Swami A. Mining association rules between sets of items in large databases. *Proc ACM SIGMOD*. 1993:207–216.](https://dl.acm.org/doi/10.1145/170036.170072)

[19] [Tan P-N, Kumar V, Srivastava J. Selecting the right objective measure for association analysis. *Inf Syst*. 2004;29(4):293–313.](https://dl.acm.org/doi/10.1145/775047.775053)

[20] [Tibshirani R. Regression shrinkage and selection via the lasso. *J R Stat Soc Ser B*. 1996;58(1):267–288.](https://academic.oup.com/jrsssb/article/58/1/267/7027929)

[21] [Carvalho CM, Polson NG, Scott JG. The horseshoe estimator for sparse signals. *Biometrika*. 2010;97(2):465–480.](https://academic.oup.com/biomet/article-abstract/97/2/465/219397)

[22] [Mitchell TJ, Beauchamp JJ. Bayesian variable selection in linear regression. *J Am Stat Assoc*. 1988;83(404):1023–1032.](https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478694)

[23] [Meinshausen N, Bühlmann P. Stability selection. *J R Stat Soc Ser B*. 2010;72(4):417–473.](https://arxiv.org/abs/0809.2932)

[24] [Hand DJ, Yu K. Idiot's Bayes — not so stupid after all? *Int Stat Rev*. 2001;69(3):385–398.](https://doi.org/10.1111/j.1751-5823.2001.tb00465.x)

[25] [Ng AY, Jordan MI. On discriminative vs. generative classifiers: a comparison of logistic regression and naive Bayes. *NIPS*. 2002.](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)

[26] [Bishop YMM, Fienberg SE, Holland PW. *Discrete Multivariate Analysis*. MIT Press; 1975 (reprinted Springer 2007).](https://link.springer.com/book/10.1007/978-0-387-72806-3)

[27] [Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB. *Bayesian Data Analysis*. 3rd ed. Chapman and Hall/CRC; 2013.](http://www.stat.columbia.edu/~gelman/book/)

[28] [O'Hagan A, Buck CE, Daneshkhah A, et al. *Uncertain Judgements: Eliciting Experts' Probabilities*. Wiley; 2006.](https://onlinelibrary.wiley.com/doi/book/10.1002/0470033312)

[29] [Evans M, Moshonov H. Checking for prior-data conflict. *Bayesian Anal*. 2006;1(4):893–914.](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Checking-for-prior-data-conflict/10.1214/06-BA129.full)

[30] [Box GEP. Sampling and Bayes' inference in scientific modelling and robustness. *J R Stat Soc Ser A*. 1980;143(4):383–430.](https://academic.oup.com/jrsssa/article/143/4/383/7105478)

[31] [Gabry J, Simpson D, Vehtari A, Betancourt M, Gelman A. Visualization in Bayesian workflow. *J R Stat Soc Ser A*. 2019;182(2):389–402.](https://doi.org/10.1111/rssa.12378)

[32] [Kass RE, Raftery AE. Bayes factors. *J Am Stat Assoc*. 1995;90(430):773–795.](https://doi.org/10.1080/01621459.1995.10476572)

[33] Jeffreys H. *Theory of Probability*. 3rd ed. Oxford University Press; 1961.

[34] [Dusa A. *QCA with R: A Comprehensive Resource*. Springer; 2019.](https://bookdown.org/dusadrian/QCAbook/)

[35] [Ragin CC. Set relations in social research: evaluating their consistency and coverage. *Polit Anal*. 2006;14(3):291–310.](https://doi.org/10.1093/pan/mpj019)

[36] [Geng L, Hamilton HJ. Interestingness measures for data mining: a survey. *ACM Comput Surv*. 2006;38(3):Article 9.](https://dl.acm.org/doi/10.1145/1132960.1132963)

[37] [Wu T, Chen Y, Han J. Re-examination of interestingness measures in pattern mining. *Data Min Knowl Discov*. 2010;21(3):371–397.](https://doi.org/10.1007/s10618-009-0161-2)

[38] [Icazatti A, et al. PreliZ: a tool-box for prior elicitation. *JOSS*. 2023;8(89):5499.](https://doi.org/10.21105/joss.05499)

[39] Mehta CR, Senchaudhuri P. Conditional versus unconditional exact tests for comparing two binomials. Cytel Software Corporation; 2003.

[40] Berger JO, Sellke T. Testing a point null hypothesis: the irreconcilability of p values and evidence. *J Am Stat Assoc*. 1987;82(397):112–122.

[41] [Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. *Stat Sci*. 2001;16(2):101–133.](https://doi.org/10.1214/ss/1009213286)

[42] [Agresti A, Min Y. Unconditional small-sample confidence intervals for the odds ratio. *Biostatistics*. 2002;3(3):379–386.](https://users.stat.ufl.edu/~aa/articles/agresti_min_2002.pdf)

[43] [Hahsler M. A probabilistic comparison of commonly used interest measures for association rules. 2015 (updated 2024).](https://mhahsler.github.io/arules/docs/measures)
