# Bayesian prior elicitation for Beta distributions: a practical survey

**Thor Whalen**

**The single most important finding for practitioners building data exploration tools is that non-statisticians elicit more accurate priors when asked about observable quantities—"How many out of 20 would you expect?"—rather than abstract parameters or distribution shapes.** This survey covers four families of elicitation methods, sensitivity analysis approaches, UX design evidence, and production-ready libraries across Python, R, and JavaScript. The practical landscape has matured considerably: PreliZ (Python) and SHELF (R) now provide end-to-end elicitation workflows, while jStat offers the computational building blocks for browser-based implementations. For the specific use case of a data exploration tool where domain experts specify beliefs about binary proportions P(Y|X) combined with small-sample data (n=10–50), the choice of elicitation method and default prior will measurably affect posterior inference—making both UX and statistical design decisions consequential.

---

## Section 1: Four methods for getting Beta priors out of human heads

### 1a. Quantile-based elicitation

The user states something like "I'm 90% sure P is between 0.2 and 0.6," and the system solves for Beta(α, β) such that the CDF matches those constraints. Formally, given quantile pairs (q₁, p₁) and (q₂, p₂), we need:

```
F_Beta(q₁; α, β) = p₁
F_Beta(q₂; α, β) = p₂
```

Since the Beta CDF has no closed-form inverse in both parameters simultaneously, **numerical optimization is required**. Van Dorp and Mazzuchi proved existence of a solution for any valid pair of quantile constraints [1], and subsequent work confirmed uniqueness [2].

Three algorithmic approaches dominate the literature. **Least-squares minimization**—minimizing Σ[F_Beta(qᵢ; α, β) − pᵢ]² over (α, β)—is the workhorse used by SHELF, rriskDistributions, and most custom implementations. **Root-finding** treats the two CDF equations as a system and solves via `fsolve` or Newton's method. **Maximum entropy fitting** finds the least-informative Beta distribution satisfying the constraints, implemented in PreliZ's `maxent()` function [3].

**SHELF** (Sheffield Elicitation Framework) handles quantile-to-Beta fitting through its `fitdist()` function, which accepts elicited value-probability pairs, derives starting values via normal approximation, then runs `optim()` to minimize the sum of squared CDF differences [4]. The function fits multiple candidate distributions simultaneously (normal, Student-t, gamma, log-normal, Beta) and reports goodness-of-fit for each:

```r
library(SHELF)
v <- c(0.2, 0.4, 0.6)           # elicited quantile values
p <- c(0.25, 0.5, 0.75)         # cumulative probabilities
fit <- fitdist(vals = v, probs = p, lower = 0, upper = 1)
fit$Beta                          # returns shape1, shape2
plotfit(fit, d = "beta")
```

The **prevalence** R package takes a different approach: its `betaExpert()` function accepts a best guess (mode or mean) plus an uncertainty range, then exploits the relationship β = (α(1−ω) + 2ω − 1)/ω for mode ω to reduce the problem to one-dimensional optimization via `optimize()` [5]. This is based on the methodology of Branscum et al. [6] and the BetaBuster software:

```r
library(prevalence)
betaExpert(best = 0.80, lower = 0.40, upper = 0.90, p = 0.90)
```

The **rriskDistributions** package provides `get.beta.par()`, which fits Beta parameters from 2+ quantiles using L-BFGS-B optimization with fallback to conjugate gradient [7].

In **Python**, no dedicated elicitation function exists in scipy, but the pattern is well-established using `scipy.optimize`:

```python
from scipy.stats import beta
from scipy.optimize import fsolve

def beta_from_quantiles(q1, p1, q2, p2):
    """Solve for Alpha, Beta given P(X < q1) = p1, P(X < q2) = p2."""
    def equations(params):
        a, b = params
        return [beta.cdf(q1, a, b) - p1, beta.cdf(q2, a, b) - p2]
    return fsolve(equations, [2, 2])

# 90% credible interval [0.2, 0.6]
alpha, beta_param = beta_from_quantiles(0.2, 0.05, 0.6, 0.95)
```

**PreliZ** offers the cleanest Python API via maximum-entropy fitting [3]:

```python
import preliz as pz
dist = pz.Beta()
pz.maxent(dist, 0.2, 0.6, 0.90)   # 90% mass in [0.2, 0.6]
print(dist.alpha, dist.beta)
```

In **JavaScript**, jStat provides `jStat.beta.cdf(x, α, β)` and `jStat.beta.inv(p, α, β)` but no built-in fitting [8]. Production implementations pair jStat's CDF with a minimizer such as `numeric.js` or `ml-levenberg-marquardt`:

```javascript
const jStat = require('jstat');
function fitBetaFromQuantiles(x1, p1, x2, p2) {
    let bestA = 1, bestB = 1, bestErr = Infinity;
    // Grid search (replace with Nelder-Mead for production)
    for (let a = 0.1; a < 20; a += 0.1) {
        for (let b = 0.1; b < 20; b += 0.1) {
            const err = Math.pow(jStat.beta.cdf(x1, a, b) - p1, 2)
                      + Math.pow(jStat.beta.cdf(x2, a, b) - p2, 2);
            if (err < bestErr) { bestErr = err; bestA = a; bestB = b; }
        }
    }
    return { alpha: bestA, beta: bestB };
}
```

### 1b. Mean plus strength: the concentration parameterization

This method asks the user two things: "What do you think P is?" (prior mean μ) and "How confident are you?" (mapped to concentration κ = α + β). The math is exact and requires **no optimization**:

```
α = μ · κ        β = (1 − μ) · κ
```

Kruschke's widely-used textbook provides the canonical treatment, framing κ as "the number of new observations needed to change your mind" [9]. For a mode parameterization (requiring κ > 2): α = ω(κ − 2) + 1, β = (1 − ω)(κ − 2) + 1. The **effective sample size** (ESS) of Beta(α, β) is α + β, formally established by Morita et al. [10].

No single published scale maps verbal descriptors to κ values, but converging evidence from Kruschke [9], the RBesT clinical trials package [11], and general decision-analysis practice suggests the following practical mapping:

| Verbal descriptor | κ = α + β | Equivalent pseudo-observations |
|---|---|---|
| "Know almost nothing" | 2 | 0 (uniform) |
| "Vague sense" | 5–10 | 3–8 |
| "Moderate confidence" | 10–25 | 8–23 |
| "Quite confident" | 25–50 | 23–48 |
| "Very confident" | 50–100 | ~small study |
| "Near certain" | 100–500+ | ~large study |

The critical UX insight: **ask the expert "How many observations would it take to change your mind?"**—that number approximates κ. Stan, PyMC, and JAGS all support the mean-concentration parameterization natively. PyMC's API is particularly clean: `pm.Beta('theta', mu=0.4, sigma=0.15)` [12].

### 1c. The roulette method: chips, bins, and histograms

The roulette (or "chip-and-bin") method originated with Gore in 1987 [13], was validated by Johnson et al. [14], and formalized within the SHELF framework [4]. The expert receives a grid of **m equally-sized bins** covering [0, 1] and allocates **n chips** (typically 20, each representing 5% probability) across bins to express their belief distribution.

Converting chips to a fitted Beta involves two steps. First, chip counts are converted to cumulative probabilities at bin boundaries: P(X ≤ bᵢ) = Σⱼ₌₁ⁱ cⱼ / Σcⱼ. Second, a Beta distribution is fitted via least-squares minimization of Σᵢ[F_Beta(bᵢ; α, β) − P̂(X ≤ bᵢ)]², using the same `optim()` machinery as quantile-based fitting but with more data points from the histogram [4].

Goldstein and Rothschild demonstrated that **graphical histogram interfaces are substantially more accurate than quantile-based methods** for non-experts [15]. This finding has driven widespread adoption of digital roulette implementations:

- **MATCH** (Morris, Oakley, and Crowe): A web-based tool supporting five elicitation methods including roulette, with multi-user remote sessions and R-based backend fitting [16]. Used by GSK in 50+ clinical trials [17].
- **SHELF's** built-in Shiny app: `elicit(method = "roulette")` with configurable bins and grid height.
- **PreliZ**: `pz.roulette()` provides a Python/Jupyter implementation [3].
- **distBuilder** (Quentin André): A pure JavaScript embeddable "ball-and-bucket" widget, 3 lines of code to add to any web page, based on Goldstein's work [18].
- **drawyourprior** (Johns Hopkins): An R Shiny app where the user literally draws a density curve, and the app fits a Beta distribution [19].

### 1d. Predictive elicitation: judging simulated data instead of parameters

Rather than asking experts about abstract distribution parameters, **predictive elicitation** asks about observable quantities: "If you observed 20 patients, how many would respond?" The expert's predictions are then translated back into prior parameters. Kadane and Wolfson established the foundational principles: only observable quantities should be assessed, experts should provide quantiles or probabilities rather than moments, and feedback must be provided [20].

For a Binomial(n, θ) likelihood with Beta(α, β) prior, the prior predictive distribution is Beta-Binomial. The algorithm matches elicited predictive quantiles to the Beta-Binomial CDF, solving for (α, β). Chaloner et al. pioneered graphical predictive elicitation where experts view simulated datasets and judge plausibility [21].

The most significant modern implementation is **PreliZ**, which offers `pz.predictive_sliders()` for interactive prior-predictive exploration and `pz.ppa()` (Prior Predictive Assistant) for iteratively refining priors [3]. The 2025 PriorWeaver system takes this further, reframing elicitation as "dataset construction"—analysts express beliefs about observable variables and relationships rather than parameters, and the system derives priors automatically [22].

**Pros**: Experts reason about familiar quantities; avoids cognitive difficulty of distribution-shape specification; produces model-consistent priors. **Cons**: Converting predictive information back to parameters can be mathematically complex for non-conjugate models; conflates parameter uncertainty with sampling variability; requires more computational infrastructure; scales poorly to high dimensions [20][23].

---

## Section 2: Sensitivity, skepticism sliders, and sensible defaults

### 2a. Seeing whether the prior or data is driving the posterior

For the Beta-binomial, all three curves—prior, likelihood, and posterior—are available in closed form, making the classic "three-curve overlay" trivially implementable. The prior is Beta(α, β), the normalized likelihood is proportional to Beta(y+1, n−y+1), and the posterior is Beta(α+y, β+n−y). Gabry et al. provide best practices for Bayesian visualization workflows [24].

For formal prior-data conflict diagnostics, several approaches exist. **Box's prior predictive p-value** evaluates whether the observed data is surprising under the prior predictive distribution [25]. **Evans and Moshonov's conflict p-value** computes P(m(y') ≤ m(y_obs)), where m(·) is the prior predictive density [26]. **Nott et al.** use prior-to-posterior KL divergence as a checking statistic, comparing the observed divergence to its prior predictive distribution [27].

The most practical modern tool is the **priorsense** R package by Kallioinen et al., which implements **power-scaling sensitivity analysis** [28]. It raises the prior to a power α—π(θ)^α—and uses importance sampling to efficiently estimate how the posterior changes as α varies, without refitting the model. High prior sensitivity combined with low likelihood sensitivity signals prior-data conflict. This is, functionally, the closest existing implementation to a "skepticism slider."

Interactive tools for visualizing prior influence include **Seeing Theory** (Brown University, D3.js) with its elegant Beta-binomial updating demonstration [29], the **R Psychologist Bayesian visualization** (Kristoffer Magnusson, D3.js) [30], and the **MD Anderson Beta-Binomial updater** Shiny app [31]. The Depaoli et al. Shiny app lets users specify alternative priors and compare resulting posteriors side-by-side [32].

### 2b. A single dial from skeptical to confident

The **power prior framework** (Ibrahim and Chen [33]) provides the mathematical foundation. Given historical data D₀, the power prior is:

```
π(θ | D₀, a₀) ∝ L(θ | D₀)^{a₀} · π₀(θ)     where a₀ ∈ [0, 1]
```

For the Beta-binomial specifically, if historical data has y₀ successes in n₀ trials with initial prior Beta(c₁, c₂), the power prior is Beta(a₀·y₀ + c₁, a₀·(n₀ − y₀) + c₂). Setting a₀ = 0 ignores the historical information entirely; a₀ = 1 fully incorporates it. Ibrahim et al. provide the definitive review [34].

**Repurposing for a UI skepticism slider** without historical data is straightforward. Given any user-specified informative prior π_info(θ) and a flat reference prior, define:

```
π_skeptic(θ; a₀) ∝ π_info(θ)^{a₀} · π_flat(θ)^{1−a₀}
```

For Beta priors, this yields **Beta(a₀·(α_info − 1) + 1, a₀·(β_info − 1) + 1)**, cleanly interpolating between the informative prior (a₀ = 1) and Beta(1, 1) (a₀ = 0).

An even simpler approach uses the **concentration parameterization directly**: Beta(κ·μ₀, κ·(1 − μ₀)) where μ₀ is the prior mean and κ is the slider value. κ = 2 gives a uniform prior; κ = 4 gives Beta(2, 2); κ = 50 gives a strongly informative prior. This is arguably the most implementation-friendly "skepticism slider."

The **unit information prior** (Kass and Wasserman [35]) provides a principled baseline: a prior carrying exactly one observation's worth of information. For Bernoulli data at θ = 0.5, this corresponds roughly to Beta(1, 1) or Beta(2, 2) depending on centering convention. Related frameworks include Hobbs et al.'s **commensurate prior** [36] and Zellner's **g-prior** family, both offering single-parameter informativeness control.

R packages implementing power priors include **hdbayes** (Stan-based, supports power prior, normalized power prior, and commensurate prior for GLMs), **NPP** (normalized power prior for Bernoulli, Normal, Poisson), **bayesDP** (discount power prior with automatic a₀ selection), and **BayesPPD** (power prior for sample size determination) [37].

### 2c. What to do when the user just wants a default

For the Beta-binomial with **n = 10–50**, the prior will noticeably influence the posterior. There is no free lunch, but there is a near-consensus recommendation.

**Beta(1, 1) — the uniform/Bayes-Laplace prior** — is the strongest candidate for a default. ESS = 2 pseudo-observations, the posterior mean follows Laplace's rule of succession (y+1)/(n+2), and the posterior mode equals the MLE. Tuyl, Gerlach, and Mengersen argued extensively for Beta(1, 1) as the "consensus prior," showing it is well-behaved for zero events and produces honest uncertainty [38][39]. It conforms with the frequentist "Rule of Three" (3/n as an approximate 95% upper bound when y = 0). Its simplicity is a major UX advantage.

**Beta(0.5, 0.5) — Jeffreys' prior** — is reparameterization-invariant and coincides with the reference prior [40]. Its ESS is just 1. However, its **U-shape** concentrates mass at 0 and 1, which is counterintuitive for many applications and problematic for zero events, where it pulls the posterior further toward zero than warranted [38].

**Beta(2, 2)** provides mild regularization toward 0.5 with ESS = 4, has zero density at the boundaries (preventing degenerate estimates), and connects to the Agresti-Coull confidence interval, which has excellent frequentist coverage properties even for small n [41]. The Stan community's prior choice recommendations favor weakly informative priors that "rule out unreasonable parameter values but are not so strong as to rule out values that might make sense" [42]—Beta(2, 2) fits this description well.

**Practical recommendation for implementers**: Use **Beta(1, 1) as the default** for maximum transparency and minimal bias. Offer **Beta(2, 2)** as a "regularized" alternative when boundary estimates are undesirable. Always display the prior's effective sample size to make its influence transparent. For n = 10–50, **always conduct and display sensitivity analysis**—the prior will matter.

---

## Section 3: Designing elicitation interfaces that non-statisticians can actually use

### 3a. What user studies tell us about elicitation accuracy

The most directly relevant empirical study is **Sarma and Kay (2020)**, who tested interactive prior-setting visualizations with 50 Bayesian practitioners [43]. They identified three strategies users naturally employ: *centrality matching* (setting the center), *interval matching* (setting credible intervals), and *visual mass allocation* (dragging density mass). The key design implication: interfaces should first establish which strategy the user prefers, then guide accordingly.

**Goldstein and Rothschild (2014)** demonstrated that graphical histogram-based methods are **substantially more accurate** than quantile-based elicitation for non-experts [15]. This single finding has arguably had the largest impact on elicitation UI design. **Hullman et al. (2023)** showed that even lightweight belief elicitation before data analysis led to **21% more correct inferences and 12% fewer false discoveries** [44]—motivating elicitation not just for better priors but for better analysis overall.

The cognitive bias literature identifies several threats to elicitation accuracy. **Partition dependence** (Fox and Clemen [45]) is the most insidious for histogram/roulette methods: assessed probabilities are systematically biased toward a uniform distribution over the presented bins, meaning the number and placement of bins directly bias results. **Anchoring** causes initial values to disproportionately influence final judgments. **Overconfidence** leads to credible intervals that are too narrow. Bojke et al.'s NIHR protocol recommends **eliciting extremes before central values** to mitigate anchoring, and separating lower/upper bound elicitation from median elicitation [46]. The MOLE (More-or-Less Elicitation) method by Welsh et al. presents choices rather than requesting direct estimates, mitigating both anchoring and overconfidence [47].

Kynn confirmed that **elicitation based on counts is less prone to cognitive errors** than elicitation of probabilities directly [48]—supporting the frequency-framing approach ("How many out of 20?") over probability framing ("What percentage?").

### 3b. SHELF as the gold standard protocol

The **Sheffield Elicitation Framework** (SHELF), created by Tony O'Hagan and Jeremy Oakley, is the most widely adopted structured elicitation protocol [4][49]. Its seven steps are: (1) evidence dossier preparation, (2) briefing and bias training, (3) individual judgments on paper, (4) sharing and facilitated discussion, (5) group consensus, (6) parametric distribution fitting, and (7) feedback and iteration.

SHELF's key design principles—**private initial judgments** to prevent groupthink, behavioral aggregation through facilitated discussion rather than mathematical pooling, and mandatory feedback loops—are directly transferable to digital single-expert interfaces. GSK has used SHELF in over 50 clinical trials, with Dallow, Best, and Montague documenting the process [17]. The European Food Safety Authority references SHELF in its official expert elicitation guidance [50].

The SHELF R package (v1.12.1, Jeremy Oakley) provides both programmatic functions (`fitdist()`, `plotfit()`, `feedback()`, `linearpool()`) and **six Shiny apps** covering single expert, multiple experts, bivariate, Dirichlet, extension methods, and survival extrapolation scenarios [4]. The **MATCH web tool** (Morris, Oakley, and Crowe) wraps SHELF's R backend in a web interface supporting five elicitation techniques with multi-user remote sessions and timestamped audit trails [16].

### 3c. The current landscape of elicitation tools

The most production-ready tools span three ecosystems:

**Python**: PreliZ (ArviZ ecosystem) is the standout, offering `maxent()` for constraint-based fitting, `roulette()` for histogram elicitation, `predictive_sliders()` for prior-predictive exploration, and `plot_interactive()` for ipywidgets-based parameter exploration in Jupyter [3]. The newer **elicito** package (Bockting and Bürkner, 2025) provides a modular simulation-based framework supporting both structural and predictive methods [51]. Several Streamlit apps demonstrate Beta-binomial prior elicitation, including Bayesian A/B testing calculators and an MMM prior elicitation tool [52].

**R**: Beyond SHELF and prevalence, **rriskDistributions** offers GUI-based quantile fitting [7], **LaplacesDemon** includes chip-and-bin elicitation via `elicit()` [53], **vizdraws** provides animated prior-to-posterior transitions [54], and the **PriorElicitation** Shiny app from the Oslo Centre for Biostatistics supports simulation-based experiments [55].

**JavaScript/Web**: The **distBuilder** widget (Quentin André) is the most embeddable JavaScript solution—a pure-JS ball-and-bucket histogram builder requiring just 3 lines to integrate [18]. **Seeing Theory** (Brown University) demonstrates polished D3.js Beta-binomial updating [29]. **Observable notebooks** by Mattias Villani provide interactive D3-based Bayesian learning demonstrations [56]. No dedicated React component library for prior elicitation currently exists, but the standard pattern combines jStat for computation with React for UI and D3/Recharts for visualization. The **Guesstimate** app (React + jStat) demonstrates this architecture for Monte Carlo simulation [57].

### 3d. Design patterns that work

Five UX patterns have proven effective for prior elicitation:

**Range slider for credible intervals.** Two sliders (lower/upper bound) plus a confidence selector ("I'm 90% sure it's between X and Y") map directly to quantile-based fitting. PreliZ's `maxent()` implements the backend; the frontend is a standard double-slider component. This is the most natural entry point for interval-matching users (Sarma and Kay's most common strategy [43]).

**Draggable density curves.** The drawyourprior Shiny app lets users draw a curve directly, fitting a Beta to the shape [19]. Sarma and Kay's interface uses a draggable control point that changes the distribution's mean and SD, with live density updates [43]. PriorWeaver extends this to a "paintbrush" approach where analysts paint expectations into charts [22].

**Chips and roulette (gamification).** The roulette metaphor remains the strongest method for non-experts. Twenty chips on a grid—each representing 5% probability—provides tangible, immediate visual feedback. MATCH, SHELF, the Five-Step Method [58], and distBuilder all implement this digitally. The key UX insight: the chip layout *is* the probability distribution, making the abstract concrete.

**Natural language and observable-space input.** PriorWeaver's core innovation is accepting natural-language statements about observables ("people in their 40s earn $40k–$60k") and deriving priors automatically [22]. Frequency framing—"How many out of 20?"—exploits the well-documented finding that count-based reasoning is less error-prone than probability reasoning [48]. SHELF frames questions as "What's the smallest/largest plausible value?" and "What value is it equally likely to be above as below?"

**Progressive disclosure.** Nielsen Norman Group's principle of revealing complexity gradually is particularly valuable for elicitation. SHELF's protocol is inherently progressive: evidence → individual judgments → sharing → consensus → fitting → feedback. The Five-Step Method decomposes elicitation into location first, then spread [58]. Sarma and Kay recommend that the interface "first attempt to establish the user's desired high-level approach... then provide guidance in what information might be useful" [43]. For a data exploration tool, this means: offer a simple default (Beta(1,1)), then a one-slider "confidence" adjustment, then reveal advanced options (quantile specification, histogram drawing) only on request.

---

## Section 4: Libraries and tools at a glance

### Python ecosystem

| Package | Key functions | Elicitation method | Install |
|---|---|---|---|
| **scipy** | `beta.cdf()`, `beta.ppf()` + `optimize.fsolve()` | Quantile fitting (manual) | `pip install scipy` |
| **PreliZ** | `maxent()`, `roulette()`, `predictive_sliders()`, `ppa()` | MaxEnt, roulette, predictive | `pip install preliz` |
| **PyMC** | `pm.Beta(mu=, sigma=)` or `pm.Beta(alpha=, beta=)` | Three parameterizations | `pip install pymc` |
| **elicited** | `elicitPERT(min, mode, max)` | PERT-Beta | `pip install elicited` |
| **elicito** | Modular simulation-based framework | Structural + predictive | arXiv:2506.16830 |

The canonical Python workflow for quantile-based fitting:

```python
from scipy.stats import beta
from scipy.optimize import fsolve

def fit_beta(q1, p1, q2, p2):
    return fsolve(lambda ab: [beta.cdf(q1, *ab) - p1,
                               beta.cdf(q2, *ab) - p2], [2, 2])

# "90% sure P is between 0.2 and 0.6"
a, b = fit_beta(0.2, 0.05, 0.6, 0.95)
# Bayesian update with 7 successes in 20 trials
a_post, b_post = a + 7, b + (20 - 7)
```

### R ecosystem

| Package | Key functions | Elicitation method | CRAN |
|---|---|---|---|
| **SHELF** | `fitdist()`, `elicit()`, `plotfit()`, `feedback()` | Quantile, roulette, probability | Yes |
| **prevalence** | `betaExpert()`, `betaPERT()` | Mode + percentile | Yes |
| **rriskDistributions** | `get.beta.par()`, `fit.perc()` (GUI) | Quantile matching | Yes |
| **priorsense** | `powerscale_plot_dens()`, `powerscale_sensitivity()` | Sensitivity analysis | Yes |
| **hdbayes** | Power prior, NPP for GLMs | Prior strength control | Yes |
| **RBesT** | `ess()`, `mixbeta()` | ESS, mixture priors | Yes |

### JavaScript ecosystem

| Package | Beta support | Fitting? | Install |
|---|---|---|---|
| **jStat** | Full (PDF, CDF, inv, sample) | No (needs custom optimizer) | `npm install jstat` |
| **simple-statistics** | **None** | No | — |
| **distBuilder** | Histogram widget | Outputs bin counts | Pure JS embed |
| **D3.js** | Visualization only | No | `npm install d3` |

The practical JavaScript gap is real: **no npm package provides Beta-from-quantile fitting out of the box**. The implementation pattern is jStat for CDF evaluation + a minimizer (`numeric.js`, `optimization-js`) for fitting + D3 or a React charting library for visualization. For a React-based data exploration tool, wrapping the scipy approach in a thin Python API (or porting the ~15-line fsolve algorithm to TypeScript with jStat) is the pragmatic path.

### Web tools worth knowing

The **MATCH tool** (University of Sheffield/Nottingham) offers the most complete web-based elicitation experience with five methods, multi-user sessions, and audit trails [16]. **Seeing Theory** (Brown) provides the most polished interactive Bayesian updating visualization [29]. The **MD Anderson Beta-Binomial updater** is a clean Shiny reference implementation [31]. **Observable notebooks** by Villani offer hackable D3-based demonstrations [56].

---

## Conclusion

Three insights emerge from this survey that are not obvious from any single source. First, the **elicitation method matters less than the framing**: asking "How many out of 20?" consistently outperforms "What probability?" across user studies, regardless of whether the backend uses quantile matching, roulette, or predictive methods [15][20][48]. For a data exploration tool, this means the UI should present the count-based framing as the primary path, with distribution-level controls as progressive disclosure.

Second, for small samples (n = 10–50), the **concentration parameterization** (mean + κ) provides the most natural mapping to a "skepticism slider" and requires zero numerical optimization—a significant advantage for browser-based implementations where computational overhead matters. The formula Beta(a₀·(α − 1) + 1, a₀·(β − 1) + 1) cleanly interpolates between any informative prior and a uniform, and can be exposed as a single slider labeled "How much should your belief override the data?"

Third, the **default prior question has a defensible answer**: Beta(1, 1) as the non-informative default, with Beta(2, 2) offered as a regularized alternative that prevents boundary estimates. The key is to always display the prior's effective sample size alongside the data sample size, making the relative influence transparent to the user. When n = 20 and ESS = 2, the user can see that the data dominates; when n = 10 and ESS = 50, the prior's dominance is visible and adjustable. This transparency—more than any particular default choice—is what makes Bayesian updating trustworthy for non-statistician users.

---

**References**

[1] Van Dorp JR, Mazzuchi TA. Solving for the parameters of a Beta distribution under two quantile constraints. J Stat Comput Simul. 2000;67(2):189–201. [https://doi.org/10.1080/00949650008812041](https://doi.org/10.1080/00949650008812041)

[2] Garthwaite PH, Kadane JB, O'Hagan A. Statistical methods for eliciting probability distributions. J Am Stat Assoc. 2005;100(470):680–701. [https://doi.org/10.1198/016214505000000105](https://doi.org/10.1198/016214505000000105)

[3] Icazatti A, Abril-Pla O, Klami A, Martin OA. PreliZ: A tool-box for prior elicitation. J Open Source Softw. 2023;8(89):5499. [https://github.com/arviz-devs/preliz](https://github.com/arviz-devs/preliz)

[4] Oakley J. SHELF: The Sheffield Elicitation Framework (R package). CRAN; 2025. [https://CRAN.R-project.org/package=SHELF](https://CRAN.R-project.org/package=SHELF)

[5] Devleesschauwer B. prevalence: Tools for prevalence assessment studies (R package). CRAN; 2022. [https://CRAN.R-project.org/package=prevalence](https://CRAN.R-project.org/package=prevalence)

[6] Branscum AJ, Gardner IA, Johnson WO. Estimation of diagnostic-test sensitivity and specificity through Bayesian modeling. Prev Vet Med. 2005;68:145–163. [https://doi.org/10.1016/j.prevetmed.2004.12.005](https://doi.org/10.1016/j.prevetmed.2004.12.005)

[7] Belgorodski N, et al. rriskDistributions: Fitting distributions to given data or known quantiles (R package). CRAN; 2017. [https://CRAN.R-project.org/package=rriskDistributions](https://CRAN.R-project.org/package=rriskDistributions)

[8] jStat: JavaScript statistical library. GitHub; 2023. [https://github.com/jstat/jstat](https://github.com/jstat/jstat)

[9] Kruschke JK. Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan. 2nd ed. Academic Press; 2014.

[10] Morita S, Thall PF, Müller P. Determining the effective sample size of a parametric prior. Biometrics. 2008;64(2):595–602. [https://doi.org/10.1111/j.1541-0420.2007.00888.x](https://doi.org/10.1111/j.1541-0420.2007.00888.x)

[11] Weber S, et al. RBesT: R Bayesian Evidence Synthesis Tools (R package). CRAN; 2024. [https://CRAN.R-project.org/package=RBesT](https://CRAN.R-project.org/package=RBesT)

[12] PyMC Developers. PyMC: Bayesian modeling and probabilistic programming in Python. 2024. [https://www.pymc.io](https://www.pymc.io)

[13] Gore SM. Biostatistics and the Medical Research Council. MRC News. 1987;35:19–20.

[14] Johnson SR, et al. Methods to elicit beliefs for Bayesian priors: a systematic review. J Clin Epidemiol. 2010;63(4):355–369. [https://doi.org/10.1016/j.jclinepi.2009.06.003](https://doi.org/10.1016/j.jclinepi.2009.06.003)

[15] Goldstein DG, Rothschild D. Lay understanding of probability distributions. Judgm Decis Mak. 2014;9(1):1–14. [https://journal.sjdm.org/13/131029/jdm131029.html](https://journal.sjdm.org/13/131029/jdm131029.html)

[16] Morris DE, Oakley JE, Crowe JA. A web-based tool for eliciting probability distributions from experts. Environ Model Softw. 2014;52:1–4. [https://doi.org/10.1016/j.envsoft.2013.10.010](https://doi.org/10.1016/j.envsoft.2013.10.010)

[17] Dallow N, Best N, Montague TH. Better decision making in drug development through adoption of formal prior elicitation. Pharm Stat. 2018;17:301–316. [https://doi.org/10.1002/pst.1854](https://doi.org/10.1002/pst.1854)

[18] André Q. distBuilder: A JavaScript distribution builder. [https://quentinandre.net/software/distbuilder/](https://quentinandre.net/software/distbuilder/)

[19] Johns Hopkins Data Science Lab. drawyourprior: Draw your prior (R Shiny app). GitHub; 2020. [https://github.com/jhudsl/drawyourprior](https://github.com/jhudsl/drawyourprior)

[20] Kadane JB, Wolfson LJ. Experiences in elicitation. The Statistician. 1998;47:3–19. [https://doi.org/10.1111/1467-9884.00114](https://doi.org/10.1111/1467-9884.00114)

[21] Chaloner KM, Church T, Louis TA, Matts JP. Graphical elicitation of a prior distribution for a clinical trial. The Statistician. 1993;42:341–353. [https://doi.org/10.2307/2348468](https://doi.org/10.2307/2348468)

[22] PriorWeaver: Prior elicitation via iterative dataset construction. arXiv:2510.06550; 2025. [https://arxiv.org/abs/2510.06550](https://arxiv.org/abs/2510.06550)

[23] Mikkola P, Martin OA, et al. Prior knowledge elicitation: the past, present, and future. Bayesian Anal. 2024;19(4):1129–1161. [https://doi.org/10.1214/23-BA1381](https://doi.org/10.1214/23-BA1381)

[24] Gabry J, Simpson D, Vehtari A, Betancourt M, Gelman A. Visualization in Bayesian workflow. J R Stat Soc A. 2019;182:389–402. [https://doi.org/10.1111/rssa.12378](https://doi.org/10.1111/rssa.12378)

[25] Box GEP. Sampling and Bayes' inference in scientific modelling and robustness. J R Stat Soc A. 1980;143:383–430. [https://doi.org/10.2307/2982063](https://doi.org/10.2307/2982063)

[26] Evans M, Moshonov H. Checking for prior-data conflict. Bayesian Anal. 2006;1(4):893–914. [https://doi.org/10.1214/06-BA129](https://doi.org/10.1214/06-BA129)

[27] Nott DJ, Wang X, Evans M, Englert BG. Checking for prior-data conflict using prior-to-posterior divergences. Stat Sci. 2020;35(2):234–253. [https://doi.org/10.1214/19-STS731](https://doi.org/10.1214/19-STS731)

[28] Kallioinen N, Paananen T, Bürkner PC, Vehtari A. Detecting and diagnosing prior and likelihood sensitivity with power-scaling. Stat Comput. 2024;34:57. [https://n-kall.github.io/priorsense/](https://n-kall.github.io/priorsense/)

[29] Kunin D. Seeing Theory: A visual introduction to probability and statistics. Brown University; 2017. [https://seeing-theory.brown.edu/bayesian-inference/](https://seeing-theory.brown.edu/bayesian-inference/)

[30] Magnusson K. R Psychologist: Understanding Bayesian inference with interactive visualizations. [https://rpsychologist.com/d3/bayes/](https://rpsychologist.com/d3/bayes/)

[31] Lee JJ, Kuo YW. Bayesian update for a Beta-Binomial distribution (Shiny app). MD Anderson Cancer Center. [https://biostatistics.mdanderson.org/shinyapps/BU1BB/](https://biostatistics.mdanderson.org/shinyapps/BU1BB/)

[32] Depaoli S, Winter SD, Visser M. The importance of prior sensitivity analysis in Bayesian statistics: demonstrations using an interactive Shiny app. Front Psychol. 2020;11:608045. [https://doi.org/10.3389/fpsyg.2020.00608](https://doi.org/10.3389/fpsyg.2020.00608)

[33] Chen MH, Ibrahim JG. Power prior distributions for regression models. Stat Sci. 2000;15(1):46–60. [https://doi.org/10.1214/ss/1009212673](https://doi.org/10.1214/ss/1009212673)

[34] Ibrahim JG, Chen MH, Gwon Y, Chen F. The power prior: theory and applications. Stat Med. 2015;34:3724–3749. [https://doi.org/10.1002/sim.6728](https://doi.org/10.1002/sim.6728)

[35] Kass RE, Wasserman LA. A reference Bayesian test for nested hypotheses and its relationship to the Schwarz criterion. J Am Stat Assoc. 1995;90:928–934. [https://doi.org/10.1080/01621459.1995.10476592](https://doi.org/10.1080/01621459.1995.10476592)

[36] Hobbs BP, Carlin BP, Mandrekar SJ, Sargent DJ. Hierarchical commensurate and power prior models for adaptive incorporation of historical information in clinical trials. Biometrics. 2011;67(3):1047–1056. [https://doi.org/10.1111/j.1541-0420.2011.01564.x](https://doi.org/10.1111/j.1541-0420.2011.01564.x)

[37] Ethan Alt, et al. hdbayes: Bayesian analysis of historical data (R package). CRAN; 2024. [https://CRAN.R-project.org/package=hdbayes](https://CRAN.R-project.org/package=hdbayes)

[38] Tuyl F, Gerlach R, Mengersen K. A comparison of Bayes-Laplace, Jeffreys, and other priors: the case of zero events. Am Stat. 2008;62(1):40–44. [https://doi.org/10.1198/000313008X267839](https://doi.org/10.1198/000313008X267839)

[39] Tuyl F, Gerlach R, Mengersen K. Posterior predictive arguments in favor of the Bayes-Laplace prior as the consensus prior for binomial and multinomial parameters. Bayesian Anal. 2009;4(1):151–158. [https://doi.org/10.1214/09-BA405](https://doi.org/10.1214/09-BA405)

[40] Jeffreys H. An invariant form for the prior probability in estimation problems. Proc R Soc A. 1946;186:453–461. [https://doi.org/10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056)

[41] Agresti A, Coull BA. Approximate is better than "exact" for interval estimation of binomial proportions. Am Stat. 1998;52(2):119–126. [https://doi.org/10.1080/00031305.1998.10480550](https://doi.org/10.1080/00031305.1998.10480550)

[42] Stan Development Team. Prior choice recommendations. GitHub wiki; 2024. [https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)

[43] Sarma A, Kay M. Prior setting in practice: strategies and rationales used in choosing prior distributions for Bayesian analysis. In: Proc CHI 2020. ACM; 2020:1–12. [https://doi.org/10.1145/3313831.3376377](https://doi.org/10.1145/3313831.3376377)

[44] Hullman J, et al. Visual belief elicitation reduces false discovery. In: Proc CHI 2023. ACM; 2023. [https://doi.org/10.1145/3544548.3580808](https://doi.org/10.1145/3544548.3580808)

[45] Fox CR, Clemen RT. Subjective probability assessment in decision analysis: partition dependence and bias toward the ignorance prior. Manag Sci. 2005;51(9):1417–1432. [https://doi.org/10.1287/mnsc.1050.0409](https://doi.org/10.1287/mnsc.1050.0409)

[46] Bojke L, et al. Developing a reference protocol for structured expert elicitation in health-care decision-making. Health Technol Assess. 2021;25(37). [https://doi.org/10.3310/hta25370](https://doi.org/10.3310/hta25370)

[47] Welsh MB, et al. More-or-less elicitation (MOLE): reducing bias in range estimation and forecasting. Eur J Oper Res. 2021;295(3):972–985. [https://doi.org/10.1016/j.ejor.2021.04.026](https://doi.org/10.1016/j.ejor.2021.04.026)

[48] Kynn M. The 'heuristics and biases' bias in expert elicitation. J R Stat Soc A. 2008;171(1):239–264. [https://doi.org/10.1111/j.1467-985X.2007.00499.x](https://doi.org/10.1111/j.1467-985X.2007.00499.x)

[49] O'Hagan A, Buck CE, Daneshkhah A, et al. Uncertain Judgements: Eliciting Experts' Probabilities. Chichester: Wiley; 2006.

[50] European Food Safety Authority. Guidance on expert knowledge elicitation in food and feed safety risk assessment. EFSA J. 2014;12(6):3734. [https://doi.org/10.2903/j.efsa.2014.3734](https://doi.org/10.2903/j.efsa.2014.3734)

[51] Bockting F, Bürkner PC. elicito: A Python package for simulation-based prior elicitation. arXiv:2506.16830; 2025. [https://arxiv.org/abs/2506.16830](https://arxiv.org/abs/2506.16830)

[52] Streamlit community. Various Bayesian A/B testing and prior elicitation apps. [https://streamlit.io/gallery](https://streamlit.io/gallery)

[53] Statisticat LLC. LaplacesDemon: Complete environment for Bayesian inference (R package). CRAN; 2021. [https://CRAN.R-project.org/package=LaplacesDemon](https://CRAN.R-project.org/package=LaplacesDemon)

[54] Rafi Z, Greenland S. vizdraws: Visualize draws from the prior and posterior distributions (R package). CRAN; 2023. [https://CRAN.R-project.org/package=vizdraws](https://CRAN.R-project.org/package=vizdraws)

[55] Oslo Centre for Biostatistics and Epidemiology. PriorElicitation (R Shiny app). GitHub; 2021. [https://github.com/ocbe-uio/PriorElicitation](https://github.com/ocbe-uio/PriorElicitation)

[56] Villani M. Bayesian learning collection. Observable; 2023. [https://observablehq.com/collection/@mattiasvillani/bayesian-learning](https://observablehq.com/collection/@mattiasvillani/bayesian-learning)

[57] Guesstimate. Guesstimate: A spreadsheet for things that aren't certain. GitHub; 2020. [https://github.com/getguesstimate/guesstimate-app](https://github.com/getguesstimate/guesstimate-app)

[58] Veen D, et al. Proposal for a five-step method to elicit expert judgment. Front Psychol. 2017;8:2110. [https://doi.org/10.3389/fpsyg.2017.02110](https://doi.org/10.3389/fpsyg.2017.02110)