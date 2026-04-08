# Visualization for Uncertainty-Aware Binary Data Exploration

*Author: Thor Whalen*
*Date: March 2026*

---

**When every estimate has wide intervals, uncertainty must be the primary visual signal — not an overlay on point estimates.** This survey addresses the design of a modern React/TypeScript frontend for exploring relationships in small-sample binary datasets (10–50 cases, 10–30 binary variables), where the combinatorial space is vast but the evidence is sparse. No existing tool combines Bayesian analysis of multiple 2×2 tables, interactive multi-pair exploration, and uncertainty-first visualization in a web-delivered package. What follows is a comprehensive review of the chart types, libraries, interaction patterns, and existing systems that inform the construction of such a tool — grounded in the principle that making uncertainty first-class is not a design embellishment but a cognitive necessity.

---

## 1. Visual encodings that put uncertainty front and center

The central challenge is encoding *two* quantities per visual element — an effect size and a measure of confidence — without overwhelming the viewer or allowing one channel to dominate the other. Eight visualization techniques address different facets of this problem.

### 1.1 Uncertainty heatmaps with bivariate color

For the overview of all pairwise associations among 10–30 binary variables, a matrix heatmap is the natural starting point. The question is how to prevent viewers from over-interpreting uncertain cells. The **Value-Suppressing Uncertainty Palette (VSUP)** developed by Correll, Moritz, and Heer [1] uses a tree-quantization scheme that allocates more distinct color bins when uncertainty is low and fewer when uncertainty is high. Highly uncertain cells collapse to a single neutral color, making it *structurally impossible* to over-interpret them. A crowdsourced evaluation (n=300) showed VSUP users weighted uncertainty significantly more in decision tasks (K-S test D=0.5, p=0.03). Alternative approaches — mapping opacity to certainty or cell size to confidence — are intuitive but perceptually harder to decode. Bivariate choropleth palettes (e.g., Joshua Stevens, Cynthia Brewer) assign unique colors to each value×uncertainty combination, but user studies consistently show **poor decoding accuracy for continuous bivariate scales** [1]. Discrete scales with 3×3 or 4×4 bins perform better.

For implementation, the original VSUP code is available at `uwdata/papers-vsup` on GitHub [1], and the `pals` and `multiscales` R packages provide ready-made VSUP palettes. No standalone JavaScript VSUP library exists, so a custom D3 implementation using `d3-scale-chromatic` and a tree-quantized discrete palette is recommended. In the React/TypeScript stack, this means using D3 purely for scale computation while rendering SVG cells declaratively.

### 1.2 Mosaic plots for individual 2×2 tables

Mosaic plots encode the four cells of a 2×2 contingency table as rectangles whose width and height are proportional to marginal frequencies [2]. Departures from independence are visible as cells that are "too large" or "too small" relative to what random association would produce. Shading cells by standardized residuals (blue for larger-than-expected, red for smaller) as in Friendly's extended mosaic plots [2] highlights statistically meaningful departures. However, area-based comparisons are perceptually difficult — aspect ratio judgments are unreliable — and mosaic plots **cannot directly encode uncertainty or credible intervals**. They are best used as detail-on-demand for a single selected variable pair, annotated with observed and expected counts and supplemented by posterior distribution displays. JavaScript implementations include AnyChart's `anychart.mosaic()`, JSCharting, and custom D3 layouts via Observable notebooks.

### 1.3 Posterior distribution plots and prior→posterior updating

For Bayesian analysis of binary data, the Beta-Binomial conjugate model produces closed-form posteriors: given a Beta(α, β) prior and observing *s* successes in *n* trials, the posterior is Beta(s+α, n−s+β) [3]. Visualizing these posteriors — especially overlaying the posterior distributions for P(Y=1|X=1) versus P(Y=1|X=0) — is essential for communicating what the data say about association.

The **half-eye plot** from Matthew Kay's `ggdist` package [4] combines a density curve (the "slab") with a point estimate and interval (the "interval"), producing a compact, information-dense display. Kay's 2024 IEEE TVCG paper establishes ggdist as the state of the art for distributional visualization in R, providing `stat_halfeye()`, `stat_eye()`, `stat_dots()` (quantile dotplots), and `stat_slabinterval()` [4]. For prior→posterior updating, showing the prior as a dashed or lower-opacity ghost curve behind the filled posterior is the standard approach. Animated transitions from prior to posterior — with each observation incrementally updating the distribution — are highly effective for pedagogy, as demonstrated by Brown University's *Seeing Theory* [5] and Kristoffer Magnusson's D3-based Bayesian inference visualizations at rpsychologist.com [6].

Best practices include displaying the **Highest Density Interval (HDI)** rather than equal-tailed intervals for skewed posteriors (common with small binary data) and computing the posterior distribution of the *difference* between proportions (or the log-odds ratio), which directly answers the association question [3]. In JavaScript, no standalone posterior visualization library exists; custom D3 implementations using `d3-shape` for area/line rendering and parametric Beta PDF computation are the primary approach.

### 1.4 Forest plots for ranking many pairs

With up to **435 variable pairs** from 30 binary variables, a forest plot — ranking associations by posterior effect size with horizontal credible intervals — provides the most efficient overview for identifying strong, well-supported associations. Each row represents one variable pair; the point marks the posterior median; the line spans the 95% credible interval; a vertical null reference line at zero allows instant identification of intervals that exclude no effect [7].

Sorting options matter: sorting by point estimate highlights the strongest effects, but sorting by the *lower bound* of the credible interval identifies the "most confidently positive" effects — a more useful ranking when uncertainty is dominant. For the interactive web context, a D3.js custom forest plot is straightforward (horizontal lines + circles + axis), and the implementation should support dynamic re-sorting, threshold-based filtering (e.g., hide pairs whose credible interval includes zero), and click-to-drill-down to posterior distributions for any selected pair.

### 1.5 UpSet plots reveal which Boolean patterns actually exist

When analyzing binary data with many variables, the key context question is: *which combinations of variable values actually occur?* With 30 binary variables, the theoretical combinatorial space is 2^30 (over a billion), but with 10–50 cases, only a tiny fraction of configurations are populated. **UpSet plots** [8] visualize exactly this: they show only the non-empty intersections as bars, with a matrix below indicating which variables are active in each combination.

UpSet received the **IEEE InfoVis 10-year Test of Time Award in 2024** [8], confirming its lasting impact. For JavaScript, two production-ready implementations exist. **UpSet.js** by Samuel Gratzl (`@upsetjs/react`) is a TypeScript/React component with interactive hover/click selection, SVG export, box plots for numerical attributes per intersection, dark/light themes, logarithmic scale, and adapters for React, Vue, Vanilla JS, R, Python, PowerBI, and Tableau [9]. **UpSet 2.0** from the Visual Design Lab at Utah (`@visdesignlab/upset2-react`) adds provenance tracking, alt-text generation, and embedding [10]. For this use case, the **deviation metric** built into UpSet — showing how much each intersection deviates from the expectation under independence — is directly relevant to association analysis.

### 1.6 Hypothetical Outcome Plots and the Hullman/Kay tradition

The most impactful recent advance in uncertainty communication is the **Hypothetical Outcome Plot (HOP)**, developed by Hullman, Resnick, and Adar [11]. HOPs animate a finite set of draws from a distribution, each shown as a separate frame at approximately **400ms per frame**. Instead of showing a static summary, HOPs make viewers *experience* uncertainty through countable, concrete outcomes. Kale, Nguyen, Kay, and Hullman [12] demonstrated that observers correctly infer underlying trends at lower evidence levels with HOPs compared to static uncertainty displays — **participants were 35–41 percentage points more accurate** on multivariate probability estimation tasks compared to error bars.

The cognitive mechanism is powerful: HOPs force viewers to build up a mental representation of the distribution over time, preventing fixation on a single point estimate (the "deterministic construal error" that plagues static displays). For multivariate data, HOPs naturally express correlations — co-movement between two proportions across frames — which static displays cannot show. **Quantile dotplots** [13], a static alternative from the same research tradition, represent a distribution as a fixed number of equally-likely dots and yield better decisions than PDFs, CDFs, and text descriptions in transit decision-making tasks [14].

Padilla, Kay, and Hullman's comprehensive review chapter [15] synthesizes the evidence: frequency-framed displays (HOPs and quantile dotplots) should be preferred over interval-based displays whenever possible. In JavaScript, no standalone HOPs library exists, but implementation is straightforward: sample from the posterior, animate chart transitions at ~400ms intervals using Motion (formerly Framer Motion) or React Spring. The original HOPs experiment interfaces were built with D3.js [11]. For the 2×2 table explorer, animating plausible mosaic plots or bar charts drawn from the posterior would let users watch the data "dance" and develop intuition about what the evidence supports.

### 1.7 Icon arrays for transparent small-sample display

Icon arrays display exactly *n* icons — one per case — colored by classification (e.g., green = both variables true, red = both false). They leverage **frequency framing**, expressing probabilities as "X out of N" rather than abstract percentages, which is cognitively easier for non-statisticians [16]. For a 2×2 table with n=50 cases, an icon array of 50 icons directly represents the empirical joint distribution. Viewers see exactly how few cases support the association estimate — a critical reality check with small samples. Best practice recommends randomly shuffling icon positions rather than grouping by category, though this slightly increases perceived risk [17]. Implementations include Brian Zikmund-Fisher's `iconarray.com` [16], the `riskyr` R package, and straightforward custom D3 implementations (render N SVG circles, color by 2×2 cell membership).

### 1.8 Network views expose association structure

An association network represents each binary variable as a node and each pairwise association as a weighted edge, with edge width proportional to absolute effect size, edge color indicating direction (positive/negative), and edge opacity encoding confidence [18]. Force-directed layouts position strongly associated variables close together, revealing clusters. With 10–30 nodes, the graph is manageable. The critical interaction is a **certainty threshold slider** that dynamically filters edges — letting users discover which associations survive at various confidence levels.

**D3's `d3-force`** module is the standard for custom force-directed layouts. **Cytoscape.js** [19] provides a full graph theory library with built-in community detection, compound nodes, and analysis algorithms. For this use case, applying the Louvain algorithm to the thresholded association matrix identifies variable clusters, complementing the heatmap view by emphasizing structural relationships rather than individual cell values.

---

## 2. The JavaScript library landscape for statistical graphics in React

### 2.1 Visx is the strongest fit for custom Bayesian visualization

The recommended primary library is **Visx** (Airbnb), currently at v3.12.0 [20]. Visx is 100% React-native — every component is a React component — using D3 purely for mathematical calculations under the hood. Its modular architecture provides precisely the building blocks needed: `@visx/heatmaps` (HeatmapCircle, HeatmapRect), `@visx/shape` (Area, LinePath, Bar), `@visx/scale`, `@visx/axis`, `@visx/tooltip`, `@visx/brush`, `@visx/zoom`, `@visx/stats`, and `@visx/threshold` for area difference charts useful in uncertainty bands. TypeScript support is excellent (rewritten in TypeScript for v1.0). The "bring your own animation" philosophy means it pairs naturally with **Motion** (formerly Framer Motion, v12.x, **30M+ monthly npm downloads**) [21] for HOPs-style animated uncertainty displays. Bundle size is modular — import only what you need — yielding ~150–250KB total versus ~2MB for Plotly.js alone.

The key architectural pattern is: **D3 for math, Visx for React components, Motion for animation.** D3 utility modules (`d3-scale`, `d3-array`, `d3-scale-chromatic`, `d3-interpolate`, `d3-shape`) handle KDE computation, bivariate color interpolation, and statistical transforms. Visx components render SVG elements declaratively in JSX. Motion's `AnimatePresence` and spring-based animations power HOPs and smooth transitions between views.

### 2.2 How the alternatives compare

**D3.js v7.9.0** [22] offers unlimited expressiveness and ~113K GitHub stars, but its imperative DOM manipulation conflicts with React's virtual DOM paradigm. The "D3 for math, React for DOM" pattern mitigates this, but animations via `d3-transition` require ref-based escape hatches. D3 remains the right choice for mathematical primitives but should not own the DOM in a React application.

**Observable Plot v0.6.16** [23] provides a concise grammar-of-graphics API with built-in statistical transforms (bin, group, density), but has no animation system, no native forest plot or mosaic plot marks, and React integration is side-effect-based (useRef/useEffect). Suitable for rapid prototyping but not for the custom encodings this application requires.

**Plotly.js v3.4.0** [24] has the richest built-in statistical chart catalog (violin plots, 2D density, error bars, heatmaps) and strong interactivity, but its **~2MB bundle size**, limited customization for novel encodings (bivariate heatmaps, mosaic plots), and awkward React integration (Plotly manages its own DOM) make it a poor choice for a custom-built uncertainty-first application.

**Nivo v0.99.0** [25] provides a built-in HeatMap component and React Spring animations, but lacks density plots, forest plots, and statistical visualization primitives. Its Marimekko chart type is related to mosaic plots but not equivalent. Single-maintainer risk is a concern.

**Recharts v2.15.x** [26] excels at standard dashboards but is **nearly impossible to customize** for non-standard chart types. No heatmap, density, forest, or mosaic components exist. Not suitable.

**Vega-Lite v5.x** [27] provides a powerful declarative grammar with built-in selections, error bands, and density transforms. Its selection model supports coordinated views within a single specification. However, cross-view coordination across independent React components requires bridging via signal listeners to external state management. Vega-Lite is a strong secondary choice for rapid prototyping of individual chart types before building custom Visx components for production.

**deck.gl** [28] targets large-scale geospatial data with WebGL — completely wrong domain for 10–50 cases × 10–30 variables.

### 2.3 No specialized uncertainty library exists in JavaScript

A thorough search found no production-ready JavaScript/TypeScript library for Bayesian or uncertainty visualization. The R ecosystem has ggdist [4] and tidybayes [29]; Python has ArviZ [30]. In JavaScript, relevant implementations are one-off D3 creations: Magnusson's rpsychologist.com Bayesian visualizations [6], Brown University's Seeing Theory [5], and Vega's HOPs example [27]. The gap is real and confirms that this application must build its uncertainty visualization primitives from lower-level components (Visx + D3 + Motion).

---

## 3. Interaction patterns for coordinated uncertainty exploration

### 3.1 Coordinated multiple views with use-coordination

The application requires multiple synchronized views — heatmap, forest plot, network graph, posterior distribution panel — where selecting in one view highlights in all others. The purpose-built solution is **`use-coordination`**, presented at IEEE VIS 2024 by Keller, Manz, and Gehlenborg [31]. It provides a declarative JSON-based coordination grammar for React: views subscribe to named "coordination scopes" for typed "coordination types" (selection, zoom, highlight) via a `useCoordination(viewUid, ['coordType1', ...])` hook. A `CoordinationProvider` wraps the application. The library is visualization-library-agnostic — it works with D3, Vega, Visx, or plain SVG.

For global application state (filter thresholds, active prior settings, computational results), **Zustand v5.x** [32] is the recommended state management library (~47K GitHub stars). Its store-centric model with fine-grained subscriptions ensures that only the views consuming a changed state slice re-render — critical for high-frequency interactions like brushing. Transient updates (hover highlighting) can bypass re-renders entirely using Zustand's `subscribe` API. The legacy option, Crossfilter/dc.js, is **abandoned** — the original Crossfilter API is frozen, and `react-dc` is explicitly labeled "not production-ready" [33].

### 3.2 Prior elicitation that works in the observable space

Specifying Bayesian priors is the most challenging interaction design problem. The state-of-the-art finding comes from **PriorWeaver** (2025) [34]: non-statisticians are significantly more comfortable and accurate when they elicit priors in the **observable space** (familiar quantities like "How many out of 30 would you expect to have both X and Y?") rather than in parameter space (abstract distribution parameters). A study with 17 Bayesian novices found PriorWeaver produced priors more aligned with participant knowledge (p<0.05) [34].

The **SHELF (Sheffield Elicitation Framework)** [35] provides five elicitation methods as R/Shiny apps, including the roulette method (chip-allocation to histogram bins with real-time parametric fitting) and the quartile method (specifying median and quartiles via sliders). For JavaScript, **distBuilder** [36] by Quentin André offers a pure-JS distribution builder where users allocate probability mass to bins with three lines of code. For the binary association context, three recommended UI approaches are:

- **Interval input**: A two-handled range slider ("I believe the association is between 0.1 and 0.5") mapped to a Beta prior, with a confidence dropdown controlling prior width
- **Roulette allocation**: SHELF-inspired histogram grid with click-to-allocate and real-time distribution fitting using `stdlib-js/stats-base-dists` [37] for PDF/CDF/quantile computation
- **Observable-space elicitation**: Ask "In a typical sample of 30, how many would have both X and Y?" and derive the prior behind the scenes — the PriorWeaver principle [34]

All three should provide visual feedback showing the implied distribution and prior predictive checks ("Given your prior, here's what the data would typically look like").

### 3.3 Filtering, ranking, and real-time threshold adjustment

With C(30,2) = 435 variable pairs, the filtering UI must support dynamic query sliders for association strength, credible interval width, and posterior probability of direction, with results updating in real-time as thresholds change. At this scale, **pure JavaScript filtering is instant** (<1ms for 435 records), so no server-side computation or Web Worker is needed for the filtering step itself. **TanStack Virtual** [38] provides headless virtualized scrolling via the `useVirtualizer` hook, and **TanStack Table** [39] adds faceted filtering, column-level range inputs, and multi-column sorting — integrating with TanStack Virtual for virtualized rows.

However, **Bayesian posterior computation** (sampling, MCMC, or conjugate posterior evaluation for all 435 pairs) may exceed the 16ms frame budget. For this, **Web Workers with Comlink** [40] (Google's library that wraps the Worker API into async function calls) offload computation to a background thread, keeping the UI responsive. The architecture is: main thread (React UI) ↔ Zustand store ↔ Web Worker (Beta-Binomial posterior computation for all pairs).

### 3.4 Progressive disclosure of statistical detail

Following Shneiderman's visual information-seeking mantra [41] — "overview first, zoom and filter, then details on demand" — a three-level progressive disclosure model is recommended:

- **Glance (hover, 0ms)**: Variable pair name, association direction arrow, strength indicator via color
- **Tooltip (hover, 200ms dwell)**: 2×2 contingency table, point estimate with 95% credible interval, mini posterior density sparkline — rendered as rich React content via **Floating UI** (`@floating-ui/react`) [42], the successor to Popper.js
- **Detail panel (click)**: Full posterior distribution plot with prior overlay, HOPs animation, mosaic plot, icon array, forest plot position highlight — in a side panel or modal

A single Floating UI tooltip instance should be repositioned rather than created/destroyed per cell, for performance with hundreds of heatmap cells.

---

## 4. Existing tools and the gap they leave

### 4.1 Bayesian 2×2 table tools exist but don't scale to multi-pair exploration

**JASP** [43] performs Bayesian contingency table analysis using Gunel and Dickey Bayes factors [44], but operates on a single table at a time with no multi-variable exploration and no web interface. **bayesAB** [45], an R package, implements the exact Beta-Binomial conjugate model needed — with built-in `plot()` methods showing prior, posterior, and posterior of difference — but only for single A/B comparisons. The closest web-based tool is **Yanir Seroussi's Bayesian Split Test Calculator** [46], a pure-JavaScript client-side app where users set prior beliefs, enter counts, and see posterior distributions update interactively. Its interaction paradigm (prior specification → live posterior update) is directly transferable, but it handles only one comparison at a time.

### 4.2 QCA handles the right data type but ignores uncertainty entirely

Qualitative Comparative Analysis (QCA) is specifically designed for small-N binary data (10–50 cases with 10–30 conditions) — the exact dataset shape. **fsQCA** [47] (82% market share, by Charles Ragin) produces truth tables and Boolean minimization but has zero probabilistic inference, no uncertainty quantification, and basic static visualization. The **QCA R package** (v3.23, by Adrian Dusa) [48] is the most comprehensive and correct implementation, featuring the CCubes algorithm for fast Boolean minimization. **Coincidence Analysis (CNA)** [49], an alternative framework by Baumgartner and Ambühl, discovers INUS-causal structures from binary data and offers robustness scoring via the `frscore` package — a form of sensitivity analysis that partially addresses the "which findings are trustworthy" question. The `causalHyperGraph` package [49] visualizes CNA models as causal hypergraphs. All QCA/CNA tools operate in a **deterministic paradigm** — they identify crisp set-theoretic relationships with no posterior distributions. Adding Bayesian uncertainty visualization to QCA/CNA-like analyses would be extremely valuable.

### 4.3 Association rule exploration offers the best UI precedents

The **arulesViz** R package [50] by Michael Hahsler provides the most relevant exploration interface: a matrix visualization (antecedent × consequent items with color-coded interest measures), a network graph of rules using visNetwork (interactive JavaScript/HTML output), and an interactive **`ruleExplorer()`** Shiny app for filtering, sorting, and visualizing rules across multiple views. The grouped matrix view — showing all pairwise associations at once with color-coded cells — is essentially the same visual concept as the proposed uncertainty heatmap, minus the uncertainty encoding. The `htmlwidget`-based output (visNetwork graphs, Plotly scatter plots, DT tables) demonstrates that R-generated interactive visualizations can work in web contexts. However, association rules assume large transaction databases and provide no uncertainty quantification.

### 4.4 The uncertainty visualization ecosystem lives in R and Python

The **ggdist** R package [4] by Matthew Kay is the gold standard for static uncertainty visualization, providing quantile dotplots, gradient intervals, half-eye plots, and other primitives backed by rigorous empirical evaluation. **ShinyStan** [51] (Stan Development Team) provides the closest existing paradigm to "interactive Bayesian exploration in a web browser," with interactive density, trace, and bivariate scatter plots for posterior parameters — but it targets MCMC diagnostics, not data exploration. **ArviZ** [30] (Python, NumFOCUS affiliated, 75M+ downloads) provides a unified Python interface for Bayesian model exploration with forest plots, pair plots, and posterior densities, but has no web GUI. The **IPME (Interactive Probabilistic Models Explorer)** [52] is an academic prototype that enables interactive conditioning on a Bayesian DAG with live posterior updates — conceptually very close to the desired tool but not production software.

### 4.5 The MU Collective defines the research frontier

The **Midwest Uncertainty Collective** at Northwestern [53], led by Jessica Hullman and Matthew Kay, is the leading research group on uncertainty visualization. Their recent output directly informs this design:

- **"Odds and Insights"** (CHI 2024) [54] examines decision quality during exploratory data analysis under uncertainty
- **"In Dice We Trust"** (CHI 2024, Best Paper) [55] studies uncertainty displays for maintaining trust in data
- **VMC** (IEEE VIS 2024) [56] provides a grammar for specifying visual model checking displays
- **Padilla, Kay, and Hullman's review chapter** (2022) [15] is the definitive survey of uncertainty visualization theory and evidence

Their collective finding: **frequency-framed displays (HOPs and quantile dotplots) should be preferred over interval-based displays** whenever the goal is accurate probability judgment by non-experts.

---

## 5. Recommended architecture and view hierarchy

The system should layer views from overview to detail following Shneiderman's mantra [41], with each view addressing a different analytical question:

| View | Primary question | Priority | Implementation |
|------|-----------------|----------|---------------|
| **VSUP Heatmap** | Which pairs show strong, certain associations? | Overview | Visx + D3 bivariate color scale |
| **Forest Plot** | What are the ranked effect sizes with uncertainty? | Overview | Visx custom (shape + scale + axis) |
| **Association Network** | What structural clusters exist among variables? | Overview | D3-force or Cytoscape.js |
| **UpSet Plot** | Which Boolean profiles actually occur? | Context | @upsetjs/react |
| **Posterior Distribution** | What does the evidence say about one pair? | Detail | Visx + D3 Beta PDF computation |
| **HOPs Animation** | What could the data plausibly look like? | Detail | Visx + Motion |
| **Mosaic Plot** | What is the raw 2×2 table structure? | Detail | Visx custom (Rect + partition) |
| **Icon Array** | How few cases support this estimate? | Detail | D3/SVG (n circles) |

The recommended technology stack is:

| Concern | Library | Role |
|---------|---------|------|
| Visualization primitives | Visx v3.12+ | React-native SVG components |
| Statistical math | D3 v7 submodules | Scales, KDE, color interpolation |
| Animation | Motion v12+ | HOPs, transitions, spring physics |
| State coordination | Zustand v5 | Global filter/selection state |
| View linking | use-coordination | Formal CMV coordination model |
| Distribution math | stdlib-js | Beta PDF/CDF/quantile computation |
| Virtual scrolling | TanStack Virtual | Pair results list |
| Tooltips | Floating UI | Rich statistical detail-on-demand |
| Boolean patterns | @upsetjs/react | UpSet plot component |
| Graph layout | Cytoscape.js or D3-force | Association network |
| Background computation | Web Workers + Comlink | Posterior computation for 435 pairs |

Total estimated bundle impact is **~150–250KB** for Visx + D3 submodules + Motion — roughly one-eighth the size of Plotly.js alone.

---

## Conclusion

The design space for uncertainty-aware binary data exploration is rich in research evidence but thin in production tooling. The dominant finding from the Hullman/Kay research tradition [11][12][13][14][15] is that **frequency-framed displays outperform abstract statistical representations** for probability judgment — making HOPs and quantile dotplots the preferred uncertainty encodings over confidence intervals and error bars. The VSUP palette [1] solves the bivariate heatmap problem by making uncertain cells structurally illegible. UpSet plots [8] solve the combinatorial explosion problem by showing only populated Boolean configurations.

On the engineering side, the **Visx + D3 + Motion** stack [20][21][22] provides the lowest-friction path to custom statistical graphics in React/TypeScript, and **use-coordination** [31] offers a rigorous, recently published framework for coordinated multiple views. The most novel opportunity lies at the intersection of QCA/CNA-style configurational analysis [47][48][49] and Bayesian uncertainty visualization — no tool currently occupies this space. Prior elicitation should follow PriorWeaver's principle [34] of working in the observable space, and the system should treat every view as an answer to a different analytical question, unified by shared selection and filter state through Zustand [32] and coordinated highlighting.

The gap is clear: existing tools handle either the statistics (bayesAB [45], JASP [43]) or the exploration (arulesViz [50]) or the uncertainty visualization (ggdist [4]) or the Boolean structure (QCA [48], CNA [49]) — but none integrates all four. Building this integration in the browser, with uncertainty as the primary visual signal, would constitute a genuinely novel contribution to the visual analytics landscape.

---

## REFERENCES

[1] Correll, M., Moritz, D., & Heer, J. (2018). Value-Suppressing Uncertainty Palettes. *Proc. CHI 2018*. [uwdata/papers-vsup](https://github.com/uwdata/papers-vsup)

[2] Friendly, M. (1994). Mosaic Displays for Multi-Way Contingency Tables. *JASA*, 89(425), 190–200.

[3] Kruschke, J.K. (2014). *Doing Bayesian Data Analysis*, 2nd ed. Academic Press.

[4] Kay, M. (2024). ggdist: Visualizations of Distributions and Uncertainty in the Grammar of Graphics. *IEEE TVCG*, 30(1), 414–424. [mjskay/ggdist](https://mjskay.github.io/ggdist/)

[5] Kunin, D., et al. Seeing Theory. Brown University. [seeing-theory.brown.edu](https://seeing-theory.brown.edu)

[6] Magnusson, K. Understanding Bayesian Inference (Interactive D3 Visualization). [rpsychologist.com/d3/bayes](https://rpsychologist.com/d3/bayes/)

[7] Lewis, S. & Clarke, M. (2001). Forest plots: trying to see the wood and the trees. *BMJ*, 322(7300), 1479–1480.

[8] Lex, A., Gehlenborg, N., Strobelt, H., Vuillemot, R., & Pfister, H. (2014). UpSet: Visualization of Intersecting Sets. *IEEE TVCG (InfoVis)*, 20(12), 1983–1992. [upset.app](https://upset.app)

[9] Gratzl, S. UpSet.js — JavaScript Implementation of UpSet. [upsetjs.netlify.app](https://upsetjs.netlify.app)

[10] Gadhave, K., et al. (2019). UpSet 2: From Prototype to Tool. *IEEE VIS Short Papers*. [vdl.sci.utah.edu/upset2](https://vdl.sci.utah.edu/upset2)

[11] Hullman, J., Resnick, P., & Adar, E. (2015). Hypothetical Outcome Plots Outperform Error Bars and Violin Plots for Inferences about Reliability of Variable Ordering. *PLoS ONE*, 10(11), e0142444.

[12] Kale, A., Nguyen, F., Kay, M., & Hullman, J. (2019). Hypothetical Outcome Plots Help Untrained Observers Judge Trends in Ambiguous Data. *IEEE TVCG*, 25(1), 892–902.

[13] Kay, M., Kola, T., Hullman, J., & Munson, S. (2016). When (ish) is My Bus? User-Centered Visualizations of Uncertainty in Everyday, Mobile Predictive Systems. *Proc. CHI 2016*, 5092–5103.

[14] Fernandes, M., Walls, L., Munson, S., Hullman, J., & Kay, M. (2018). Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit Decision-Making. *Proc. CHI 2018*.

[15] Padilla, L., Kay, M., & Hullman, J. (2022). Uncertainty Visualization. In *Computational Statistics in Data Science*, Chapter 22. Wiley.

[16] Zikmund-Fisher, B.J., Fagerlin, A., & Ubel, P.A. (2008). Improving understanding of adjuvant therapy options by using simpler risk graphics. *Cancer*, 113(12), 3382–3390. [iconarray.com](https://iconarray.com)

[17] Stone, E.R., et al. (2018). Exploring factors that influence perceptions of icon arrays. *SJDM*, 13(4).

[18] Epskamp, S., Cramer, A.O.J., Waldorp, L.J., Schmittmann, V.D., & Borsboom, D. (2012). qgraph: Network Visualizations of Relationships in Psychometric Data. *J. Statistical Software*, 48(4), 1–18.

[19] Franz, M., et al. (2016). Cytoscape.js: a graph theory library for visualisation and analysis. *Bioinformatics*, 32(2), 309–311. [js.cytoscape.org](https://js.cytoscape.org)

[20] Airbnb. Visx — A Collection of Expressive, Low-Level Visualization Primitives for React. [airbnb.io/visx](https://airbnb.io/visx)

[21] Motion (formerly Framer Motion). [motion.dev](https://motion.dev)

[22] Bostock, M. D3.js — Data-Driven Documents. v7.9.0. [d3js.org](https://d3js.org)

[23] Observable. Observable Plot. [observablehq.com/plot](https://observablehq.com/plot)

[24] Plotly. Plotly.js — Open Source JavaScript Graphing Library. [plotly.com/javascript](https://plotly.com/javascript)

[25] Nivo. [nivo.rocks](https://nivo.rocks)

[26] Recharts — A Composable Charting Library Built on React Components. [recharts.org](https://recharts.org)

[27] Satyanarayan, A., Moritz, D., Wongsuphasawat, K., & Heer, J. (2017). Vega-Lite: A Grammar of Interactive Graphics. *IEEE TVCG*, 23(1), 341–350. [vega.github.io/vega-lite](https://vega.github.io/vega-lite)

[28] vis.gl. deck.gl — Large-Scale WebGL-Powered Data Visualization. [deck.gl](https://deck.gl)

[29] Kay, M. tidybayes: Tidy Data and Geoms for Bayesian Models. [mjskay.github.io/tidybayes](https://mjskay.github.io/tidybayes/)

[30] Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ: A Unified Library for Exploratory Analysis of Bayesian Models. *JOSS*, 4(33), 1143. [arviz-devs.github.io/arviz](https://arviz-devs.github.io/arviz/)

[31] Keller, M.S., Manz, T., & Gehlenborg, N. (2024). Use-Coordination: Model, Grammar, and Library for Implementation of Coordinated Multiple Views. *IEEE VIS 2024*, 166–170. [use-coordination.dev](https://use-coordination.dev)

[32] Zustand — Bear Necessities for State Management in React. [zustand.docs.pmnd.rs](https://zustand.docs.pmnd.rs)

[33] Crossfilter. GitHub Issues #185 (Feb 2024): Modern alternatives. [github.com/crossfilter/crossfilter](https://github.com/crossfilter/crossfilter)

[34] PriorWeaver (2025). Prior Elicitation via Iterative Dataset Construction in the Observable Space. arXiv:2510.06550. [arxiv.org/abs/2510.06550](https://arxiv.org/abs/2510.06550)

[35] Morris, D.E., Oakley, J.E., & Crowe, J.A. (2014). A web-based tool for eliciting probability distributions from experts. *Environmental Modelling & Software*, 52, 1–4. [shelf.sites.sheffield.ac.uk](https://shelf.sites.sheffield.ac.uk/software)

[36] André, Q. distBuilder — Distribution Builder for Behavioral Experiments. [quentinandre.net/software/distbuilder](https://quentinandre.net/software/distbuilder/)

[37] stdlib-js. Statistical Distributions. [stdlib.io](https://stdlib.io)

[38] TanStack Virtual. [tanstack.com/virtual](https://tanstack.com/virtual)

[39] TanStack Table. [tanstack.com/table](https://tanstack.com/table)

[40] Google. Comlink — Web Workers Made Easy. [github.com/GoogleChromeLabs/comlink](https://github.com/GoogleChromeLabs/comlink)

[41] Shneiderman, B. (1996). The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations. *Proc. IEEE Symposium on Visual Languages*, 336–343.

[42] Floating UI. [floating-ui.com](https://floating-ui.com)

[43] JASP Team. JASP — A Fresh Way to Do Statistics. [jasp-stats.org](https://jasp-stats.org)

[44] Gunel, E. & Dickey, J. (1974). Bayes Factors for Independence in Contingency Tables. *Biometrika*, 61(3), 545–557.

[45] Portman, F. bayesAB — Fast Bayesian A/B Testing in R. [github.com/FrankPortman/bayesAB](https://github.com/FrankPortman/bayesAB)

[46] Seroussi, Y. Bayesian A/B Test Calculator. [yanirs.github.io/tools/split-test-calculator](https://yanirs.github.io/tools/split-test-calculator/)

[47] Ragin, C.C. fsQCA Software. [sites.socsci.uci.edu/~cragin/fsQCA](https://sites.socsci.uci.edu/~cragin/fsQCA/software.shtml)

[48] Dusa, A. (2019). QCA with R: A Comprehensive Resource. *Springer*. [cran.r-project.org/package=QCA](https://cran.r-project.org/package=QCA)

[49] Baumgartner, M. & Ambühl, M. Coincidence Analysis (CNA). [cran.r-project.org/package=cna](https://cran.r-project.org/package=cna)

[50] Hahsler, M. arulesViz — Visualizing Association Rules. [github.com/mhahsler/arulesViz](https://github.com/mhahsler/arulesViz)

[51] Gabry, J. ShinyStan — Interactive Visual and Numerical Diagnostics for Bayesian Models. [mc-stan.org/shinystan](https://mc-stan.org/shinystan/)

[52] Puljiz, D., et al. (2020). IPME: An Interactive Probabilistic Models Explorer. *Frontiers in Computer Science*, 2, 567344.

[53] Midwest Uncertainty Collective (MU Collective). Northwestern University. [mucollective.northwestern.edu](https://mucollective.northwestern.edu/)

[54] MU Collective (2024). Odds and Insights: Decision Quality in Exploratory Data Analysis Under Uncertainty. *Proc. CHI 2024*.

[55] MU Collective (2024). In Dice We Trust: Uncertainty Displays for Maintaining Trust in Data. *Proc. CHI 2024* (Best Paper).

[56] Guo, A., Kale, A., Kay, M., & Hullman, J. (2024). VMC: A Grammar for Visualizing Statistical Model Checks. *IEEE VIS 2024*.