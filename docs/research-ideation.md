# Research Ideation: Toy Projects on Top of Bézier Splatting

Date: 2026-07-06. Synthesis of a four-track literature review (appendices A–D) plus a read of this
codebase's extension points. The renderer here — curve-structured 2D Gaussians with a ~150× cheaper
backward pass than DiffVG, prune/densify optimizer surgery, and SVG export — changes the economics
of three things at once: (1) putting *structure* rather than control points under the optimizer,
(2) putting *gradients* rather than just scalar rewards into RL loops, and (3) putting the
*renderer inside the training loop* of dense proposal networks at scale.

Codebase extension points referenced throughout:

- `src/bezier_splatting/model.py` — `VectorGraphicsScene` holds all parameters (open control
  points, closed shared/interior points, colors, opacities, widths, learned depth) and converts
  curves → Gaussians in `get_gaussians`.
- `src/bezier_splatting/optimization.py` — `fit_image` training loop; `_prune_and_densify` and the
  optimizer-state surgery helpers (`_splice_optimizer_state` etc.) already support structural
  edits mid-optimization.
- `src/bezier_splatting/losses.py` — reconstruction + shape regularizers (xing, curvature,
  boundary-joint) — the natural home for new structural losses.
- `src/bezier_splatting/topology.py` — prune/densify mask logic — the natural home for
  structure-aware split/merge/snap moves.
- `src/bezier_splatting/svg.py` — export — where richer primitives (gradients, shapes, text)
  would surface.

---

## The one-line map of what the literature says is open

- **Vector graphics broadly (App. A):** every major optimization pipeline (LIVE, O&R, CLIPasso,
  VectorFusion, SVGDreamer++) is still DiffVG-bound and slow; the splat→SVG export fidelity gap is
  unanalyzed; gradient fills are free in splat-land but unexported; no benchmark exists for
  differentiable rasterizers themselves.
- **WYSIWYG parameterization (App. B):** *no* differentiable renderer anywhere optimizes
  editor-native parameters (rect `w/h/radius/rotation`, ellipse, arrow, live text) end-to-end;
  Crello-style ground-truth editor files have never been used for inverse rendering; nobody
  measures round-trip functional editability.
- **RL × rendering (App. C):** a full 2025–26 wave (RLRF, Reason-SVG, SGP-RL, TikZilla, RRVF)
  does GRPO with rendered-SVG rewards, and every one uses a non-differentiable rasterizer and
  discards the gradient. Hybrid-gradient machinery (SHAC, GI-PPO, α-order estimators) exists but
  has never touched a renderer. Per-primitive credit assignment, reward-hacking audits in vector
  parameter space, and OCR/legibility rewards for vector text are all unclaimed.
- **Dense parameter proposal (App. D):** feedforward proposers exist for unstructured 2D Gaussians
  (Instant GaussianImage) and for DiffVG paths (Im2Vec, SuperSVG), but the intersection — a
  network that predicts curve-structured splat scenes — is empty. No learned iterative refiner
  ("RAFT for control points") exists; no amortize-vs-optimize Pareto benchmark exists for
  vectorization; the SwiftSketch distillation recipe has not been applied to filled color VG.

---

## Direction A — WYSIWYG-native parameterization (text, shapes, images)

The core reframe: today the scene is a *soup* of curves; an editor document is a *typed tree* of
objects with low-dimensional semantic parameters. The renderer doesn't need to change — each
editor primitive is a differentiable **generator** that emits Bézier control points, which then
flow through the existing `get_gaussians` → `rasterize` path unchanged.

### A1. Editor-primitive generator layer ("splat the handles, not the points")

Add primitive classes whose *learnable parameters are the editor handles*: rounded rect
`(cx, cy, w, h, corner_r, θ)`, ellipse `(cx, cy, rx, ry, θ)`, line/arrow
`(x1, y1, x2, y2, width, head_size)`, polygon-with-n-sides, star. Each deterministically emits
closed-curve control points (rounded rect = 4 lines + 4 circular-arc-approximating cubics — all
smooth functions of the handle params), so gradients flow from pixels to handles for free.
Fit UI screenshots, slides, flat icons with a mixed population of primitives + free-form curves.

- **Novelty anchor:** App. B gap 1 — literally does not exist; Chat2SVG uses primitives only as
  LLM initialization then dissolves them into point edits.
- **Toy scope:** a `PrimitiveScene` wrapper around `VectorGraphicsScene` (or a parallel parameter
  bank feeding `_assemble_boundary_cp`); evaluate on synthetic editor renders where ground-truth
  handles are known → report handle-recovery error, not just PSNR.
- **Risk:** local minima are worse in low-dim parameter spaces (a rect can't deform its way out of
  a bad pose the way 10 control points can). Mitigation: many random restarts are cheap at splat
  speed; or direction-C proposer provides init.

### A2. Structure collapse / MDL vectorization ("snap moves" in the prune-densify loop)

Extend `_prune_and_densify` with *typed structural moves*: periodically test whether a cluster of
free-form curves is explained (within residual ε) by a single rect/ellipse/line — fit the
primitive to the cluster, compare re-rendered residual, and if it wins under an MDL objective
(bits for primitive type + params vs. bits for the curves it replaces), splice the swap into the
optimizer with the existing state-surgery helpers. The optimizer then continues on the new
parameterization. The trajectory is: pixels → curve soup → progressively "crystallizing" document.

- **Novelty anchor:** App. B gap 4 (no structure-aware split/merge exists) + App. A gap on
  principled path-budget/MDL control. Also directly monetizes this repo's most distinctive asset:
  optimizer surgery machinery that already supports mid-run structural edits.
- **Toy scope:** rect + ellipse + line vocabulary only, greedy RANSAC-style proposal from curve
  clusters, MDL acceptance test. Evaluate: description length vs PSNR Pareto against LIVE/O&R.
- **Risk:** proposal quality (which clusters to test) — start with color+adjacency clustering as
  in O&R's DBSCAN init.

### A3. Live text objects with a differentiable font axis

A text primitive `(string, font_embedding, size, position, tracking, color)` where glyph outlines
come from a small library of fonts (or a VecFontSDF/DualVector-style learned glyph decoder), get
affinely placed by the layout params, and are splatted like any closed curve. String is proposed
discretely (OCR pass); everything else optimizes by gradient. Font selection = softmax-weighted
blend over per-font outlines annealed to hard choice (or Gumbel-softmax). Add an
**OCR-in-the-loop legibility loss** (frozen recognizer on the rendered crop) — the vector-domain
version of TextDiffuser/AnyText's character losses.

- **Novelty anchor:** App. B gap 2 — De-rendering Stylized Texts (ICCV'21) handles isolated raster
  text crops; nobody jointly optimizes a live text object *among shapes in a whole-document fit*,
  and nobody has an OCR loss flowing through a vector renderer.
- **Toy scope:** 10 fonts, single-line text on synthetic posters/slides; metric = string-preserved
  font/size/position recovery + legibility under downstream edit.
- **Risk:** font blending may produce non-fonts mid-optimization — anneal fast; keep string fixed
  (discrete search stays outside the gradient loop).

### A4. Constraint recovery and re-parameterization (snapping as inference)

During/after fitting, detect near-satisfied relations — equal spacing, alignment of centers/edges,
shared colors, symmetry — using the differentiable alignment losses from layout generation
(Kikuchi '21, LACE) run in *inverse* mode as detectors. Then snap: re-parameterize the document so
the constraint is structural (one shared y-coordinate parameter instead of five) and keep
optimizing. Output is a document whose degrees of freedom look like a designer's.

- **Novelty anchor:** App. B gap 3 — constraint *satisfaction* exists for generation; constraint
  *recovery* inside an inverse-rendering loop is untouched. Pairs with the Ellis-style program
  induction tradition.
- **Toy scope:** alignment + equal-spacing + shared-style only, on synthetic slide layouts;
  metric = constraint precision/recall vs ground truth + robustness of edits (drag one object,
  the aligned ones follow).

### A5. Round-trip editability benchmark (the missing metric for all of the above)

Take documents with ground-truth editor structure (Crello dataset; or programmatically generated
SVG/PPTX), render them, reconstruct with {this repo vanilla, A1, A2, LIVE, O&R, an LLM
vectorizer}, then apply scripted edits (move title, recolor shape, swap font, drag object from
under another) to both recovered and ground-truth files, re-render, and compare. Score
"functional editability," not pixels.

- **Novelty anchor:** App. B gaps 6–7 — Crello has never been used for inverse rendering;
  SVGEditBench et al. score LLM code edits, never optimization-based reconstructions.
- **Why it matters strategically:** this benchmark is the *evaluation substrate* for A1–A4, and a
  benchmark paper positions the repo as community infrastructure (echoing App. A's note that no
  differentiable-rasterizer benchmark exists either).

### A6. Cheap infrastructure wins (small but real)

- **Gradient-fill export:** per-Gaussian color along a curve is a free gradient-fill
  parameterization; fit per-curve linear/radial `<linearGradient>` stops at export
  (App. A gap 3; SGLIVE showed gradients matter but fits them the slow way).
- **Splat→SVG fidelity correction:** quantify the appearance gap between the Gaussian proxy and
  the exported crisp SVG, then close it with a short DiffVG-exact (or width/opacity-calibrated)
  fine-tune stage (App. A gap 1).
- **Image objects:** a rect-clipped raster texture as a first-class primitive (splat cluster or
  compositing layer) so photos inside documents stop eating hundreds of curves (App. B gap 5).

---

## Direction B — Differentiable rendering inside RLVR / non-verifiable-reward RL

The historical arc (App. C) is the pitch: 2018–19 used RL *because* renderers weren't
differentiable (SPIRAL, Learning to Paint); 2020–23 differentiable renderers deleted RL from
stroke-based rendering (DiffVG, Paint Transformer); 2025–26 RL returned for LLM semantics (RLRF,
Reason-SVG, SGP-RL) but *dropped the gradients*. Nobody has closed the loop.

### B1. Refine-then-reinforce: splat refinement inside the GRPO reward

Policy (small VLM, e.g. Qwen2.5-VL-3B, SFT'd on image→SVG) emits SVG. Parse it into
`VectorGraphicsScene` params, run **k gradient steps** through the splat renderer, compute the
reward on the *refined* result. Variants: (a) reward-only smoothing — reward after local search
flattens the reward landscape and stops near-miss rollouts from getting zero credit; (b) DAgger
flavor — also distill the refined parameters back as continuous-token supervision; (c) report the
amortization frontier: policy-only vs policy+k-step vs full optimization.

- **Novelty anchor:** App. C gaps 1 and 6 — no published instance; RLRF explicitly names
  non-differentiability as its limitation. The 150× backward makes k-step-per-rollout affordable
  (32-rollout GRPO batches × k=20 steps is trivial).
- **Toy scope:** icon-scale image→SVG, GRPO via any open RLVR stack, reward = DINO/L2 fidelity +
  path-count economy. The claim to test: refinement-smoothed rewards learn faster and hack less.
- **Risk:** parser SVG→scene must be robust to malformed model output (format-gate reward as in
  SGP-RL); the refiner can "carry" a bad policy — track pre- vs post-refinement gap as a metric.

### B2. Per-curve credit assignment as a process reward

With a fast renderer, compute each path's marginal contribution (leave-one-out re-render — the
whole scene re-renders in milliseconds — or gradient×param attribution) and turn it into
**dense per-token-span advantages** for GRPO over the SVG sequence, instead of one scalar
terminal reward for hundreds of tokens.

- **Novelty anchor:** App. C gap 5 — trivially computable, unused everywhere; the vector-graphics
  analog of process reward models, but *exact* rather than learned.
- **Toy scope:** compare terminal-reward GRPO vs per-curve-advantage GRPO on identical budgets;
  measure sample efficiency and path-economy of the learned policy.
- **Risk:** marginal contribution ignores interactions (occlusion); Shapley-style sampled
  coalitions are still cheap at splat speed.

### B3. Hybrid discrete/continuous policy gradients (GI-PPO for graphics)

SVG emission factors into discrete structure (which primitives, how many, z-order → score-function
gradients) and continuous parameters (coordinates, colors, widths → *pathwise* gradients through
the splatter). Instantiate GI-PPO / SHAC-style blending: continuous heads get analytic
∂reward/∂params; discrete tokens get GRPO; α-blend per Suh et al.'s bias/variance diagnostic
(rasterizers have their own discontinuity structure at occlusion boundaries — the diagnostic is
genuinely needed, not decoration).

- **Novelty anchor:** App. C gaps 1–2 — the algorithms exist, the renderer instantiation doesn't.
- **Toy scope:** skip the LLM: a small DETR-style policy (direction C architecture) emitting
  typed primitives, trained against CLIP/DINO text rewards. Cleanest possible testbed for the
  estimator question; LLM version is the follow-up.

### B4. Reward-hacking audits by gradient search in vector-parameter space

Use the renderer's gradients to *adversarially optimize curve parameters against reward models
directly* (CLIP, DINO, VLM judges, OCR, aesthetic RMs) — a ~10³-dim search instead of pixel-space
attacks. Catalog which rewards are robust, what the hacks look like (CLIPDraw's adversarial
scribbles, systematized), and whether vector structure is itself a defense. Second study:
**renderer-mismatch hacking** — train against the splat renderer, evaluate under resvg/browser
rendering; does the policy exploit Gaussian softness that a conformant renderer won't reproduce?

- **Novelty anchor:** App. C gaps 3–4 — no adversarial reward auditing in vector-parameter space;
  verifier-reliability for render-as-reward is asserted (RRVF) but uninstrumented.
- **Toy scope:** pure optimization, no RL training needed — the cheapest project on this list,
  and it de-risks B1–B3's reward choices. Natural first project of the direction.

### B5. Vector-native reward stack (verifiable + non-verifiable)

Assemble the missing reward suite: verifiable terms nobody combines (path economy, valid topology
via the existing xing/area machinery, symmetry/snapping, render-at-2-scales consistency) plus the
missing non-verifiable one — a preference RM over *(rendered image, SVG structure)* pairs scoring
editability/economy, used ReFL-style by direct backprop through the splatter rather than via
policy gradients. OCR legibility rewards for text-in-SVG (App. C: OCR rewards exist for diffusion,
never for vector generators — and they're the most hacked, so B4 feeds in).

---

## Direction C — Dense models for parameter proposal

The 3D world already ran this play: per-scene 3DGS optimization → feedforward splat prediction
(PixelSplat, Splatter Image, GS-LRM) → iterative refiners (iLRM). 2D images got Instant
GaussianImage (unstructured Gaussians, no curves, no SVG). The curve-structured version is an
empty slot, and this repo is precisely the training-loop renderer + teacher-generator for it.

### C1. "Instant Bézier Splatting": one-shot proposer + the Pareto benchmark

A Paint-Transformer/DETR-style model: image encoder → learnable curve queries → per-query heads
for {open/closed type, control points, color, opacity, width, depth, keep-confidence}. Train
self-supervised through the splat renderer (render loss only, Paint-Transformer-style random-scene
self-training works with zero data), with confidence-based pruning for cardinality (or an
Instant-GI-style spawn-density map seeding curve positions). Then publish the missing benchmark:
quality-vs-time frontier of {feedforward, feedforward + k refinement steps, full `fit_image`}.

- **Novelty anchor:** App. D gaps 1, 2, 8 — the intersection is empty as of mid-2026; no
  semi-amortized benchmark exists for vectorization.
- **Toy scope:** 128 queries, 128×128 icons/emoji first; the existing renderer is the loss.
  Depth head gives amortized *layer decomposition* for free (App. D gap 6 — never predicted,
  only inherited).

### C2. Teacher-student distillation at corpus scale

`fit_image` is ~10× faster than DiffVG pipelines → run it over 100k images to mint
(image, curve-set) pairs, then train the C1 architecture with Hungarian matching against teacher
curves + render-loss fine-tuning. This is SwiftSketch's recipe (SDS teacher → stroke-diffusion
student) applied for the first time to filled, colored, layered vector graphics.

- **Novelty anchor:** App. D gap 5 — straightforwardly executable and unoccupied.
- **Synergy:** teacher runs with A2's structure-collapse moves ⇒ the student learns to propose
  *typed primitives*, which is exactly the proposer A1 needs and the policy B3 needs.

### C3. Learned curve refiner ("RAFT for control points")

A recurrent network ingests (current render, target, residual) — and, uniquely feasible here,
the **true parameter gradients as input features**, since the backward pass is nearly free — and
emits control-point/color updates. Compare against Adam-through-the-renderer on: steps to
convergence, robustness to bad init, ability to make *non-local* jumps (the move Adam can't do
and that prune/densify only approximates heuristically).

- **Novelty anchor:** App. D gaps 3 — DeepView/RAFT-style learned updates exist for MPIs, flow,
  and 3DGS (iLRM) but not for 2D vector primitives; gradients-as-network-input has no instance.
- **Bonus experiment:** let the refiner also emit split/merge/spawn proposals → a *learned*
  replacement for the heuristic densification schedule (App. D gap 4; App. A's
  "densification theory is missing" gap, attacked with learning instead of theory).

### C4. Diffusion over curve sets with splat-guided sampling

A diffusion/flow model over the scene parameter set (variable cardinality via LayoutDM-style pad
tokens or DiffGS-style density fields), trained on C2's teacher corpus; at sampling time, apply
reconstruction guidance using gradients from the splat renderer to lock generation onto a target
image. Unifies a generative prior over "how designers structure documents" with photometric
fidelity.

- **Novelty anchor:** App. D gap 10 — no counterpart exists (VecFusion/SketchKnitter are
  vector-supervised, fonts/sketches only, unguided).

### C5. Meta-learned initialization (the weekend-sized variant)

Tancik-style meta-learning over an image class to produce curve-set *initializations* (not a full
proposer) from which `fit_image` converges in far fewer steps. Lowest risk, no architecture work,
directly measurable (steps-to-PSNR), and never tried for vector parameter sets (App. D gap 9).

---

## How the directions compose (the larger arc)

A single narrative connects everything: **a neural design de-renderer**. Direction A defines the
target representation (typed, constrained, editable documents) and the benchmark (A5). Direction C
supplies amortized proposers so fitting is interactive (C1/C2) and structure moves are learned
(C3). Direction B trains the discrete/semantic layer that gradients can't reach (which primitives,
what text, what grouping) with the renderer as verifier — using gradients where they exist (B3)
and dense per-primitive credit where they don't (B2). Each toy project stands alone; together they
are one system.

## Shortlist (novelty × feasibility on this codebase)

1. **B4 — reward-hacking audit in vector-parameter space.** Pure optimization, no training infra,
   fills a named gap, de-risks every other RL idea. Start here for direction B.
2. **A1 + A2 — editor-primitive generators + MDL snap moves.** Entirely within this repo's
   existing machinery (generators feed `get_gaussians`; snaps reuse optimizer surgery), confirmed
   novel, produces demos nobody has (a fit that "crystallizes" into a document).
3. **C1 (+C5 as the cheap probe) — Instant Bézier Splatting + the amortize-vs-optimize
   benchmark.** Empty intersection, self-supervised so no dataset blocker, and the benchmark
   framing makes even modest quality publishable.
4. **B2 — per-curve credit assignment for GRPO.** The sharpest single algorithmic idea in the RL
   direction: exact process rewards from a fast renderer. Needs an RLVR training stack, so
   sequence it after B4.
5. **A5 — Crello round-trip editability benchmark.** Highest strategic leverage per unit of
   glamour; do it as soon as any of A1/A2 produce reconstructions worth scoring.

---

# Appendix A — Literature review: differentiable vector graphics & vectorization

(Verbatim survey output.)

## 1. Differentiable rasterizers (the direct competition/substrate)

### DiffVG — "Differentiable Vector Graphics Rasterization for Editing and Learning"
**Li, Lukáč, Gharbi, Ragan-Kelley — SIGGRAPH Asia 2020 (ACM TOG 39(6))**
The foundational differentiable rasterizer for SVG primitives (Bézier paths, strokes, gradients). Handles the discontinuity of coverage at path boundaries by combining anti-aliased area sampling with explicit Monte Carlo edge/boundary sampling to get correct gradients w.r.t. curve control points. Limitations widely cited by successors: boundary sampling is slow (especially backward), gradients are noisy/unstable, no topology change (fixed path count and connectivity), scales poorly to thousands of paths.
*Relevance: this is the baseline every fast-rasterizer paper (including yours) benchmarks against; its gradient-noise and speed pathologies define the problem space.*

### Bézier Splatting — "Bézier Splatting for Fast and Differentiable Vector Graphics Rendering"
**Xi Liu, Chaoyi Zhou, Nanxuan Zhao, Siyu Huang (Clemson + Adobe) — arXiv 2503.16424, NeurIPS 2025 poster**
Samples 2D Gaussians along Bézier curves (and interior regions for closed shapes) and rasterizes them with a Gaussian-splatting-style tile-based renderer; Gaussians natively supply positional gradients at boundaries, avoiding DiffVG's boundary sampling. Claims ~30× faster forward and ~150× faster backward rasterization per step for open curves vs DiffVG, plus an adaptive pruning-and-densification scheme (borrowed conceptually from 3DGS) to escape local minima, and export to standard XML SVG.
*Relevance: the paper replicated here; its weak points (splat-vs-true-SVG appearance gap at export, closed-shape fill fidelity, stroke width semantics) are the natural attack surface.*

### Follow-ups / direct citers (2025–2026)
- **"Birth of a Painting: Differentiable Brushstroke Reconstruction"** — same author group, arXiv 2511.13191 (2025). Applies the splatting-based differentiable-stroke machinery to reconstruct paintings as ordered brushstrokes, with stroke-level temporal decomposition. *Shows the renderer generalizes beyond SVG fitting to stroke media.*
- **"NURBS Splatting: A Unified Differentiable Rendering Framework for Vector Graphics"** — Qiu & Zhou, arXiv 2606.31764 (2026). Generalizes curve splatting from Bézier to NURBS: rational weights, non-uniform knots, long splines, and region filling, rendered as continuous Gaussian fields. *Direct successor — indicates "splat any parametric primitive" is an active direction.*
- **"Vector Scaffolding: Inter-Scale Orchestration for Differentiable Image Vectorization"** — arXiv 2605.11913 (2026). Hierarchical coarse-to-fine optimization framework aimed at preventing topology collapse. *Directly targets the topology/local-minima problem that densification only partially solves.*
- **"DiffBMP: Differentiable Rendering with Bitmap Primitives"** — arXiv 2602.22625 (2026). Differentiable engine optimizing thousands of *bitmap* primitives with layered export. *Adjacent primitive choice.*
- **Knit2Vector** (Computer-Aided Design, 2026) — differentiable vectorization for 3D knitted texture reconstruction; cites Bézier Splatting as infrastructure.

## 2. Optimization-based image vectorization

### LIVE — "Towards Layer-wise Image Vectorization" — Ma et al., CVPR 2022 (oral), arXiv 2206.04655
Progressively adds closed Bézier paths one layer at a time, initializing each new path at the largest reconstruction-error component, and optimizes via DiffVG with a UDF loss and self-intersection (Xing) regularizer. Produces compact, layer-ordered SVGs but is notoriously slow (hours for complex images).
*Relevance: the canonical optimization-loop consumer of a differentiable rasterizer — a fast renderer makes LIVE-style pipelines interactive.*

### O&R — "Optimize & Reduce: A Top-Down Approach for Image Vectorization" — Hirschorn et al., AAAI 2024, arXiv 2312.11334
Inverts LIVE: start with many paths, alternate DiffVG optimization with path reduction (merging/pruning low-importance paths). Roughly an order of magnitude faster than LIVE while being domain-agnostic.
*Relevance: pruning schedules here are a direct analogue of the densify/prune logic; a splatting backend could make O&R near-real-time.*

### SGLIVE — "Segmentation-guided Layer-wise Image Vectorization with Gradient Fills" — Zhou et al., ECCV 2024, arXiv 2408.15741
Extends LIVE with segmentation-guided path initialization and optimizable **radial gradient fills** rather than flat colors.
*Relevance: gradient fills are something the Gaussian-splat representation gets almost for free (per-Gaussian color) but that current Bézier Splatting SVG export doesn't exploit.*

### SuperSVG — "Superpixel-based Scalable Vector Graphics Synthesis" — Hu et al., CVPR 2024, arXiv 2406.09794
Feedforward (learned) vectorization: superpixel decomposition, coarse + refinement prediction models, two-stage self-training with DiffVG in the loop.
*Relevance: the "network predicts, differentiable renderer supervises" pattern — a faster renderer directly cuts training cost.*

### Layered Image Vectorization via Semantic Simplification — Wang et al., arXiv 2406.05404 (TVCG 2025)
Progressive semantic image simplification (diffusion/segmentation priors) generates a coarse-to-fine stack, each level vectorized so layers correspond to semantic structure.
*Relevance: addresses semantic layer ordering, a blind spot of splat-based optimization.*

### LayerPeeler — "Autoregressive Peeling for Layer-wise Image Vectorization" — Wu et al., SIGGRAPH Asia 2025, arXiv 2505.23740
A VLM identifies topmost occluding layers; a finetuned diffusion model "peels" them iteratively, recovering occluded content and yielding amodal paths per layer.
*Relevance: amodal completion is impossible in pure pixel-loss splat optimization — a hybrid (VLM-guided peel + fast splat fitting) is an obvious ideation direction.*

### AmodalSVG — "Amodal Image Vectorization via Semantic Layer Peeling" — arXiv 2604.10940 (2026)
Vectorizes images so occluded regions are represented by complete paths.
*Relevance: confirms amodal/layered vectorization is an active 2026 frontier.*

## 3. Neural SVG generation (feedforward / autoregressive / LLM-based)

- **DeepSVG** — Carlier et al., NeurIPS 2020, arXiv 2007.11301. Hierarchical transformer VAE over SVG command sequences; SVG-Icons8 dataset. *Defines the "SVG as token sequence" paradigm.*
- **Im2Vec** — Reddy et al., CVPR 2021, arXiv 2102.02798. Encoder–RNN decoder emitting closed Bézier paths, trained **without vector supervision** via DiffVG + multi-resolution raster loss. *Archetype of "differentiable renderer as the only supervision."*
- **IconShop** — Wu et al., SIGGRAPH Asia 2023 (TOG), arXiv 2304.14400. Autoregressive transformer tokenizing SVG paths + text for text-guided icon synthesis.
- **StrokeNUWA** — Tang et al., arXiv 2401.17093 (2024). VQ-stroke tokens so an LLM can generate vector graphics as token sequences.
- **StarVector** — Rodriguez et al., CVPR 2025, arXiv 2312.11556. Multimodal LLM generating SVG *code* from images/text; SVG-Bench. *Code-LLM output could initialize splat optimization.*
- **OmniSVG** — Yang et al., NeurIPS 2025, arXiv 2504.06263. Unified multimodal SVG generator on Qwen-VL; MMSVG-2M dataset. *Its geometric-precision failures are where differentiable refinement helps.*
- **LLM4SVG** — Xing et al., CVPR 2025, arXiv 2412.11102. Structured SVG encoding with learnable semantic tokens. *Evidence raw-XML LLM output needs geometric grounding.*
- **Chat2SVG** — Wu, Su, Liao — CVPR 2025, arXiv 2411.16602. LLM drafts semantic SVG templates from basic primitives; diffusion-guided dual-stage optimization refines path parameters through a differentiable renderer. *Concrete "LLM proposes, differentiable renderer disposes" — bottlenecked by DiffVG speed.*
- **SVGFusion** (arXiv 2412.10437), **NeuralSVG** (arXiv 2501.03992), **Reason-SVG** (arXiv 2505.24499, RL with drawing-with-thought), **RoboSVG** (arXiv 2510.22684), **DuetSVG** (arXiv 2512.10894), **Render-in-the-Loop** (arXiv 2604.20730, visual self-feedback), **VectorGym** (arXiv 2603.29852, benchmark). *2026's center of gravity: LLM generation with rendering feedback and standardized benchmarks.*

## 4. Score-distillation & sketch works

- **CLIPDraw** — Frans et al., arXiv 2106.14843 (2021). DiffVG strokes optimized against CLIP similarity.
- **CLIPasso** — Vinker et al., SIGGRAPH 2022 (TOG), arXiv 2202.05822. Photo-to-sketch as budgeted Bézier strokes with CLIP semantic + geometric losses; saliency-based initialization; abstraction via stroke count.
- **VectorFusion** — Jain et al., CVPR 2023, arXiv 2211.11319. Text-to-SVG by SDS from Stable Diffusion through DiffVG.
- **Word-as-Image** — Iluz et al., SIGGRAPH 2023 (TOG), arXiv 2303.01818. Deforms letter outlines via SDS to depict a concept while preserving legibility (ACAP deformation + tone losses).
- **DiffSketcher** — Xing et al., NeurIPS 2023, arXiv 2306.14685. Text-guided vector sketches, attention-based stroke init.
- **SVGDreamer** — Xing et al., CVPR 2024, arXiv 2312.16476. SIVE object-level decomposition + Vectorized Particle-based Score Distillation.
- **SVGDreamer++** — TPAMI 2025, arXiv 2411.17832. Hierarchical vectorization for object/part-level editability, adaptive path count. *SOTA SDS text-to-SVG; still DiffVG-bound and slow.*
- **Text-to-Vector with Neural Path Representation** — Zhang et al., SIGGRAPH 2024 (TOG), arXiv 2405.10317. Learned latent path space constraining paths to valid, non-self-intersecting shapes. *Addresses the degenerate-path problem of raw control-point optimization.*

## 5. Gaussian-splatting-adjacent 2D image representations

- **GaussianImage** — Zhang et al., ECCV 2024, arXiv 2403.08551. Image as anisotropic 2D Gaussians with accumulated blending; 2000 FPS. *Bézier Splatting ≈ GaussianImage with curve-structured Gaussians; comparing structured vs free Gaussians quantifies the "cost of editability."*
- **Image-GS** — Zhang et al., SIGGRAPH 2025, arXiv 2407.01866. Content-adaptive 2D Gaussians: gradient-magnitude spawning + progressive error-driven allocation. *Allocation strategies transplantable to curve densification.*
- **Fast 2DGS** — arXiv 2512.12774. Efficient 2D Gaussian fitting with deep Gaussian priors.

## 6. Differentiable text/font vector outlines

- **DeepVecFont** — Wang & Lian, SIGGRAPH Asia 2021 (TOG), arXiv 2110.06688; **DeepVecFont-v2** — CVPR 2023, arXiv 2303.14585. Dual-modality font synthesis with differentiable rasterization refinement.
- **Multi-Implicit Neural Font Representation** — Reddy et al., NeurIPS 2021, arXiv 2106.06866. Fonts as permutation-invariant implicit curve fields preserving corners.
- **VecFontSDF** — Xia et al., CVPR 2023, arXiv 2303.12675. "Parabolic SDF" primitives converting exactly to quadratic Bézier outlines.
- **DualVector** — Liu et al., CVPR 2023, arXiv 2305.10462. Dual-part boolean-combined closed paths, unsupervised from raster, exporting standard SVG glyphs.

*Fonts are the highest-precision test of differentiable Bézier machinery — sharp corners, exact junctions, boolean topology are where Gaussian smoothing hurts most; none of these use a splatting rasterizer yet.*

## 7. Open problems the community explicitly flags

(a) DiffVG's gradient noise and cost; (b) topology collapse and local minima; (c) layer ordering and amodal completeness; (d) manual path-count/abstraction selection; (e) optimized SVGs unstructured and hard to edit; (f) LLM-generated SVG lacks geometric fidelity.

## Gaps and opportunities

- **Splat-to-SVG fidelity gap** — no principled analysis or correction of the appearance mismatch between the Gaussian proxy and exported crisp SVG.
- **Fast renderer × existing pipelines** — LIVE, O&R, SGLIVE, CLIPasso, VectorFusion, SVGDreamer++ are DiffVG-bound; systematic backend swaps are unclaimed.
- **Gradient and mesh fills** — per-Gaussian color along a curve is a free gradient-fill parameterization; export throws it away.
- **Sharp features** — hybrid splat-plus-analytic-corner rasterizers or sharper kernels have no published treatment.
- **Layer order / true alpha compositing** — differentiable ordering that discretizes to SVG stacking remains open in the splat setting.
- **Amodal path completion under splat optimization** — VLM/segmentation occlusion reasoning + fast splat refinement, undone.
- **Densification/pruning theory** — no principled path-budget controller (rate-distortion/MDL) anywhere in the field.
- **LLM-in-the-loop refinement at interactive rates** — no work uses a splat renderer as an RL reward/repair engine.
- **Video/animation** — temporally coherent editable animated SVG from video is nearly untouched.
- **Benchmark vacuum for renderers** — no standard benchmark for differentiable rasterizers themselves (gradient accuracy, convergence, topology robustness).

---

# Appendix B — Literature review: WYSIWYG/editor-native parameterization

(Verbatim survey output.)

## 1. Structured / primitive-based decomposition and shape programs

- **CSGNet** — Sharma et al., CVPR 2018, arXiv 1712.08290. RNN parses 2D/3D shapes into CSG programs over primitives, trained via policy gradients. *Images can be inverted into discrete primitive programs, not just curves.*
- **Learning to Infer Graphics Programs from Hand-Drawn Images** — Ellis et al., NeurIPS 2018, arXiv 1707.09627. Neural proposal of drawing primitives + program synthesis recovering loops/conditionals/variable bindings. *The "program induction on top of primitives" layer; editor repeat/align semantics resemble these loops.*
- **Differentiable Compositing** — Reddy et al., SIGGRAPH Asia 2020, arXiv 2010.08788. Differentiable compositing over discrete, depth-ordered elements; image losses backprop to element positions, ordering, identity. *Closest existing analogue to "optimize a scene of editor objects with a pixel loss," including soft z-order.*
- **Differentiable Blocks World** — Monnier et al., NeurIPS 2023, arXiv 2307.05473. Textured superquadric primitives optimized against photometric loss via differentiable rendering. *3D template: primitive parameters as the optimization variables.*
- **Abstracting Sketches through Simple Primitives** — Alaniz et al., ECCV 2022, arXiv 2207.13543. Strokes mapped to nearest primitive + affine transform under a budget. *Stroke-to-primitive "snapping" as an operator.*
- **CLIPasso** — Vinker et al., SIGGRAPH 2022, arXiv 2202.05822. Budgeted Bézier strokes optimized through DiffVG with CLIP loss. *Abstraction-budget idea transfers to "how few editor objects explain this image."*
- **Chat2SVG** — Wu et al., CVPR 2025, arXiv 2411.16602. LLM writes primitive-restricted SVG template; diffusion-guided optimization refines. *Precedent and foil: structure used for init, not preserved as parameterization.*
- **Sketch-n-Sketch** — Hempel et al., UIST 2019, arXiv 1907.10699. Output-directed programming: direct manipulation of rendered SVG translated into program edits. *Defines what "editable structure" means operationally.*

## 2. Differentiable text and typography

- **De-rendering Stylized Texts** — Shimoda et al., ICCV 2021, arXiv 2110.01890. Parses raster text into string, font, size, effects, inpainted background; verifies via *differentiable text rendering*. *Most on-point prior for "text as an object" — but only isolated text regions.*
- **DeepVecFont / v2** — arXiv 2110.06688, 2303.14585. Dual-modality glyph synthesis with differentiable rasterization refinement.
- **Multi-Implicit Fonts** — arXiv 2106.06866. Corner-preserving implicit glyph sets; a differentiable "font axis."
- **VecFontSDF** — arXiv 2303.12675. Pseudo-SDF primitives converting exactly to quadratic Béziers. *Bridge between implicit differentiability and editor-native output.*
- **DualVector** — arXiv 2305.10462. Boolean dual-part glyph representation learned without vector supervision.
- **Word-As-Image** — arXiv 2303.01818. DiffVG + SDS letter deformation with ACAP and tone legibility regularizers. *Hand-crafted legibility proxies where an OCR loss could be used.*
- **Dynamic Typography** — arXiv 2404.11614. Video-diffusion SDS letterform animation with legibility preservation.
- **Kinetic Typography Diffusion** — arXiv 2407.10476. Guided video diffusion with a glyph-readability loss.
- **TextDiffuser** (arXiv 2305.10855) / **AnyText** (ICLR 2024). Diffusion text-painters with character-aware and OCR-derived losses. *Strongest precedents for OCR-in-the-loop recognizability losses, transplantable to a differentiable text-object renderer.*
- **VecFusion** — arXiv 2312.10540. Raster-then-vector cascaded diffusion for glyphs. *Generative prior to initialize/regularize text-object fitting.*

## 3. Reverse-engineering design files (UI, slides, documents)

- **pix2code** — Beltramelli, EICS 2018, arXiv 1705.07962. CNN+LSTM screenshot→DSL UI code.
- **Screen Parsing** — Wu et al., UIST 2021, arXiv 2109.08763. UI element detection + containment/grouping graph prediction. *The hierarchy-recovery half of screenshot-to-Figma.*
- **Pix2Struct** — Lee et al., ICML 2023, arXiv 2210.03347. Screenshot→simplified-HTML pretraining.
- **Design2Code** — Si et al., 2024, arXiv 2403.03163. Screenshot→HTML/CSS benchmark with element-matching and layout metrics. *Template for measuring structured reconstruction beyond pixel loss.*
- **UI Layers Merger** — Chen et al., 2022, arXiv 2206.13389. Merges fragmented design-file layers to match perceived components. *The inverse of vectorizer fragmentation.*
- **COLE** — Jia et al., 2023, arXiv 2311.16974. Hierarchical intention-to-design generation outputting typed multi-layer editable designs. *Concrete target schema.*
- **PPTAgent** — Zheng et al., 2025, arXiv 2501.03936. Edit-based slide generation on real PPTX + PPTEval. **SlidesBench/AutoPresent** — Ge et al., CVPR 2025, arXiv 2501.00912. Instruction→slide-code benchmark. **DECKBench** — arXiv 2602.13318 (2026), visual reverse-engineering of decks.

## 4. Constraint-based and relational layout

- **LayoutGAN** — Li et al., ICLR 2019. Differentiable wireframe-rendering discriminator over typed elements.
- **Neural Design Network** — Lee et al., ECCV 2020, arXiv 1912.09421. Relational constraints completed into a graph, decoded to layout.
- **Constrained Layout via Latent Optimization (CLG-LO)** — Kikuchi et al., ACM MM 2021, arXiv 2108.00871. *Differentiable alignment and non-overlap losses* — directly reusable as snapping regularizers.
- **LayoutDM** — Inoue et al., CVPR 2023 (and arXiv 2305.02567; LayoutDiffusion arXiv 2303.11589). Discrete diffusion over tokenized layouts with masking/logit-adjustment control.
- **LACE** — Chen et al., ICLR 2024, arXiv 2402.04754. Continuous layout diffusion with differentiable aesthetic-constraint losses.
- **LayoutVLM** — Sun et al., 2024, arXiv 2412.02193. VLM writes layouts *and* self-consistent relational constraints, satisfied via differentiable optimization (3D). *Most direct precedent for "LLM proposes structure/constraints, gradient descent satisfies them."*
- **CanvasVAE** — Yamaguchi, ICCV 2021, arXiv 2108.01249. VAE over typed document elements; introduces **Crello** (real design templates with complete occluded-element data). Follow-ups: **FlexDM** (CVPR 2023, arXiv 2303.14100), **Multimodal Markup Document Models** (arXiv 2409.19051). *Crello is the key dataset: paired renders and editor-native source structure.*

## 5. Image primitives, gradients, hybrid/layered vector scenes

- **Im2Vec** (arXiv 2102.02798); **LIVE** (CVPR 2022); **SuperSVG** (arXiv 2406.09794); **SGLIVE gradient fills** (arXiv 2408.15741); **Layered Vectorization via Semantic Simplification** (arXiv 2406.05404); **LayerTracer** (arXiv 2502.01105 — diffusion transformer generating *cognitively-aligned layered* SVGs, scored partly by shape-count parsimony); classical **gradient meshes** (Sun et al., SIGGRAPH 2007; Lai et al., TOG 2009); **VectorFusion / SVGDreamer / SVGDreamer++**; **Neural Path Representation** (arXiv 2405.10317 — optimizing in a learned structured-shape latent avoids degenerate geometry); **NIVeL** (arXiv 2405.15217); **AmodalSVG** (arXiv 2604.10940); **IconShop/StarVector/OmniSVG** as code-generation complements.

## 6. Editability and structure metrics

- **SVGEditBench** — arXiv 2404.13710 (v2: 2502.19453). Scoreable SVG code-editing tasks for LLMs.
- **VectorEdits** — arXiv 2506.15903. Paired before/after SVGs with natural-language edit instructions.
- **Hierarchical SVG Tokenization** — arXiv 2604.05072 (2026). Evaluates semantic layering, editability, redundancy, code usability; notes raster metrics can't assess structural meaning.
- **VFig** — arXiv 2603.24575 (2026). SVG "cleanliness" as fraction of semantic primitives. **VectorGym** — arXiv 2603.29852.
*All evaluate generated or code-edited SVGs — none evaluates optimization-based reconstructions against ground-truth editor source files.*

## Gaps and opportunities

- **No differentiable renderer optimizes editor-native primitive parameters end-to-end** (rect w/h/corner-radius/rotation, ellipse, arrowheads/caps/dashes) with residual-driven fallback to free-form Béziers.
- **Live text objects inside general scene reconstruction are unsolved** — jointly optimizing (string, discrete font, size, spacing, box geometry) among shapes and images, with OCR legibility loss through a vector renderer.
- **Constraint recovery, not just satisfaction** — detect near-satisfied relations, hard-snap, re-parameterize with constraints as the new DOF.
- **No structure-aware split/merge operators** — object-level "structure collapse" (curves → rect/ellipse/text under an MDL objective) has no published 2D analogue.
- **Hybrid raster+vector documents are not fit differentiably** — jointly deciding image-object vs gradient-shape regions in one optimization.
- **Ground-truth editor files (Crello, PPTX corpora) unused for inverse rendering** — "render → optimize → compare recovered object list to source structure" is a new evaluation paradigm.
- **Editability metrics don't measure round-trip edit behavior** — apply the same edit to recovered and ground-truth files, re-render, compare.
- **Amodal completeness disconnected from editor semantics** (booleans, masks, clip groups).
- **Generative priors combined with differentiable fitting only crudely** — no layout/document prior acting as a regularizer *inside* the rendering-loss optimization.
- **Speed advantage unexploited for structured search** — fast inner-loop rendering makes outer-loop discrete structure search (Gumbel-softmax primitive typing, beam search over groups/constraints) tractable; unpublished.

---

# Appendix C — Literature review: RL(VR) × differentiable rendering

(Verbatim survey output.)

## 1. RLVR fundamentals and recent practice

- **Tülu 3** — 2024, arXiv 2411.15124. Introduces/names RLVR: deterministic verification functions replace learned reward models. *A rasterizer + pixel metric is exactly a verification function.*
- **DeepSeek-R1** — 2025, arXiv 2501.12948 (GRPO from DeepSeekMath, arXiv 2402.03300). Pure large-scale RL with rule-based rewards elicits reasoning; GRPO is the default algorithm in every render-as-reward SVG paper below.
- **RLEF** — 2024, arXiv 2410.02089. Code RL grounded in execution feedback. *Closest code-domain analog of render-look-revise.*
- **Rubrics as Rewards** — 2025, arXiv 2507.17746 (hacking follow-up: arXiv 2605.12474). Checklist rubrics as modular reward criteria. *Template for decomposing visual quality into multi-term rewards.*
- **Scaling Laws for Reward Model Overoptimization** — Gao et al., 2022, arXiv 2210.10760. *Baseline theory for pushing against CLIP/DINO/VLM image rewards.*
- **RLAIF** — 2023, arXiv 2309.00267. AI preference labels match RLHF. *Justifies VLM-as-judge over renders.*
- **Self-Rewarding Language Models** — 2024, arXiv 2401.10020.

## 2. LLMs generating visual code judged by rendering (the core intersection)

- **RLRF: Rendering-Aware RL for Vector Graphics Generation** — 2025, arXiv 2505.20793, NeurIPS 2025. VLM generates SVG rollouts, rendered and compared to input; fidelity + code-efficiency rewards via GRPO. **Explicitly motivated by "the non-differentiability of SVG rendering."** *The exact paradigm — a differentiable renderer attacks its stated core limitation.*
- **Reason-SVG** — 2025, arXiv 2505.24499. "Drawing-with-Thought" + GRPO with hybrid reward (validity, visual/semantic quality, reasoning).
- **SGP-RL / SGP-GenBench** — 2025, arXiv 2509.05208. RLVR text-to-SVG: format-validity gate + SigLIP/DINO rewards on renders; lifts Qwen-2.5-7B to frontier-level. Companion: arXiv 2408.08313 (ICLR 2025 Spotlight) — LLMs near chance at semantic questions about SVG code without rendering. *The "asymmetry of verification" argument.*
- **RRVF** — 2025, arXiv 2507.20766. MLLMs trained from raw images alone: reason → emit code → render → visual-similarity RL reward.
- **TikZilla** — 2026, arXiv 2603.03072. SFT on DaTikZ-V4 + RL with rewards from an image encoder trained via inverse graphics on rendered TikZ. Builds on AutomaTikZ (arXiv 2310.00367), DeTikZify (arXiv 2405.15306, MCTS with compile feedback). *Learned render-space reward encoders beat generic CLIP rewards.*
- **Render-in-the-Loop** — 2026, arXiv 2604.20730. Per-primitive rendering as *state*, not just terminal reward.
- **IntroSVG** — 2026, arXiv 2603.09312 (generator–critic from rendering feedback); **Multi-Task Multi-Reward SVG RL** — 2026, arXiv 2603.16189 (GRPO, four rewards on rasterized output). *2026 norm: multi-term rendered rewards under GRPO — every term computed on an image whose gradient is discarded.*
- **Chart-to-code RL**: MSRL (arXiv 2508.13587), ChartMaster (arXiv 2508.17608), MM-ReCoder (arXiv 2604.01600), Dual Self-Consistency RL (arXiv 2604.06079); evaluated on ChartMimic/Plot2Code.
- **Design2Code** (arXiv 2403.03163); **UICoder** (arXiv 2406.07739 — compiler success + CLIP filters, compile rate 0.03→0.79); **WebRenderBench** (arXiv 2510.04097).
- **Benchmarks/models**: SVGBench, SVGEditBench (arXiv 2404.13710), SVGenius (arXiv 2506.03139), StarVector, OmniSVG.

## 3. RL for image synthesis

- **DDPO** — 2023, arXiv 2305.13301; **DPOK** — 2023, arXiv 2305.16381. Policy gradients on diffusion for non-differentiable rewards; both note hacking without regularization.
- **ImageReward / ReFL** — 2023, arXiv 2304.05977. **ReFL exploits reward-model differentiability to fine-tune by direct backprop instead of RL** — precedent for "differentiable reward → skip policy gradients," one step later in the pipeline than a differentiable rasterizer.
- **Flow-GRPO** — 2025, arXiv 2505.05470; **DanceGRPO** — arXiv 2505.07818. GRPO for flow/diffusion; OCR rewards for text rendering.
- **Reward hacking in T2I RL** — 2026, arXiv 2601.03468; data-regularized RL (arXiv 2512.04332) finds **OCR rewards the most hacking-prone**. *Catalog of failure modes; vector graphics' low-dim parameterization is a plausible structural defense worth testing.*

## 4. Stroke-based rendering with RL — the historical arc

- **SPIRAL** — 2018, arXiv 1804.01118. RL emits brushstroke commands to a non-differentiable engine; GAN discriminator reward. *RL used precisely because the renderer was non-differentiable.*
- **Doodle-SDQ** — 2018, arXiv 1810.05977. Imitation then Q-learning. *Pure RL on drawing needs demonstration bootstrapping — analogous to SFT-before-GRPO.*
- **Learning to Paint** — Huang et al., 2019, arXiv 1903.04411. DDPG over Bézier strokes with a *learned neural renderer* giving the actor pathwise gradients. *First "differentiable renderer inside the RL loop," with Bézier strokes specifically.*
- **Neural Painters** — 2019, arXiv 1904.08410. Learned differentiable proxy renderer.
- **DiffVG** — 2020. Its release triggered the field-wide RL→gradients switch.
- **Stylized Neural Painting** — 2020, arXiv 2011.08114; **Paint Transformer** — 2021, arXiv 2108.03798. Explicitly reject RL for differentiable parameter search / feedforward set prediction. *Once rendering became differentiable, RL disappeared — the open question is whether the 2025 return of RL should re-absorb the gradients.*
- **CLIPDraw / VectorFusion / SVGDreamer** — the pure-gradient pole; weaknesses (slow per-sample optimization, CLIP adversarial doodles) are what an amortized RL-trained policy would fix.

## 5. Hybrid gradient + RL through differentiable simulators

- **PODS** — Mora et al., ICML 2021. Analytic value gradients through the simulator.
- **SHAC** — Xu et al., 2022, arXiv 2204.07137 (AHAC: arXiv 2405.17784). Short-horizon backprop windows + terminal critic.
- **"Do Differentiable Simulators Give Better Policy Gradients?"** — Suh et al., ICML 2022 Outstanding Paper, arXiv 2202.00817. Not always: discontinuities bias/inflate first-order estimates; proposes α-order interpolation. *The frame for when rasterizer gradients (occlusion discontinuities!) should be trusted vs blended.*
- **GI-PPO** — Son et al., NeurIPS 2023, arXiv 2312.08710. Adaptive α-policy integrating analytic environment gradients into PPO. *Most directly reusable for combining GRPO on SVG tokens with pathwise rasterizer gradients.*

## 6. Non-verifiable rewards

- **Deep RL from Human Preferences** — Christiano et al., 2017, arXiv 1706.03741.
- **RL-VLM-F** — 2024, arXiv 2402.03681. VLM pairwise *preferences* over image observations distilled into a reward. *Validated design: prefer pairwise comparison distillation to direct VLM scoring.*
- Plus overoptimization scaling laws, ImageReward/HPS, artifact-hacking studies, rubric rewards; CLIPDraw adversarial scribbles as the canonical single-model-reward hack.

## Gaps and opportunities

- **No one uses the rasterizer's gradients in the RL loop** — all 2025–26 SVG-RL renders non-differentiably and relies on score-function estimators alone; a GI-PPO/SHAC-style hybrid is open and well-scoped.
- **Hybrid discrete/continuous action factorization unexplored for vector graphics** — discrete structure vs continuous parameters, with α-order machinery for gradient trust.
- **Gradient-based reward-hacking probes** — adversarially optimize curve parameters against CLIP/DINO/VLM/OCR rewards in ~10³-dim vector space; no published adversarial reward auditing there.
- **Render-as-verifier under-instrumented** — no study of pixel-metric choice, resolution, or renderer discrepancies (browser vs resvg vs splat softness) on RLVR outcomes.
- **Dense/intermediate credit assignment via rendering is nascent** — per-primitive marginal contribution as advantage/process reward is trivially computable and unused.
- **Amortization gap** — policy emits, splatter refines k steps, RL trains against the refined result (or distills back); no published instance.
- **The historical arc invites a synthesis paper.**
- **OCR and text-in-SVG rewards untouched** — no RL-trained vector generator for legible typography, where differentiable legibility gradients exist.
- **No ImageReward/HPS-equivalent for vector graphics** — preference RM over (render, SVG structure) pairs, usable ReFL-style through the renderer.
- **Verifiable "vector-ness" rewards are low-hanging fruit** — path economy, topology validity, symmetry/snapping, two-scale render consistency.

---

# Appendix D — Literature review: dense models for parameter proposal

(Verbatim survey output.)

## 1. Feedforward vectorization through a differentiable rasterizer

- **DiffVG** — 2020, SIGGRAPH Asia. Demonstrated per-image optimization *and* training predictive networks through the rasterizer.
- **Im2Vec** — 2021, CVPR oral, arXiv 2102.02798. VAE decoding closed Béziers (circle-deformation parameterization), raster-only supervision. *Canonical proof; limited to simple graphics.*
- **SuperSVG** — 2024, CVPR, arXiv 2406.09794. Superpixel decomposition + coarse/refinement models + dynamic path warping self-training. *Closest "dense model proposes many curves for a photo."*
- **Deep Vectorization of Technical Drawings** — 2020, ECCV, arXiv 2003.05471. Clean → per-patch transformer primitives → optimization refinement → merge. *Early explicit predict-then-refine hybrid.*
- **General Virtual Sketching Framework** — 2021, SIGGRAPH. Recurrent stroke drawing with dynamic cropping and a learned "when to stop."
- **Optimize & Reduce** — 2024, AAAI, arXiv 2312.11334. DBSCAN color-cluster init + importance pruning. *The non-learned baselines a neural proposer must beat.*
- **LIVE** — 2022, CVPR, arXiv 2206.04655. Greedy layer-wise addition — precisely the decisions a feedforward model must make in one shot.
- **Vector Grimoire** — 2025, ICML, arXiv 2410.05991. VQ codebook of vector shapes learned under raster supervision + AR transformer. *Discrete-token interface between LLM-style decoders and a splatting renderer.*

## 2. Amortized inference for splatting representations (the core analogy)

- **Splatter Image** (CVPR 2024), **PixelSplat** (CVPR 2024), **MVSplat** (ECCV 2024), **GS-LRM** (ECCV 2024): dense networks emit 3D Gaussian parameter sets in one pass, trained through the differentiable splat renderer; all sidestep cardinality by binding Gaussians to pixels. (Survey: arXiv 2507.14501.)
- **iLRM** — 2025, arXiv 2507.23277. Iterative refinement of splat sets decoupled from pixels. *Learned iterative refinement scales better than single-shot regression.*
- **GaussianImage** — ECCV 2024, arXiv 2403.08551. Per-image optimized unstructured 2D Gaussians — the teacher-style workload follow-ups amortize.
- **Instant GaussianImage** — 2025, arXiv 2506.23479. Network predicts a Gaussian position probability map, Floyd–Steinberg dithered to an *adaptive count*, then attribute regression + short fine-tune. *Closest to amortized 2D splat fitting — unstructured, no curves, no SVG.*
- **Fast 2DGS** — 2025, arXiv 2512.12774. Spatial-distribution network + attribute network + minimal fine-tune. *"Predict layout, then attributes, then briefly refine" as the emerging recipe.*

## 3. Learned init + test-time optimization hybrids

- **Learned Initializations for Coordinate Networks** — Tancik et al., CVPR 2021. MAML/Reptile init → far faster per-signal convergence. *Formal template for warm-starting `fit_image`.*
- **InstantSplat** — 2024, arXiv 2403.20309. DUSt3R foundation-model proposal + short joint splat/pose optimization. *Analog: segmentation/edge foundation model proposes curve layout for splat refinement.*
- **ATT3D** (ICCV 2023) / **LATTE3D** (ECCV 2024, arXiv 2403.15385). Amortized score distillation across prompts, ~400 ms + optional test-time refinement. *Same economics argued for vector graphics.*
- **Tutorial on Amortized Optimization** — Amos, arXiv 2202.00665. Vocabulary: fully- vs semi-amortized, objective- vs regression-based losses.

## 4. Learned optimizers / iterative refinement

- **DeepView** — CVPR 2019. Learned gradient descent for MPIs: CNN ingests estimate + gradient signals, emits updates.
- **RAFT** — ECCV 2020. Correlation volume + GRU iterative updates — the architectural blueprint; **no published work applies RAFT-style learned updates to 2D vector-primitive parameters through a differentiable rasterizer.**

## 5. Diffusion / flow over parameter sets

- **DiffGS** — NeurIPS 2024, arXiv 2410.19657. Latent diffusion over functional 3DGS with arbitrary-count extraction.
- **VecFusion** — CVPR 2024, arXiv 2312.10540. Raster-then-vector cascaded diffusion for precise control points.
- **SketchKnitter** — ICLR 2023 spotlight. Diffusion over stroke points + pen states.
- **SwiftSketch** — SIGGRAPH 2025, arXiv 2502.08642. Image-conditioned stroke diffusion in <1 s, trained on **ControlSketch — a dataset minted by a slow SDS-optimization teacher**. *Clearest published "distill per-image optimization into a student" — sketches only.*
- **LayoutDM** — CVPR 2023, arXiv 2303.08137. PAD tokens for variable-length sets, logit-adjustment constraints.

## 6. Stroke prediction and predict-then-render loops

- **Stylized Neural Painting** — CVPR 2021. Neural proxy renderer + optimal-transport loss for the zero-gradient problem. *The workaround fast analytic splatting makes unnecessary.*
- **Paint Transformer** — ICCV 2021 oral, arXiv 2108.03798. DETR-style stroke queries + per-stroke confidence + coarse-to-fine, self-trained on random synthesized stroke images with no dataset. *Most transferable architecture for one-shot Bézier-splat proposal.*
- **MambaPainter** — SIGGRAPH Asia 2024 posters, arXiv 2410.12524. SSM predicting 100+ strokes in one step.
- **SketchRNN** (arXiv 1704.03477) / **Sketchformer** (arXiv 2002.10381). Autoregressive stopping as the classic cardinality answer.
- **CLIPasso / CLIPascene** — saliency-guided initialization as a primitive learned proposer; the slow teachers SwiftSketch distilled.

## 7. LLM-adjacent decoders (no renderer in the training loop)

- **DeepSVG** — NeurIPS 2020. Hierarchical, non-autoregressive vector decoding; needs vector supervision.
- **StarVector / OmniSVG** — token-loss trained VLM SVG coders; no raster-gradient feedback.
- **Render-in-the-Loop** — 2026, arXiv 2604.20730. Rendering as observational (not gradient) feedback for MLLM decoding.

## 8. Variable-cardinality / set prediction machinery

- **DETR** — ECCV 2020. Queries + Hungarian matching + no-object class; descendants already predict parametric primitives (Paint Transformer strokes, Point2Primitive CAD curves, RoomFormer polygons, 3D wireframe DETRs, arXiv 2606.14811). Bipartite matching needs a teacher curve set; otherwise render loss + confidence pruning or probability-map counts.

## Gaps and opportunities

- **Nobody has built a feedforward parameter proposer for Bézier splatting itself** — curve-bound splats with type/depth/SVG export occupy an empty intersection; no follow-up to 2503.16424 does this as of mid-2026.
- **The 150× backward changes training economics, unexploited** — cheap render losses for large-scale proposer pretraining *and* cheap teacher-dataset generation.
- **Learned iterative refinement for 2D vector primitives is absent** — including feeding true gradients as network inputs.
- **Variable cardinality + topology (open/closed, layers) has no one-shot solution** — a learned density map seeding curves is a concrete opening.
- **SwiftSketch's distillation recipe not applied to full-color filled vectorization.**
- **Depth/layer ordering is never predicted, only inherited** — amortized layer decomposition supervised only by the composited render.
- **LLM/VLM SVG decoders disconnected from differentiable rendering** — continuous heads fine-tuned by raster gradients have no published instance.
- **No semi-amortized benchmark for vectorization** — the {feedforward, +k steps, full optimization} Pareto frontier is unreported.
- **Meta-learned initializations never tried on vector parameter sets.**
- **Diffusion over curve sets with rendering-guided sampling is open.**
