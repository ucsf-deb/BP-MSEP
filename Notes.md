MSEP Simulation

This simulates the performance of a mixed logistic model with weighted predictors to get $z_{\tt BP}$.  We use numerical integration to evaluate the predictor given a simulated set of $Y$ data. `bigsum()` is the top-level call; it's in `evaluator.jl`.

`dev`ed the `MixedModels` package to see what it does.  It will be under `~/.julia/dev/`.
Added `DataFrames`

Removing `using MixedModels` reduced time to include `maker.jl` from about 70s to 30s.  I am not currently using it.

I had been testing with `maker.jl` as a standalone program that was not a module.  Put it in main model by moving `using` directives to `MSEP.jl`

Add all packages directly used by `MSEP.jl` to Compiled list in VSCode.  But it didn't survive restart.

Built custom sysimage and enabled the `julia` option to use it.
Required adding `RelocatableFolders` to base `1.8` environment; putting it in `MSEP` environment didn't help.

None of this makes much difference in the time it takes between when I press "Run and Debug" in VSCode and when it gets to the first breakpoint in `harness.jl`: 60-70s

But using `@enter data=maker()` in the REPL does make a huge difference difference. After the first time, the startup delay is trivial.  However, `@run data=maker()` just causes it to hang.  It even hangs when I ask for help on it.

Packages
========
Notes on some late additions to the environment.

Added so Gadfly can output pdf, ps, png, or anything but svg:
Cairo
Fontconfig

Design of `Evaluator`s
======================
`Evaluator`s evaluate likelihoods.  Or, currently, they have most of the information needed to do so, including the probability density of z, the intercept, the integrator.  But they don't know the layout of
the data, in particular how clusters are defined, or, for that matter, how to compute the conditional probability of the observed outcome given z.  They don't know about the work area or the different computations one might want, e.g., E(z) or E(wz).

Handling a cutoff predictor is awkward.  The default integration from $-\infty$ to $\infty$ with an appropriately computed density (i.e., one that is 0 when $|z| \leq \tau$) should work in principle, and I defined `CTDensity()`, since deleted, to compute it.  In practice it fails very badly.  The value for the denominator is very small; when it is estimated too low it produces very large hat estimates.  I need to integrate explicitly over the proper intervals.  But this means even higher level code needs to change.  To do so it should dispatch on a different type.

`LogisticCutoffEvaluator` is just a small tweak on `LogisticSimpleEvaluator` but---oops---julia does not support inheritance of concrete types.  There are a blizzard of solutions/work-arounds.  See https://discourse.julialang.org/t/composition-and-inheritance-the-julian-way/11231 for an extensive discussion and pointers.  See https://github.com/gcalderone/ReusePatterns.jl for one solution and a nice "bibliography" of other packages in the same space.  My conclusion:
   1. `julia` doesn't do inheritance, and it's best not to fake it.  Try alternate designs.
   2. In particular, use composition and `Lazy.@forward` for delegating methods to the "superclass".
   3. Consider using traits instead.  Basic form is (Tim) Holy traits, https://www.juliabloggers.com/the-emergent-features-of-julialang-part-ii-traits/
   4. I also studied the use of Types in `MixedModels` for ideas.  As I recall, key classes where templates with types for the various components and dispatch on some mix of those.  I think this is different from the Holy traits approach.
   5. There seems to be agreement it's best to define methods using only abstract types.  I'm unconvinced that is always possible or desirable.

For now, I do not use composition (2 above) but simply do an almost entirely redundant parallel implementation, coupled with definitions for a few functions so they work with either type.  Since I currently specify a special type for the integrator, composition isn't an option.

`LogisticCutoffEvaluator` actually does not use a special density function (the `f` field).  Instead it uses a custom integrator, `CutoffAGK`, that returns the sum of the numeric integrals from $(-\infty, -\lambda)$ and $(\lambda, \infty)$. The "standard" integrator `AgnosticAGK` integrates over the whole real line $(-\infty, \infty)$.  I tweaked the tolerances some, but they may still need some work.

Because of the custom integrator, `LogisticCutoffEvaluator` can only provide `zhat`, not `zsimp` and can just reuse the same density used elsewhere.  Another way of putting it is that zsimp computed with the restricted range integrator is the weighted integrator for the cutoff. I could provide 2 integrators for the `LogisticCutoffEvaluator` so that it could compute `zsimp`, but I actually want to reduce such computations and so didn't bother.

Traits
------

What are the dimensions (possible traits) of the problem on which the computation depends?
  1. The outcome variable.  Always binary for us with 0/1 coding.  Or do I have Boolean?
  2. Functional form of relation between $z$ and the outcome.  Always logistic for us.
     1. We have intercept only.
     2. Could have covariates.  Lots of simplifications possible if none.
     3. Always linear for us.
  3. Layout of input data.  How to identify clusters.
     1. Currently each is same size and the elements are consecutive.
     2. My code assume only that elements are consecutive.
  4. Characteristics of weight function.
     1. Discontinuous?
     2. Cutoff?
     3. Inside the exponential?
     4. Possibly offsetting within exponential?
  5. Quantitites that can be computed:
     1. zhat
     2. zBP aka zsimp
     3. individual expectations (currently in `Objective` enumeration)

Misc
----

The current division of responsibilities is both fuzzy and probably suboptimal.

I defined abstract Types `LogisticEvaluator` and `CutoffEvaluator`, thinking I could inherit from both.  But, even for abstract types, `julia` is single inheritance.  So that won't work, and I made `CutoffEvaluator <: LogisticEvaluator`.

To Do
=====
- [ ] Evaluator should get data to describe the estimator and functions to use the description.
  - [x] Short name suitable for use as a variable, possibly returning a Symbol
  - [x] Augmented name for use as a variable, e.g., zSQ_sd
  - [x] longer descriptive text with more details, like $\lambda$ or how integrated.
  - [ ] unclear what the relation should be to contained density function and integrator
- [ ] Use those new naming functions in place of hardcoded names currently found in
  - [ ]  `harness.jl`
  - [ ]  possibly replacing zhat in `bigsim()` of `evaluator.jl`
  - [ ]  `bigbigsim.jl` various functions
- [ ] extend `bigbigsim()` to pass my own evaluator
  - [x] complexity: can't just pass in evaluator since `bigbigsim()` iterates over some of the parameters of the evaluator, specifically $\sigma$.
  - [x] could pass in a function that builds the evaluator
  - [ ] or change fields in evaluator to be mutable
  - [ ] or take the field out of the evaluator, but perhaps pass it in
- [ ] syncronize random numbers across different scenarios
- [ ] don't necessarily need to compute zBP on each simulation.  `LogisticCutoffEvaluator` actually can not compute zBP; that is it does not implement `zsimp()`.  And many of the simulators above it follow through.  But it's ugly.
- [ ] reduce the ~1/2 time spent doing gc
- [ ] try some large cluster sizes
- [ ] How does $\sigma \neq 1$ interact with $R$?
- [ ] Parameter duplication.  E.g., the original `simulate()` function in `evaluator.jl` has `k` and `σ` as parameters even those are also in the `Evaluator`.  Though, in fairness, that particular function constructs the evaluator.
- [ ] Same function, different keyword arguments, e.g., the 2 methods for `simulate()`.
  - [ ] Is that legal?
  - [ ] Do both methods end up with the union of all keyword arguments?
- [x] Method dispatch ignores keyword arguments.
- [ ] How to test it's actually working?
- [ ] Is the doubly conditioned MSEP always $> \tau-z_{FX}$ when $z_{FX}<\tau$?  Logically, it seems it should be.
- [x] Build evaluator for $z_{CT}$.  Non-standard because weight can't go inside exponential. Proper numerical evaluation requires changing the limits of integration.
- [ ] Conduct full set of simulations Chuck wants.
- [ ] Consider how best to present the results.
- [ ] Clear out old/obsolete code, esp from `MSEP.jl` which should be just the master switchboard.
- [ ] Why was I unable to debug into the evaluation of the likelihood?  That is, I put a breakpoint in the inner part of the density evaluation, but it never stopped there even when using `@enter`.  Possibly because the debugger doesn't worl putside the main thread.
- [ ] How might this be integrated into choosing optimal values for $\lambda$?
- [ ] How might this work tuning both confidence interval $\delta$ and $\lambda$ at once?
- [ ] Port to R.
- [ ] Use framework to get estimates of actual variability of estimates for given true $z$ or observed $z_{FX}$.
- [ ] Evaluate error handling in worker threads.  I issue explicit `error()` calls inside the integrators, and other errors can and have arisen.  I think these just silently kill the thread without being bubbled up.
  - [ ] verify that is the current behavior
  - [ ] figure out a better alternative
  - [ ] note the exceptions may be distant from the actual threading code
  