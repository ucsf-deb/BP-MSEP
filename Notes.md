MSEP Simulation
===============

This simulates the performance of a mixed logistic model with weighted predictors.  We use numerical integration to evaluate the predictor given a simulated set of $Y$ data. `bigsum()` (in `evaluator.jl`) and `big4sim()` (in `bigsim4.jl`) are example top-level calls.

`dev`ed the `MixedModels` package to see what it does.  It will be under `~/.julia/dev/`.
Added `DataFrames`

Removing `using MixedModels` reduced time to include `maker.jl` from about 70s to 30s.  I am not currently using it.

I had been testing with `maker.jl` as a standalone program that was not a module.  Put it in main model by moving `using` directives to `MSEP.jl`

Add all packages directly used by `MSEP.jl` to Compiled list in VSCode.  But it didn't survive restart.

Built custom sysimage and enabled the `julia` option to use it.
Required adding `RelocatableFolders` to base `1.8` environment; putting it in `MSEP` environment didn't help.

None of this makes much difference in the time it takes between when I press "Run and Debug" in VSCode and when it gets to the first breakpoint in `harness.jl`: 60-70s

But using `@enter data=maker()` in the REPL does make a huge difference difference. After the first time, the startup delay is trivial.  However, `@run data=maker()` just causes it to hang.  It even hangs when I ask for help on it.

Reproducible Simulations
========================

Summary
-------
We recommend `Random.seed!(k+i)` just before each generation of a new dataset, where `k` is an arbitrary constant and `i` is the iteration number.  So far, only the `big4sim()` code follows this pattern. There are a number of important limitations and caveats to this approach.

Goals
-----
`julia`, like recent versions of `NumPy`, does [not guarantee](https://docs.julialang.org/en/v1/stdlib/Random/#Reproducibility) that setting the same seed will produce the same random numbers across different versions, even different minor versions, of the language.  The same discussion is silent about whether the numbers are reproducible across different architectures (hardware and operating system combinations) using the same version.

The goal for this package is thus simply that the numbers be reproducible given the language version and the architecture.  The main purpose is to enable the rapid re-creation of a particular simulation for testing or debugging.

"rapid" here means something other than re-running the entire simulation to that point.  For much of the current code, complete replay is the only option.

Recommendation
--------------
`big4sim()` uses our currect recommendations: just before each dataset is generated (i.e., call to `maker()`), set the seed with a baseline seed + the simulation number.  For the default (as of `julia` 1.8) `Xoshiro` generator, this is safe--i.e., the streams really are independent--at least according to [some](https://discourse.julialang.org/t/multiple-independent-random-number-streams/98004/3?u=ross_boylan).

Limitations
-----------
As noted, the generated numbers may not be stable across even minor versions of `julia`, and it is unclear if they are reliably stable across architectures.

This pattern of setting the seed is not reliable in general; for many random number generators it induces some correlations across simulations.  The fact that some random people on the internet asserted that `Xoshiro` is immune to such problems is consirably short of proof.

Those who are extra-cautious might want to explore one of the generators in the `abc123` package for `julia`.  However, it seems under-documented and under-maintained.

Varying the number of threads or `julia` `Task`s can change the random numbers.  In `julia` 1.8 we [found](https://github.com/JuliaLang/julia/issues/49522#issue-1685607989) that the random numbers in the main thread varied with the number of threads (and consequently, number of tasks) the program was run with.  Although not shown in that report, this seemed true even when the tasks were spawned *after* generating the random numbers.  We think this behavior was introduced around `julia` 1.7, and it is [slated to be removed](https://github.com/JuliaLang/julia/issues/49064#issue-1632384309) in 1.10.  The problem is that spawning tasks changes the state of the parent random number generator.

More generally, this is a reminder of potential interaction of random number generators with threads and tasks; in the `julia` 1.8 design each `Task` has its own random number generator (and the state change in the parent is in service of keeping them all independent).  Such interaction imply there could be race conditions or sensitivity to seemingly unrelated things like the scheduling algorithm.

Finally, there is a possibility that some of the algorithms (e.g., quadrature, or perhaps optimizers in the future) use random numbers under the hood.  This could throw off the streams, although setting the seed immediately before generating data should limit the mischief.  But, apart from the possibility this could throw off the simulated data, it also means the results might not be completely reproducible.  Some testing with `big4sim()` shows that our current code is not resetting the random generators, at least in the main thread.

Note that all or our current code generates all the simulated data sequentially in the main thread or task.

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
  