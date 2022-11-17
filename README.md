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
- [ ] How does $\sigma \neq 1$ interact with $R$?
- [ ] Parameter duplication.  E.g., the original `simulate()` function in `evaluator.jl` has `k` and `σ` as parameters even those are also in the `Evaluator`.  Though, in fairness, that particular function constructs the evaluator.
- [ ] Same function, different keyword arguments, e.g., the 2 methods for `simulate()`.
  - [ ] Is that legal?
  - [ ] Do both methods end up with the union of all keyword arguments?
- [x] Method dispatch ignores keyword arguments.
- [ ] How to test it's actually working?
- [ ] Is the doubly conditioned MSEP always $> \tau-z_{FX}$ when $z_{FX}<\tau$?  Logically, it seems it should be.
- [ ] Build evaluator for $z_{CT}$.  Non-standard because weight can't go inside exponential.
- [ ] Conduct full set of simulations Chuck wants.
- [ ] Consider how best to present the results.
- [ ] Clear out old/obsolete code, esp from `MSEP.jl` which should be just the master switchboard.
- [ ] Why was I unable to debug into the evaluation of the likelihood?  That is, I put a breakpoint in the inner part of the density evaluation, but it never stopped there even when using `@enter`.
- [ ] How might this be integrated into choosing optimal values for $\lambda$?
- [ ] How might this work tuning both confidence interval $\delta$ and $\lambda$ at once?
- [ ] Port to R.
- [ ] Use framework to get estimates of actual variability of estimates for given true $z$ or observed $z_{FX}$.
  