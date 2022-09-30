MSEP Simulation

This simulates the performance of a mixed logistic model with weighted predictors to get $z_{\tt BP}$.  We use numerical integration to evaluate the predictor given a simulated set of $Y$ data.

`dev`ed the `MixedModels` package to see what it does.  It will be under `~/.julia/dev/`.
Added `DataFrames`

Removing `using MixedModels` reduced time to include `maker.jl` from about 70s to 30s.  I am not currently using it.

I had been testing with `maker.jl` as a standalone program that was not a module.  Put it in main model by moving `using` directives to `MSEP.jl`

Add all packages directly used by `MSEP.jl` to Compiled list in VSCode.  But it didn't survive restart.

Built custom sysimage and enabled the `julia` option to use it.
Required adding `RelocatableFolders` to base `1.8` environment; putting it in `MSEP` environment didn't help.

None of this makes much difference in the time it takes between when I press "Run and Debug" in VSCode and when it gets to the first breakpoint in `harness.jl`: 60-70s

But using `@enter data=maker()` in the REPL does make a huge difference difference. After the first time, the startup delay is trivial.  However, `@run data=maker()` just causes it to hang.  It even hangs when I ask for help on it.