MSEP Simulation

This simulates the performance of a mixed logistic model with weighted predictors to get $z_{\tt BP}$.  We use numerical integration to evaluate the predictor given a simulated set of $Y$ data.

`dev`ed the `MixedModels` package to see what it does.  It will be under `~/.julia/dev/`.
Added `DataFrames`

Add all packages directly used by `MSEP.jl` to Compiled list in VSCode.

Built custom sysimage and enabled the `julia` option to use it.
Required adding `RelocatableFolders` to base `1.8` environment; putting it in `MSEP` environment didn't help.