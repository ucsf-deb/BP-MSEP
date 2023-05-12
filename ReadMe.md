Better Predictors -- Mean Square Error of Prediction
====================================================

This project implements some [new approaches](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1938583) to predicting extreme values in mixed models, as described in [1].  They rely on various weighting functions that make the estimates more or less sensitive to large values of $z$, the standardized random effect.  The code can also evaluate the traditional best predictor.

It also includes tools to assess the quality of those predictions, often using simulations.

Currently the project focuses on mixed logistic regressions without covariates.

Warnings
========
This is a work in progress.  Some of the code in the project may not even run, if interfaces have changed since it was written.  They may change again in the future.

Although this project is `BP-MSEP` on github, the underlying package is called `MSEP`.

Code that uses the package is not well-separated from code that defines the package.

Pointers
========

`Notes.md` has notes and history of the implementation.

`Project.toml` is a standard `julia` project definition, listing required packages.

`src/` has most of the code.  Highlights:
   * `MSEP.jl` is the top-level definition of the project.
   * `bigsim4.jl` is an example of a large-scale simulation of different estimators.
   * `simulate.jl` has some other simulations
   * `harness*.jl` are other top level invocations.
   * `quad_plot.jl` has some examples of direct evaluation of the likelihood without simulation.
   * `maker.jl` generates simulated datasets
   * `evaluator.jl` includes extended discussion of this central concept.
   * `post.jl` does post-processing after data are simulated and estimated.  The mean square error of prediction is only one of the available measures.

Tools
=====

The code is all in [julia](https://julialang.org/), developed using `julia` 1.8.5.  It may work with earlier versions, perhaps with small modifications.

The central estimation is multi-threaded, and so will benefit from a machine that has multiple CPU's or cores. `-t` is a `julia` command line option to specify how many threads to use.  The parallelism is at the level of clusters.  There is a chance the code may deadlock if run single-threaded, though we only experienced that with earlier versions of `julia` (the scheduler is one of the things that has changed across `julia` versions).

We used [MS Visual Studio Code](https://code.visualstudio.com/) and the `julia` addons, though you are free to use other tools.  `Visual Studio Code` is a program editor, distinct from the more heavyweight `Visual Studio`, which includes compilers.

`julia` and `VSCode` are both available on multiple platforms, free as in beer and likely free as in speech, though you should check the licenses if you are concerned.

You will need to assemble a `julia` environment that includes the necessary packages.  Use the `instantiate` command of the `julia package manager` to get them.

Development was on `MS-Windows Server`.  It has no known Windows dependencies.

Acknowledgements
================
This work is supported in part by funds from the National Institutes of Health (R01AG071535).

Changes
=======
v0.2.0 2023-05-10 initial public release
v0.2.1 pre-release
   * Acknowledge funding.



References
==========
[1] Improving Predictions When Interests Focuses on Extreme Random Events (Jul, 10.1080/01621459.2021.1938583, 2021)
By:McCulloch, CE (McCulloch, C. E.) ; Neuhaus, JM (Neuhaus, J. M.)
Volume117
JOURNAL OF THE AMERICAN STATISTICAL ASSOCIATION Issue 538 Page 1043-+
DOI10.1080/01621459.2022.2060607
Published APR 3 2022