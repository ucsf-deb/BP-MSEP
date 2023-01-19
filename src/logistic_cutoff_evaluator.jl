"""
Evaluates data as produced by `maker()` with a cutoff mixed logistic model
We only use `Y`, a binary indicator, since the model has no observed covariates.
"""
mutable struct  LogisticCutoffEvaluator <: CutoffEvaluator
    "cutoff parameter"
    # const requires julia 1.8+
    const λ

    "parameters for the regression part of the model"
    const k

    "parameters for random effect distn"
    const σ

    "order for the numerical integration"
    const integration_order::Integer

    ## The constructor is responsible for the following
    "f(z, workarea)= conditional density*normal density
    or, if withZ is true, z*...."
    const f

    "Short name of primary estimand, e.g., zSQ"
    const targetName::String

    "used to integrate f(z) over the parts of the real line beyond λ in absolute value"
    const integrator

    "short name of numerical integration method"
    const integratorName::String

    "fuller description integration method"
    const integratorDescription::String
end


"""
Cutoff density.
a function of (z, wa) that computes the density-like 
value requested in wa.objective.
This is for the CT predictor whose weight is 1 if |z| > λ else 0.
Unlike previous cases:
    1. The weight applies outside the exponential.
    2. There is not an extra level of indirection.
    wDensity is a function that produces a function to be used by the Evaluator.
    This is the function to be used directly.

Implementation note: conceivably there would be additional gains from
realizing that λ is constant and falling back on the indirect approach
in 2.
"""
function CTDensity(z::Float64, wa::WorkArea)
    ev::LogisticSimpleEvaluator = wa.evaluator
    dat::DataFrame = wa.dat
    objective::Objective = wa.objective

    if (objective == justW || objective == WZ) && abs(z) ≤ ev.λ
        return 0.0
    end
    d = exp(-0.5 * z^2)

    for i in wa.i_start:wa.i_end
        Y = dat.Y[i]

        # conditional Y=1 | z
        # next line gets most of the CPU time
        cd = logistic(z*ev.σ + ev.k)
        if Y
            d *= cd
        else
            d *= (1.0-cd)
        end
    end
    if objective == justZ || objective == WZ
        d *= z
    end
    return d
end
