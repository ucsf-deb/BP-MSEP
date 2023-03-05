"""
This specialized integrator is just like AgnosticAGK
except that it only integrates over |z| > λ.
It is likely that both parts will be extremely small.
"""
struct CutoffAGK
    λ::Float64
    order::Int
end

function (cagk::CutoffAGK)(f; segbuf=nothing)
    δ = sqrt(eps())
    # atol=0 is the default, as is rtol=δ
    value, err = quadgk(f, -Inf, -cagk.λ, order=cagk.order,
        atol=0, segbuf=segbuf)
    if err > 2*max(δ, δ*value)
        error("CutoffAGK unable to integrate lower tail accurately")
    end
    v2, err = quadgk(f, cagk.λ, Inf, order=cagk.order,
        atol=0, segbuf=segbuf)
    if err > 2*max(δ, δ*value)
        error("CutoffAGK unable to integrate upper tail accurately")
    end
    return value+v2
end


"""
Evaluates data as produced by `maker()` with a cutoff mixed logistic model
We only use `Y`, a binary indicator, since the model has no observed covariates.
"""
mutable struct  LogisticCutoffEvaluator{TParam,TObjFn} <: CutoffEvaluator
    "cutoff parameter"
    # const requires julia 1.8+
    const λ::TParam

    "parameters for the regression part of the model"
    const k::TParam

    "parameters for random effect distn"
    const σ::TParam

    "order for the numerical integration"
    const integration_order::Int

    ## The constructor is responsible for the following
    "f(z, workarea)= conditional density*normal density
    or, if withZ is true, z*...."
    const f::TObjFn

    "Short name of primary estimand, e.g., zSQ"
    const targetName::String

    "used to integrate f(z) over the parts of the real line beyond λ in absolute value"
    const integrator::CutoffAGK

    "short name of numerical integration method"
    const integratorName::String

    "fuller description integration method"
    const integratorDescription::String
end

function LogisticCutoffEvaluator(λ, k, σ, integration_order=7)
    # we don't use the weights in zSQDensity
    LogisticCutoffEvaluator(λ, k, σ, integration_order, zSQdensity, "zCT", 
    CutoffAGK(λ, integration_order), "CAGK", "Cutoff Adaptive Gauss-Kronrod")
end

"""
Evaluate zhat for cluster defined in work area wa
Because the weighting is expressed through the limits in the 
integrator, we do not call special weight functions.

For the same reasons zsimp is NOT DEFINED for LogisticCutoffEvaluator
"""
function zhat(ev::LogisticCutoffEvaluator, wa::TWorkArea) where {TWorkArea}
    f(z) = ev.f(z, wa)
    wa.objective = justZ
    num = ev.integrator(f, segbuf=wa.segs)
    wa.objective = just1
    den = ev.integrator(f, segbuf=wa.segs)
    return num/den
end

"return a workspace of suitable type for the integrator"
function work(ev::LogisticCutoffEvaluator)
    return work(ev.integrator)
end

function work(integrator::CutoffAGK)
    # size = order looks as if it's the default
    # if so, this is more than enough.
    return alloc_segbuf(size=40)
end