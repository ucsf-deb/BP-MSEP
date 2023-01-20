"""
This specialized integrator is just like AgnosticAGK
except that it only integrates over the |z| > λ.
It is likely that one or both parts will be extremely 
small.
"""
struct CutoffAGK
    λ
    order
end

function (cagk::CutoffAGK)(f; segbuf=nothing)
    δ = sqrt(eps())
    value, err = quadgk(f, -Inf, -cagk.λ, order=cagk.order,
        atol=δ, segbuf=segbuf)
    if err > 2*max(δ, δ*value)
        error("CutoffAGK unable to integrate lower tail accurately")
    end
    v2, err = quadgk(f, cagk.λ, Inf, order=cagk.order,
        atol=δ, segbuf=segbuf)
    if err > 2*max(δ, δ*value)
        error("CutoffAGK unable to integrate upper tail accurately")
    end
    return value+v2
end


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
function zhat(ev::LogisticCutoffEvaluator, wa::WorkArea)
    f(z) = ev.f(z, wa)
    wa.objective = justZ
    num = ev.integrator(f, segbuf=wa.segs)
    wa.objective = just1
    den = ev.integrator(f, segbuf=wa.segs)
    return num/den
end

"""
Defines a computational worker thread

It receives commands through channel.  Those commands are
(i0, i1, iCluster) meaning evaluate the ratio of 
E(wz)/E(w) for cluster iCluster, which has rows i0:i1.
Since weight is 0 or 1, we do E(z)/E(1) over the intervals
of interest.  We need E(1) because we "E(x)" is computed only
to a constant of proportionality.
Write the results back into ml with appropriate locking.

i0<0 means there is no more work and the thread should exit.

ml holds the input data with individual rows and the output
data with a row for each cluster
"""
function worker(command::Channel, ml::MultiLevel, ev::LogisticCutoffEvaluator)
    wa = WorkArea(ml.individuals, ev, work(ev), justZ, 0, 0, 0)
    while true
        i0, i1, iCluster = take!(command)
        if i0 < 0
            # maybe I should make a call to kill thread
            return
        end
        wa.i_start = i0
        wa.i_end = i1
        wa.i_cluster = iCluster
        # do long-running calculations outside the lock
        zh = zhat(ev, wa)
        # there is no zsimp
        # DataFrame is thread-safe for reading, but not writing
        lock(ml.cluster_lock) do
            ml.clusters.zhat[iCluster] = zh
        end
    end
end

"""
Cutoff density.  DOES NOT WORK WELL NUMERICALLY when integrating -Inf to Inf.
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

"return a working space of suitable type for the integrator"
function work(ev::LogisticCutoffEvaluator)
    return work(ev.integrator)
end

function work(integrator::CutoffAGK)
    # size = order looks as if it's the default
    # if so, this is more than enough.
    return alloc_segbuf(size=40)
end