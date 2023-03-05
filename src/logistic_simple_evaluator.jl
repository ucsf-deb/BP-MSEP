"""
Evaluates data as produced by `maker()` with a simple mixed logistic model
We only use `Y`, a binary indicator, since the model has no observed covariates.
"""
mutable struct  LogisticSimpleEvaluator{TParam,TObjFn,TIntegrator} <: Evaluator
    "parameter for weight function"
    # const requires julia 1.8+
    const λ::TParam

    "parameters for the regression part of the model"
    const k::TParam

    "parameters for random effect distn"
    const σ::TParam

    "order for the numerical integration"
    const integration_order::Int

    ## The constructor is responsible for the following
    "f(z, workarea)= w(z)*conditional density*normal density
    or, if withZ is true, z*w(z)*...."
    const f::TObjFn

    "Short name of primary estimand, e.g., zSQ"
    const targetName::String

    "used to integrate f(z) over the real line"
    const integrator::TIntegrator

    "short name of numerical integration method"
    const integratorName::String

    "fuller description integration method"
    const integratorDescription::String
end

"""
Evaluator for conventional Best Predictor.
The same info is available from zsimp for any LogisticSimpleEvaluator;
this class and c'tor provides one with a good description and
access to zBP via zhat for consistency with other evaluators.

**Neither zsimp() nor worker() are defined for LogisticBPEvaluator.**

This struct relies on the inner evaluator being setup correctly,
but does not enforce that.  Use the corresponding constructor.
"""
struct LogisticBPEvaluator{TParam,TObjFn,TIntegrator} <: Evaluator
    inner::LogisticSimpleEvaluator{TParam,TObjFn,TIntegrator}
end

"Default to zSQ evaluator"
function LogisticSimpleEvaluator(λ, k, σ, integration_order=7)
    LogisticSimpleEvaluator(λ, k, σ, integration_order, zSQdensity, "zSQ", 
    AgnosticAGK(integration_order), "AGK", "Adaptive Gauss-Kronrod")
end

"Return evaluator using |z| for the weight function"
function LogisticABEvaluator(λ, k, σ, integration_order=7)
    LogisticSimpleEvaluator(λ, k, σ, integration_order, wDensity((z, λ)-> λ*abs(z)), "zAB", 
    AgnosticAGK(integration_order), "AGK", "Adaptive Gauss-Kronrod")
end

"""
Return Best Predictor evaluator (which does no weighting)
Note the absence of λ from the arguments.
"""
function LogisticBPEvaluator(k, σ, integration_order=7)
    LogisticBPEvaluator(LogisticSimpleEvaluator(0.0, k, σ, integration_order, zSQdensity, "zBP", 
    AgnosticAGK(integration_order), "AGK", "Adaptive Gauss-Kronrod"))
end


"""return fuller description of evaluator. full=true gives more detail
Must override default behavior since λ is not present.
"""
function description(ev::TBPEvaluator, full=false)::String where {TBPEvaluator <: LogisticBPEvaluator}
    rev = ev.inner #real evaluator
    des = "$(rev.targetName)(k=$(rev.k), σ=$(rev.σ))"
    if full
        return des * ". $(rev.integratorDescription), order $(rev.integration_order)."
    else
        return des * " $(rev.integratorName)($(rev.integration_order))"
    end
end

"return brief name of primary predictor being evaluated"
function name(ev::TBPEvaluator)::String where {TBPEvaluator <: LogisticBPEvaluator}
    name(ev.inner)
end

"return targetName with suffix"
function name_with_suffix(suf::String, ev::TBPEvaluator)::String where {TBPEvaluator <: LogisticBPEvaluator}
    return name_with_suffix(ev.inner, suf)
end

"enumerate desired calculation for WorkArea"
@enum Objective justZ justW WZ just1

"""
Working data for a particular thread.
This includes all the information needed to evaluate the function we are integrating,
    since we aren't allowed to pass arguments down other than z.
"""
mutable struct  WorkArea{TEvaluator,TSegs}
    """
    This is the entire data frame.  An individual run will only work with
    a few rows.

    This only needs to be set once at the start of the thread
    """
    const dat::DataFrame
    "Y is just the type-stable version of Y in dat"
    const Y::BitVector

    """
    An evaluator, such as that above.
    Also only set once and shared between threads
    """
    const evaluator::TEvaluator

    "working space for integrator
    This is created at the start but written to constantly."
    segs::TSegs
    
    # The following are set on each evaluation
    "dirty trick to determine whether to integrate over 1, z, w, or wz"
    objective::Objective

    "first row index of cluster of current interest"
    i_start::UInt

    "last row index of cluster, inclusive"
    i_end::UInt

    "index of cluster for output"
    i_cluster::UInt

end

"convenient constructor"
function WorkArea(dat::DataFrame, ev::TEvaluator) where {TEvaluator}
    zip::UInt = 0
    WorkArea(dat, dat.Y, ev, work(ev), WZ, zip, zip, zip)
end

"evaluate (z, w or wz) * density  for a single cluster"
function zSQdensity(z::Float64, wa::TWorkArea) where {TWorkArea}
    ev = wa.evaluator
    dat::DataFrame = wa.dat
    objective::Objective = wa.objective

    #= 
    The initial d is generally the product of weight (defined using zSQ with parameter λ) and
    the standard normal density. By combining them we can avoid many
    overflow problems.

    The exception is for objective == justZ.  In this case, there is no weighting.

    The constant multiplier invsqrt2π for the normal density is unnecessary to the final
    result of the larger computation.  Since we are omitting the Bayes denominator
    anyway, I've left it out.

    z by definition is standard normal, and so k and σ only apply to its
    use for the conditional distribution, cd, not its distribution in
    the first term.

    =#
    if objective == justZ || objective == just1
        # if this doesn't work may want Gauss-Hermite quadrature
        d = exp(-0.5 * z^2)
    else
        d = exp(-0.5 * (1.0 - 2.0 * ev.λ) * z^2)
    end
    for i in wa.i_start:wa.i_end
        Y = wa.Y[i]

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

"""
returns a function fDensity(z, wa) where weight is given by
the function wt(z, λ) which will be evaluated
inside the exponential.
"""
function wDensity(wt )
    return function(z::Float64, wa::WorkArea)  where {WorkArea}
        ev::LogisticSimpleEvaluator = wa.evaluator
        dat::DataFrame = wa.dat
        objective::Objective = wa.objective

        if objective == justZ || objective == just1
            # if this doesn't work may want Gauss-Hermite quadrature
            d = exp(-0.5 * z^2)
        else
            d = exp(-0.5 * z^2 + wt(z, ev.λ))
        end
        for i in wa.i_start:wa.i_end
            Y = wa.Y[i]

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
end

"Evaluate zhat for cluster defined in work area wa"
#TODO: check accuracy of integration
function zhat(ev::LogisticSimpleEvaluator, wa::WorkArea) where {WorkArea}
    f(z) = ev.f(z, wa)
    wa.objective = WZ
    num = ev.integrator(f, segbuf=wa.segs)
    wa.objective = justW
    den = ev.integrator(f, segbuf=wa.segs)
    return num/den
end

#TODO: check accuracy of integration
function zhat(ev::LogisticBPEvaluator, wa::WorkArea) where {WorkArea}
    zsimp(ev.inner, wa)
end

"Evaluate zsimp, a potential analogue of zBP"
function zsimp(ev::LogisticSimpleEvaluator, wa::WorkArea) where {WorkArea}
    f(z) = ev.f(z, wa)
    wa.objective = justZ
    zsimp = ev.integrator(f, segbuf=wa.segs)
    wa.objective = just1
    den1 = ev.integrator(f, segbuf=wa.segs)
    return zsimp/den1
end

"""
Defines a computational worker thread

It receives commands through channel.  Those commands are
(i0, i1, iCluster) meaning evaluate the ratio of 
E(wz)/E(w) for cluster iCluster, which has rows i0:i1.
Write the results back into ml with appropriate locking.

i0<0 means there is no more work and the thread should exit.

ml holds the input data with individual rows and the output
data with a row for each cluster
"""
function worker(command::Channel, ml::MultiLevel, ev::LogisticSimpleEvaluator)
    wa = WorkArea(ml.individuals, ev)
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
        zs = zsimp(ev, wa)
        # DataFrame is thread-safe for reading, but not writing
        lock(ml.cluster_lock) do
            ml.clusters.zhat[iCluster] = zh
            ml.clusters.zsimp[iCluster] = zs
        end
    end
end

"return a working space of suitable type for the integrator"
function work(ev::LogisticSimpleEvaluator)
    return work(ev.integrator)
end

function work(integrator::AgnosticAGK)
    # size = order looks as if it's the default
    # if so, this is more than enough.
    return alloc_segbuf(size=40)
end

function work(ev::LogisticBPEvaluator)
    return work(ev.inner)
end
