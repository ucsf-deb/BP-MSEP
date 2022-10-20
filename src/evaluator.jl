
"""
Evaluators compute a predictor for a single cluster

Evaluators, when instantiated, are functions called with a single argument
a SubDataFrame, i.e., data for a single cluster.  The return ``hat z`` for that cluster.
Typically this is done by evaluating ``E(wz)/E(w)`` where ``w`` is a weighting function and
``z`` is the random variable of interest.  And typically that is done numerically.

Typically and Evaluator will include
   - a definition of the weight function
   - the underlying density of ``z``, usually std normal
   - the conditional density of the outcomes given the data
   - a method of numerical integration, including any parameters to set its 
   behavior
"""
abstract type Evaluator end
"""
Evaluates data as produced by `maker()` with a simple mixed logistic model
We only use `Y` from the SubDataFrame, a binary indicator, since it has
no observed covariates.
"""
mutable struct  LogisticSimpleEvaluator <: Evaluator
    "parameter for weight function"
    # const requires julia 1.8+
    const λ

    "parameters for the regression part of the model"
    const k

    "parameters for random effect distn"
    const σ

    "order for the numerical integration"
    const integration_order::Integer

    ## The constructor is responsible for the following
    "f(z, workarea)= w(z)*conditional density*normal density
    or, if withZ is true, z*w(z)*...."
    const f

    "used to integrate f(z) over the real line"
    const integrator

end

function LogisticSimpleEvaluator(λ, k, σ, integration_order=7)
    LogisticSimpleEvaluator(λ, k, σ, integration_order, zSQdensity, AgnosticAGK(integration_order))
end

"enumerate desired calculation for WorkArea"
@enum Objective justZ justW WZ just1

"""
Working data for a particular thread.
This includes all the information needed to evaluate the function we are integrating,
    since we aren't allowed to pass arguments down other than z.
"""
mutable struct  WorkArea
    """
    This is the entire data frame.  An individual run will only work with
    a few rows.

    This only needs to be set once at the start of the thread
    """
    const dat::DataFrame

    """
    An evaluator, such as that above.
    Also only set once and shared between threads
    """
    const evaluator::Evaluator

    "working space for integrator
    This is created at the start but written to constantly."
    segs
    
    # The following are set on each evaluation
    "dirty trick to determine whether to integrate over z, w, or wz"
    objective::Objective

    "first row index of cluster of current interest"
    i_start::UInt

    "last row index of cluster, inclusive"
    i_end::UInt

    "index of cluster for output"
    i_cluster::UInt

end

"evaluate (z, w or wz) * density  for a single cluster"
function zSQdensity(z::Float64, wa::WorkArea)
    ev::LogisticSimpleEvaluator = wa.evaluator
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
    wa = WorkArea(ml.individuals, ev, work(ev), WZ, 0, 0, 0)
    f(z) = ev.f(z, wa)
    # g(z) = z*f(z) might be faster than the withZ trick
    while true
        i0, i1, iCluster = take!(command)
        if i0 < 0
            # maybe I should make a call to kill thread
            return
        end
        wa.i_start = i0
        wa.i_end = i1
        wa.objective = WZ
        num = ev.integrator(f, segbuf=wa.segs)
        wa.objective = justW
        den = ev.integrator(f, segbuf=wa.segs)
        wa.objective = justZ
        zsimp = ev.integrator(f, segbuf=wa.segs)
        wa.objective = just1
        den1 = ev.integrator(f, segbuf=wa.segs)
        # DataFrame is thread-safe for reading, but not writing
        lock(ml.cluster_lock) do
            ml.clusters.zhat[iCluster] = num/den
            ml.clusters.zsimp[iCluster] = zsimp/den1
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

"""
Create one simulated dataset.
Estimate zhat for each cluster with quadrature.
Return results.
"""
function simulate(; nclusters=3, nclustersize=4, k=-2.0, σ=1.0, λ=0.4, integration_order=5)::MultiLevel
    ml::MultiLevel = maker(nclusters = nclusters, nclustersize = nclustersize, k = k, σ = σ)
    ev = LogisticSimpleEvaluator(λ, k, σ, integration_order)
    ml.clusters.zhat .= -100.0 # broadcast to make new columns
    ml.clusters.zsimp .= -100.0 # broadcast to make new columns
    nT = Threads.nthreads()
    command = Channel(2*Threads.nthreads())
    # launch workers
    tasks = [Threads.@spawn worker(command, ml, ev) for i in 1:nT]
      
    # feed them jobs
    for iCluster in 1:nclusters
        put!(command, ((iCluster-1)*nclustersize+1, iCluster*nclustersize, iCluster))
    end
    # let each know there is no more work
    for i in 1:nT
        put!(command, (-1, -1, -1))
    end

    # wait for them to finish
    for t in tasks
        wait(t)
    end

    return ml
end

"""Runs many simulations and returns the cluster level results
Results columns
z   true z of cluster
zhat the zSQ estimator from E(wz)/E(z)
∑Y  total successes in cluster
zsimp  Posterior mean, no weighting
iSim    simulation number
cid     cluster id (only unique within iSim)
n       cluster size
"""
function bigsim(nouter=200; nclusters=500, nclustersize=7, k=-2.0, σ=1.0, λ=0.4, integration_order=5)::DataFrame
    totClusters = nouter*nclusters
    # Pre-allocate full size to reduce memory operations
    results = DataFrame(z=zeros(totClusters), zhat=zeros(totClusters), ∑Y=fill(0x0000, totClusters),
        zsimp=zeros(totClusters),
        iSim=repeat(1:nouter, inner=nclusters), cid=repeat(1:nclusters, nouter), n=nclustersize)
    # values we need to fill in
    copy_vals = [:z, :zhat, :∑Y,  :zsimp]
    ir = 1 # current insertion position in results
    for iSim in 1:nouter
        clust::DataFrame = simulate(nclusters = nclusters, nclustersize = nclustersize, k = k, σ = σ).clusters
        results[ir:(ir+nclusters-1), copy_vals] = clust[!, copy_vals]
        ir += nclusters
    end
    return results
end