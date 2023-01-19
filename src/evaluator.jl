
"""
Evaluators compute a predictor for a single cluster.  Generally the value is
only good up to a constant of proportionality.  Some evaluators can be asked
to compute more than one kind of value, e.g., w*z or w alone.

Typically an Evaluator will include
   - a definition of the weight function
   - the underlying density of ``z``, usually std normal
   - the conditional density of the outcomes given the data
   - a method of numerical integration, including any parameters to set its 
   behavior

Expected protocol:
zhat(::Evaluator, ::WorkArea)
return an estimate of zhat for the cluster identified in the workarea

And the following functions, provided here with a default Implementation
that assumes certain instance variables are available.
"""
abstract type Evaluator end


"return brief name of primary predictor being evaluated"
# Maybe name(ev::T)  where {T <: Evaluator}::String ?
function name(ev::Evaluator)::String
    ev.targetName
end

"return fuller description of evaluator. full=true gives more detail"
function description(ev::Evaluator, full=false)::String
    des = "$(ev.targetName)(λ=$(ev.λ), k=$(ev.k), σ=$(ev.σ))"
    if full
        return des * ". $(ev.integratorDescription), order $(ev.integration_order)."
    else
        return des * " $(ev.integratorName)($(ev.integration_order))"
    end
end

"return targetName with suffix"
function name_with_suffix(suf::String, ev::Evaluator)::String
    return ev.targetName * suf
end

"""
Evaluators for binary outcomes.
"""
abstract type LogisticEvaluator <: Evaluator end

"""
Evaluators for Cutoff (CT) weighting schemes,
which are w=0 if |z|≤λ, else w=1.

This is unlike many others because
    1. The weight is not a term evaluated inside the exponential.
    2. The discontinuities do not work well with our typical numerical tools.

Logically Cutoff and Logistic are Orthogonal, but since julia is single inheritance,
for now we focus on the case we need.
"""
abstract type CutoffEvaluator <: LogisticEvaluator end


"""
Create one simulated dataset.
Estimate zhat for each cluster with quadrature.
Return results.

This uses the default zSQ weighting scheme.
"""
function simulate(; nclusters=3, nclustersize=4, k=-2.0, σ=1.0, λ=0.4, integration_order=5)::MultiLevel
    ev = LogisticSimpleEvaluator(λ, k, σ, integration_order)
    simulate(ev; nclusters=nclusters, nclustersize=nclustersize)
end

# permits more general weighting and quadrature options
function simulate(ev::LogisticSimpleEvaluator; nclusters=3, nclustersize=4)
    ml::MultiLevel = maker(nclusters = nclusters, nclustersize = nclustersize, k = ev.k, σ = ev.σ)
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

function bigsim(ev::LogisticSimpleEvaluator, nouter=200 ; nclusters=500, nclustersize=7)
    totClusters = nouter*nclusters
    # Pre-allocate full size to reduce memory operations
    results = DataFrame(z=zeros(totClusters), zhat=zeros(totClusters), ∑Y=fill(0x0000, totClusters),
        zsimp=zeros(totClusters),
        iSim=repeat(1:nouter, inner=nclusters), cid=repeat(1:nclusters, nouter), n=nclustersize)
    # values we need to fill in
    copy_vals = [:z, :zhat, :∑Y,  :zsimp]
    ir = 1 # current insertion position in results
    for iSim in 1:nouter
        clust::DataFrame = simulate(ev, nclusters = nclusters, nclustersize = nclustersize).clusters
        results[ir:(ir+nclusters-1), copy_vals] = clust[!, copy_vals]
        ir += nclusters
    end
    return results
end