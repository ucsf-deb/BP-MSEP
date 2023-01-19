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