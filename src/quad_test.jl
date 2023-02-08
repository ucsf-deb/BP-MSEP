"""
See how sensitive results of quadrature are to order in the σ=1 - 1.25 range.

This is NOT part of the main project, though it uses that project
"""

using DataFrames, MSEP, NamedArrays, Random

"""Produce a data frame with exactly one cluster for each possible value of
sum of Y.  Position of successes is random.
"""
function one_of_each(; nClusterSize=7, seed=1958590 )
    Random.seed!(seed)
    df = DataFrame(cid=repeat(0:nClusterSize, inner=nClusterSize),
            Y=fill(false, nClusterSize*(nClusterSize+1)))
    indices = collect(1:nClusterSize)
    for i in 1:nClusterSize
        Random.randperm!(indices)
        df.Y[i*nClusterSize .+ indices[1:i]] .= true
    end
    return df
end

"""
For given data and assumed true sd of random effects,
compute results for each cluster for each of the requested
integration orders.
Returns a named array in which each row represents a single cluster (named by cid)
and each column a particular order for the quadrature (name is the order)
"""
function quad_test(df::DataFrame; σ=1.0, orders=3:15)
    nClusters = maximum(df.cid)+1
    nClusterSize = sum(df.cid .== 1)
    nOrders = length(orders)
    r = NamedArray(zeros(nClusters, nOrders), (0:(nClusters-1), orders), ("Y", "order"))
    λ = 0.4
    k = -1.0
    for iOrder in 1:nOrders
        order = orders[iOrder]
        ev = LogisticSimpleEvaluator(λ, k, σ, order)
        zip::UInt = 0
        wa = MSEP.WorkArea(df, ev, MSEP.work(ev), MSEP.WZ, zip, zip, zip)
        for iCluster in 0:(nClusters-1)
            wa.i_start = 1 + nClusterSize*iCluster
            wa.i_end = nClusterSize*(iCluster+1)
            wa.i_cluster = iCluster+1
            r[iCluster+1, iOrder] = zsimp(ev, wa)
        end
    end
    return r
end

testdat = one_of_each(nClusterSize=7)
println("ClusterSize = 7. k = -1.0.  Best Predictors for logistic.")
println("σ=1.0 results follow:")
orders = [3, 4, 5, 6, 7, 9, 11, 13, 20]
r1 =quad_test(testdat, σ=1.0, orders=orders)
println()
println("σ=1.25 results follow:")
r2 =quad_test(testdat, σ=1.25, orders=orders)
