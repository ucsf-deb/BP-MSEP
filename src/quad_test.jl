"""
See how sensitive results of quadrature are to order in the σ=1 - 1.25 range.

This is NOT part of the main project, though it uses that project
"""

using CSV, Tables
using DataFrames, MSEP, NamedArrays, Random, Serialization
using StatsFuns  #logit

"""Produce a data frame with exactly one cluster for each possible value of
sum of Y.  Position of successes is random.
"""
function one_of_each(; nClusterSize=7, seed=1958590 )
    Random.seed!(seed)
    df = DataFrame(cid=repeat(0:nClusterSize, inner=nClusterSize),
            Y=falses(nClusterSize*(nClusterSize+1)))
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
function quad_test(df::DataFrame; μ=-2.0, σ=1.0, λ=0.4, orders=3:15)
    nClusters = maximum(df.cid)+1
    nClusterSize = sum(df.cid .== 1)
    nOrders = length(orders)
    r = NamedArray(zeros(nClusters, nOrders), (0:(nClusters-1), orders), ("Y", "order"))
    for iOrder in 1:nOrders
        order = orders[iOrder]
        ev = LogisticSimpleEvaluator(λ, μ, σ, order)
        wa = MSEP.WorkArea(df, ev)
        for iCluster in 0:(nClusters-1)
            wa.i_start = 1 + nClusterSize*iCluster
            wa.i_end = nClusterSize*(iCluster+1)
            wa.i_cluster = iCluster+1
            r[iCluster+1, iOrder] = zhat(ev, wa)
        end
    end
    return r
end

#=
testdat = one_of_each(nClusterSize=7)
println("ClusterSize = 7. k = -1.0.  Best Predictors for logistic.")
println("σ=1.0 results follow:")
orders = [3, 4, 5, 6, 7, 9, 11, 13, 20]
r1 =quad_test(testdat, σ=1.0, orders=orders)
println()
println("σ=1.25 results follow:")
r2 =quad_test(testdat, σ=1.25, orders=orders)
=#

#=

"""
Ross- could you send me the values you are getting for numerical quadrature for z_SQ for the logistic model?  I want to check a Gauss-Hermite quadrature routine I have developed before I use it for further investigations.

Could you do clusters of size 5 and 7, sigma of 0.5 and 1, mu=-1 and 0, lambda=0.2 and 0.4?  So 16 scenarios in all.  And give me the predicted values for 0 out of 5, 1 out of 5, 2 out of 5, etc. 

"""
function quad2(ostr; clusterSizes=[5, 7], sigmas=[0.5, 1.0], μs=[-1.0, 0.0], λs=[0.2, 0.4])
    for clusterSize in clusterSizes
        df = one_of_each(nClusterSize = clusterSize)
        for λ in λs, k in μs, σ in sigmas
            ev = LogisticSimpleEvaluator(λ, k, σ)
            wa = MSEP.WorkArea(df, ev)
            for iCluster in 0:clusterSize
                wa.i_start = 1 + clusterSize*iCluster
                wa.i_end = clusterSize*(iCluster+1)
                wa.i_cluster = iCluster+1
                zBP = zsimp(ev, wa)
                zSQ = zhat(ev, wa)
                write(ostr, "$(clusterSize),$(σ),$(k),$(λ),$(iCluster),$(zBP),$(zSQ)\n")
            end
        end
    end
end

ofile = open("quad.csv", "w")
write(ofile, "Quadrature Results for zSQ from MSEP\n")
write(ofile,"From quad_test.jl:quad2() on $(now())\n")
write(ofile, "clusterSize,sigma,mu,lambda,sumY,zBP,zSQ\n")
quad2(ofile)
close(ofile)
=#

#=
"""
Test our latest weird result for 17/20 successes.
"""
function quad3()
    clusterSize = 20
    σ = 0.5
    df = one_of_each(nClusterSize = clusterSize)
    r = quad_test(df, σ = σ)
    open("quad3b.jld", "w") do io
        serialize(io, r)
    end
    return r
end

quad3()
=#

"all ∑Y for clusterSize 100 and params of other simulation"
function quad3c()
    clusterSize = 100
    σ = 1.0
    df = one_of_each(nClusterSize = clusterSize)
    r = quad_test(df, μ = -1.0,  σ = σ)
    open("quad3c.jld", "w") do io
        serialize(io, r)
    end
    open("quad3c.csv", "w") do io
        # CSV.write does not accept a NamedArray directly
        # and header does not autoconvert Int to String
        CSV.write(io, header=[string(n) for n in names(r, 2)], Tables.table(r))
    end
    return r
end

quad3c()
"""
See how quadrature with only relative tolerance handles something that 
has ∫wz = 0.  Should happen at 5/10 if μ=0. It does.
eps()= 2.220446049250313e-16
integrated value = 4.988808992368565e-20
estimated error = 1.7056524166776223e-19
-> error("Unable to integrate accurately")

Try with μ set so p=0.3.  I think 3/10 will not integrate to 0.
"""
function quad4()
    clusterSize = 10
    σ = 1.0
    df = one_of_each(nClusterSize = clusterSize)
    r = quad_test(df, μ = logit(0.3), σ = σ)
    open("quad4b.jld", "w") do io
        serialize(io, r)
    end
    return r
end

r4 = quad4()
print((r4[2:5, :]))