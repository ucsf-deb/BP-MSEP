#=
Generate data for CEM:
  all the predicted values for cluster size 100 for mu=-1, sigma=1.  Also the values of z and the values of the total number of successes.    
=#
using CSV
using DataFrames
using Dates
using MSEP
using Random

function do_it()
    μ = -1.0
    σ = 1.0
    λ = 0.4
    nclusters = 50
    nclustersize = 100
    Random.seed!(2345821)
    ev = LogisticSimpleEvaluator(λ, μ, σ)
    m = simulate(ev, nclusters=nclusters, nclustersize=nclustersize)
    ofile = open("CEM01.csv", "w")
    write(ofile, "Adaptive Gauss-Kronrod Results for zSQ($(λ)) from MSEP\n")
    write(ofile, "For random intercept $(μ), sd = $(σ), $(nclusters) clusters of size $(nclustersize) each.\n")
    write(ofile,"From CEM01.jl on $(now())\n")
    clus = m.clusters[:, [1, 2, 4, 5]] # drop redundant n and zsimp which is not filled in
    rename!(clus, ["truez", "id", "sumY", "zhat"])
    CSV.write(ofile, clus, append=true, writeheader=true )#bom=true makes Windows open extended characters correctly
    close(ofile)
end

do_it()
