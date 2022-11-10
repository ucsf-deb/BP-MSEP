
# little driver to check code in module

using DataFrames, Dates, MSEP, Statistics
using Gadfly, Cairo

Threads.nthreads()
now()
@time x = simulate(nclusters=8000, nclustersize=7);

# without rounding there are 20 unique values
sort(unique(round.(x.clusters.zhat, digits=5)))
first(x.clusters, 10)
combine(groupby(x.clusters, :∑Y), :zhat=>mean, :zhat=>std)


plot(x.clusters, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())

ev = LogisticSimpleEvaluator(1.6, -1., 1.5, 5, wDensity((z, λ)-> λ*abs(z)), AgnosticAGK(5))
now() # check if time is wall-clock time
#@time clust = bigsim(200; nclusters=200, nclustersize=7, k=-1.0, σ=1.5, λ=0.4, integration_order=5)
@time clust = bigsim(ev, 200 ; nclusters=200, nclustersize=7)
now()
groups= groupby(clust, :∑Y)
#combine(groups, :zhat=>mean=>:zSQ, :zhat=>std=>:zSQ_sd, 
combine(groups, :zhat=>mean=>:zAB, :zhat=>std=>:zAB_sd,:zsimp=>mean=>:zsimp,
    :zsimp=>std=>:zsimp_sd)
println("MSEP over all clusters = ", msep(clust))
println("For true z > 1.96, MSEP = ", msep(clust, 1.96))


p = plot(clust, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())
p |> PDF("sim_simple9.pdf")

data = maker()
print(data)

"""
Exercise the worker thread without actually putting it in a separate thread.
For debugging.
"""
function testNoThread()
    ml = maker()
    ml.clusters.zhat .= -100.0 # broadcast to make new columns
    ml.clusters.zsimp .= -100.0 # broadcast to make new columns
    #ev = LogisticSimpleEvaluator(0.4, -1., 1.5)
    # λ is the first input parameter below, and yet wDensity treats it as a variable
    ev = LogisticSimpleEvaluator(1.6, -1., 1.5, 5, wDensity((z, λ)-> λ*abs(z)), AgnosticAGK(5))
    command = Channel(4)

    # prepopulate command
    iCluster = 1
    nclustersize = 7
    put!(command, ((iCluster-1)*nclustersize+1, iCluster*nclustersize, iCluster))
    # and send done
    put!(command, (-1, -1, -1))

    # now run the worker
    MSEP.worker(command, ml, ev)
end

testNoThread()

@run errs, errsBP = bigbigsim(200; nclusters=500);
println("zSQ (λ =0.4) MSEP for |z|>τ")
println(errs)
println()
println("zBP MSEP for |z|>τ")
println(errsBP)
names(errs, 1)
pdat = MSEP.rearrange(errs, errsBP)
# @enter to debug
# Stat.x_jitter(range=0.4), at least in the final position
# jitters the lines but not the points
# shape=:pred,
plot(pdat, x=:σ, y=:MSEP, color=:τ, linestyle=:pred, Scale.color_discrete(), Geom.line, Scale.linestyle_discrete(order=[1, 2])) |> PDF("MSEP-compare.pdf")