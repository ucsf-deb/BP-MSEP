
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

now() # check if time is wall-clock time
@time clust = bigsim(50; nclusters=200, nclustersize=7, k=-1.0, σ=1.5, λ=0.4, integration_order=5)
now()
groups= groupby(clust, :∑Y)
combine(groups, :zhat=>mean=>:zSQ, :zhat=>std=>:zSQ_sd)
println("MSEP over all clusters = ", msep(clust))
println("For true z > 1.96, MSEP = ", msep(clust, 1.96))

p = plot(clust, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())
p |> PDF("sim_simple8.pdf")

data = maker()
print(data)
