
# little driver to check code in module

using DataFrames, Dates, MSEP, Statistics
Threads.nthreads()
now()
@time x = simulate(nclusters=8000, nclustersize=7);

# without rounding there are 20 unique values
sort(unique(round.(x.clusters.zhat, digits=5)))
first(x.clusters, 10)
combine(groupby(x.clusters, :∑Y), :zhat=>mean, :zhat=>std)

using Gadfly, Cairo
plot(x.clusters, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())

now() # check if time is wall-clock time
@time clust = bigsim()
now()

p = plot(clust, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())
p |> PDF("sim_simple7.pdf")

data = maker()
print(data)
