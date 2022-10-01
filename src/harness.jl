
# little driver to check code in module

using DataFrames, Dates, MSEP, Statistics
now()
@time x = simulate(nclusters=8000, nclustersize=7);

# without rounding there are 20 unique values
sort(unique(round.(x.clusters.zhat, digits=5)))
first(x.clusters, 10)
combine(groupby(x.clusters, :∑Y), :zhat=>mean, :zhat=>std)

using Gadfly
plot(x.clusters, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())

@time clust = bigsim()

plot(clust, x=:z, y=:zhat, Geom.histogram2d(ybincount=30, xbincount=50), Geom.smooth())
data = maker()
print(data)
