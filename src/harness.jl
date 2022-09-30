
# little driver to check code in module

using Dates, MSEP
now()
@time x = simulate(nclusters=8000, nclustersize=7);

# without rounding there are 20 unique values
sort(unique(round.(x.clusters.zhat, digits=5)))

data = maker()
print(data)
