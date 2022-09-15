using DataFrames
using Distributions
using MixedModels
using StatsFuns

"""
    maker(nclusters=3, nclustersize=4, k=-2)

Return a dataset from a simple mixed model with the indicated sizes and variables
cluster random effects are N(0, 1)
Y binary outcome
z usually unknown random effect
cid cluster id
sid subject id (unique across all clusters)
Output has one row per subject in anticipation of subject level variables
"""
function maker(nclusters=3, nclustersize=4, k=-2)
    zcluster = randn(nclusters)
    df = DataFrame(cid=repeat(1:nclusters, inner=nclustersize),
        sid=1:(nclusters*nclustersize),
        z=repeat(zcluster, inner=nclustersize))
    df.η = k .+ df.z
    df.p = logistic.(df.η)
    df.Y = rand.(Bernoulli.(df.p))
    return df
end
df = maker()
df
