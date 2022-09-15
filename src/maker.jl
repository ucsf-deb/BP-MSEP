using DataFrames
using Distributions
using MixedModels
using StatsFuns

"""
    maker(nclusters=3, nclustersize=4, k=-2.0, σ=1.0)

Return a dataset from a simple mixed model with the indicated sizes and variables
cluster random effects are N(0, σ)
Y binary outcome
cid cluster id
sid subject id (unique across all clusters)
The id's are currently integers, but perhaps should be Categorical.
MixedModels formula syntax says cluster ids should be Categorical,
but in their examples it's actually String.

All the following are usually unobserved
z true random effect (constant within cluster)
η true value on the linear scale
p true probability of success

Output has one row per subject in anticipation of subject level variables
"""
function maker(nclusters=3, nclustersize=4, k=-2.0, σ=1.0)
    zcluster = rand(Normal(0.0, σ), nclusters)
    df = DataFrame(cid=repeat(1:nclusters, inner=nclustersize),
        sid=1:(nclusters*nclustersize),
        z=repeat(zcluster, inner=nclustersize))
    # The remaining elements depend on all previous ones,
    # and DataFrames doesn't seem good at that.  It doesn't work
    # in transform.

    # Since every member of the cluster is identical, the following
    # procedure is a bit inefficient.  But in the future there will be 
    # individual covariates, and this is the right flow for that.
    df.η = k .+ df.z
    df.p = logistic.(df.η)
    df.Y = rand.(Bernoulli.(df.p))
    return df
end
df = maker()
df
