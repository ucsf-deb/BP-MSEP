using DataFrames
using Distributions
using MixedModels
using StatsFuns

"""
MultiLevel holds 2 DataFrames, one with individuals
as rows as one with clusters as row.

The cid's, cluster ids, should be the same for both.

Although the structure is read-only, expected use is that 
columns will be added to the contained data, esp for clusters.
"""
struct MultiLevel
    individuals::DataFrame
    clusters::DataFrame
end

"""
    maker(nclusters=3, nclustersize=4, k=-2.0, σ=1.0)

Return a dataset from a simple mixed model with the indicated sizes and variables.
cluster random effects are N(0, σ) and the only fixed effect is the intercept k.
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
function maker(nclusters=3, nclustersize=4, k=-2.0, σ=1.0)::MultiLevel
    zcluster = rand(Normal(0.0, σ), nclusters)
    clusters = DataFrame(z=zcluster, cid=1:nclusters, n=nclustersize)
    # use clusters.cid is an attempt to preserve whatever
    # scheme I come up with, e.g., still works if I change
    # Categorical variables for IDs.

    # if we have variable cluster sizes, get individual cids with
    # cid = cat(map((id, n)->repeat([id], n), clusters.cid, clusters.n))
    # and similarly with z
    df = DataFrame(cid=repeat(clusters.cid, inner=nclustersize),
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
    return MultiLevel(df, clusters)
end
df = maker()
df
