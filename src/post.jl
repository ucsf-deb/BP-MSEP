# post-processing of results

function msep(r::DataFrame)::Float64
    """ r is the result of bigsim.
    Compute MSEP over all clusters
    """
    return mean((r.z-r.zhat).^2)
end

function msep(r::DataFrame, τ::Float64, col = :zhat )::Float64
    "compute MSEP for all z > \tau"
    keep = r.z .> τ
    return mean((r[keep, :z] - r[keep, col]).^2)
end

function msep(r::DataFrame, τs, col = :zhat )::Vector
    return [msep(r, τ, col) for τ in τs]
end

function msepabs(r::DataFrame, τ::Float64, col = :zhat)::Float64
    "compute MSEP for all |z| > τ"
    keep = abs.(r.z) .> τ
    return mean((r[keep, :z] - r[keep, col]).^2)
end

"""
return vector of individual squared errors of prediction.
Return it only for |z| > τ
"""
function sepabs(r::DataFrame, τ::Float64, col = :zhat)::Vector{Float64}
    keep = abs.(r.z) .> τ
    (r[keep, :z] - r[keep, col]).^2
end

function msepabs(r::DataFrame, τs, col = :zhat)::Vector
    return [msepabs(r, τ, col) for τ in τs]
end

function twoToOne(xs)
    """
    Input xs are interpreted as points on a normal distribution used for
    a 2 sided test.  Compute p-value, and translate that to cutoffs for a one sided
    test.
    """
    d = Normal()
    return cquantile.(d, 2*ccdf.(d, xs))
end
