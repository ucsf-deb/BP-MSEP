# post-processing of results

function msep(r::DataFrame)::Float64
    """ r is the result of bigsim.
    Compute MSEP over all clusters
    """
    return mean((r.z-r.zhat).^2)
end

function msep(r::DataFrame, τ::Float64)::Float64
    "compute MSEP for all z > \tau"
    keep = r.z .> τ
    return mean((r[keep, :z] - r[keep, :zhat]).^2)
end

function msep(r::DataFrame, τs)::Vector
    return [msep(r, τ) for τ in τs]
end