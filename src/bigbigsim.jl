
function bigbigsim(nouter=200; nclusters=500, nclustersize=7, k=-1.0, 
    σs=[0.25, 0.5, 0.75, 1.0, 1.25], 
    τs=[-1.0, 1.5, 2.0, 2.5],
    λ=0.4,
    integration_order=5)
    errs = NamedArray(Matrix{Float64}(undef, length(σs),  length(τs)),
        (σs, τs), ("σ","τ"))
    errsBP = deepcopy(errs)
    iRow = 1
    for σ in σs
        clust = bigsim(nouter; nclusters=nclusters, nclustersize=nclustersize,
                k=k, σ=σ, λ=λ, integration_order=integration_order)
        groups= groupby(clust, :∑Y)
        byY = combine(groups, :zhat=>mean=>:zSQ, 
                        :zhat=>std=>:zSQ_sd,
                        :zsimp=>mean=>:zsimp,
                        :zsimp=>std=>:zsimp_sd)

        println("Summary predictors for σ =", σ)
        println(byY)
        println()
        errs[iRow, :] = msepabs(clust, τs)
        errsBP[iRow, :] = msepabs(clust, τs, :zsimp)
        iRow += 1
    end
    return (errs, errsBP) 
end


"""
Unpack the matrix in errs into the already allocated df
starting at row i0

Intended only for use by rearrange() and expecting errs to be
::NamedArray, though it might work for other types.

Note that df should already have a description of the underlying
predictor in one of the columns.
"""
function rearrange!(df::DataFrame, errs, i0=1 )
    for (name, v) in enamerate(errs)
        (σ, τ) = name
        df.σ[i0] = σ
        df.τ[i0] = τ
        df.MSEP[i0] = v
        i0 += 1
    end
end
"""
Convert the error reports generated above into a format
more suitable for plotting.
"""
function rearrange(errs, errsBP)::DataFrame
    (nσ, nτ) = size(errs)
    n = nσ * nτ
    # The initial contents of σ and τ are mostly intended to get
    # the type and dimensions right.  The values will be overriden in enumeration
    # order.
    r = DataFrame(pred = repeat(["zSQ", "zBP"], inner=n),
            σ=repeat(names(errs, 1), 2*nτ),
            τ=repeat(names(errs, 2), 2*nσ),
            MSEP=zeros(eltype(errs), 2*n))
    rearrange!(r, errs)
    rearrange!(r, errsBP, n+1)
    return r
end