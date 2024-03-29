
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

"""big3sim will eventually be bigger than even bigbigsim.
So it's big^3sim.  But for now the key difference is that it takes as 
a first argument a function which, when called with σ, produces an Evaluator.
We assume it is a LogisticSimpleEvaluator, which means it also has k and the 
order of integration embedded in it.  So those arguments to the previous
function do not appear here.
"""
function big3sim(evaluator_generator, nouter=200; nclusters=500, nclustersize=7,
    σs=[0.25, 0.5, 0.75, 1.0, 1.25], 
    τs=[-Inf, 1.5, 2.0, 2.5])
    # so has_simp survives outside the for loop
    has_simp = false
    errs = NamedArray(Matrix{Float64}(undef, length(σs),  length(τs)),
        (σs, τs), ("σ","τ"))
    errsBP = deepcopy(errs)
    iRow = 1
    for σ in σs
        ev = evaluator_generator(σ)
        has_simp = (name(ev) != "zCT") # primitive test. maybe test class?
        clust = bigsim(ev, nouter; nclusters=nclusters, nclustersize=nclustersize)
        groups= groupby(clust, :∑Y)
        if has_simp
            byY = combine(groups, :zhat=>mean=>name(ev), 
                        :zhat=>std=>name_with_suffix("_sd", ev),
                        :zsimp=>mean=>:zsimp,
                        :zsimp=>std=>:zsimp_sd)
        else
            byY = combine(groups, :zhat=>mean=>name(ev), 
                        :zhat=>std=>name_with_suffix("_sd", ev))
        end
        println("Summary predictors for " * description(ev))
        println(byY)
        println()
        # poor person's dispatch
        if ev.targetName == "zAS"
            errs[iRow, :] = msep(clust, τs)
            if has_simp  # should always be true, but protect against logic changes
                errsBP[iRow, :] = msep(clust, τs, :zsimp)
            end
        else
            errs[iRow, :] = msepabs(clust, τs)
            if has_simp
                errsBP[iRow, :] = msepabs(clust, τs, :zsimp)
            end
        end
        iRow += 1
    end
    if has_simp
        return (errs, errsBP) 
    else
        return (errs, Nothing)
    end
end

### The functions make the functions the are the first argument above.
function make_zAB_generator(; λ=1.6, k=-1.0, order=5)
    function (σ)
        LogisticSimpleEvaluator(λ, k, σ, order, wDensity((z, λ)-> λ*abs(z)), "zAB", 
        AgnosticAGK(order), "AGK", "Adaptive Gauss-Kronrod")
    end
end

function make_zAS_generator(; λ=1.6, k=-1.0, order=5)
    function (σ)
        LogisticSimpleEvaluator(λ, k, σ, order, wDensity((z, λ)-> λ*z), "zAS", 
        AgnosticAGK(order), "AGK", "Adaptive Gauss-Kronrod")
    end
end

function make_zCT_generator(; λ=2.0, k=-1.0, order=5)
    function (σ)
        LogisticCutoffEvaluator(λ, k, σ, order)
    end
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