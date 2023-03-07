#=
Bare minimum from bigsim4.jl to test problems iterating.
=#

#using MSEP

"Holds a list of pairs with an Evaluator Constructor as first argument and a list of λ values as the second"
struct EVRequests
    requests
    order::Int  # default order for quadrature
end

function evrfeed(c::Channel, evr::EVRequests)
    for (ctor, λs) in evr.requests
        for λ in λs
        #    put!(c, (k, σ)->ctor(λ, k, σ, 7 )) # last arg was evr.order
        put!(c, λ)
        end
    end
end

function Base.iterate(evr::EVRequests) 
    f(c::Channel) = evrfeed(c, evr)
    mychan = Channel(f)
    return Base.iterate(evr, mychan)
end

function Base.iterate(evr::EVRequests, chan)
    if !isopen(chan)
        return nothing
    end
    x = take!(chan)
    if isnothing(x)
        return nothing
    end
    return (x, chan)
end

myr = EVRequests([
    # The first needs extra indirection to ignore λ
    #((λ, k, σ, order)->LogisticBPEvaluator(k, σ, order), (0.0)),
    #(LogisticSimpleEvaluator, (0.2, 0.3, 0.4, 0.5)),
    (exp, (1.4,)),
    #(LogisticCutoffEvaluator, (1.5, 1.75, 2.0))
    ],
     7)

#=
this works OK

for x in myr
    println(description(x(-1.0, 1.0)))
end
=#

function Base.length(evr::EVRequests)
    sum(length(λs) for (_, λs) in evr.requests)
end

function names(evr::EVRequests)
    r = Array{String}(undef, length(evr))
    i = 1
    for f in evr
        #=
        ev = f(0.0, 0.0)

        nm = name(ev)
        if nm == "zBP"
            r[i] = nm
        else
            r[i] = nm * "(λ=" * string(ev.λ) * ")"
        end
        =#
        r[i] = string(f)
        i += 1
    end
    return r            
end

println(names(myr))
