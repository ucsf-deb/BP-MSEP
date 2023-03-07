struct TestR
    requests
end


function test_helper(chan::Channel, t::TestR)
    for (ctor, λs) in t.requests
        for λ in λs
            put!(chan, λ)
        end
    end
end

function Base.iterate(t::TestR)
    f(c::Channel) = test_helper(c, t)
    chan = Channel(f)
    return Base.iterate(t, chan)
end


function Base.iterate(t::TestR, chan)
    if !isopen(chan)
        return nothing
    end
    x = take!(chan)
    if isnothing(x)
        return nothing
    end
    return (x, chan)
end

function Base.length(t::TestR)
    sum(length(λs) for (_, λs) in t.requests)
end

function outside(t::TestR)
    r = Array{String}(undef, length(t))
    for x in t
        print(x,", ")
    end
    print("\n")
end

myt = TestR([(exp, (1.4,)), ])
outside(myt)
#=
for (i, f) in enumerate(myt)
    print(i, ": ", f(-1 ), ", ")
end
#println()
=#