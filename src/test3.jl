struct TestR
    requests
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

function test_helper(chan::Channel, t::TestR)
    for (ctor, λs) in t.requests
        for λ in λs
            put!(chan, (x)->ctor(λ+x))
        end
    end
end

function outside(t::TestR)
    for x in t
        print(x(-1.0),", ")
    end
    print("\n")
end

myt = TestR([(exp, [5, 0, 1]), (sqrt, (4))])
outside(myt)
#=
for (i, f) in enumerate(myt)
    print(i, ": ", f(-1 ), ", ")
end
#println()
=#