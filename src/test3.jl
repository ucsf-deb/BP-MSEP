struct TestR
    requests
end


function test_helper(chan::Channel, t::TestR)
    for (ctor, λs) in t.requests
        for λ in λs
            put!(chan, λ)
        end
    end
    #print("Exiting test_helper.\n")
end

#=
Earlier implementations attempted to pull items from
the channel use `take!` while guarding with `isopen`.  But the channel kept getting closed after `isopen` but before 
`take!`, throwing an error.  test_helper executes in a
coroutine (Task) which does not guarantee an particular
sequencing of its code, the code that closes the channel after
the task exits, and  the tests in the iterator.

So instead I use a channel iterator to control the nastiness.
But the iteration framework will always call
iterate(::TestR, state) if I iterate on TestR, and so 
I can not simply return the channel, or an iterator on the channel.  And since iterating the channel requires the 
channel as well as the channel state, the iterator state
for TestR must include both.
=#
function Base.iterate(t::TestR)
    f(c::Channel) = test_helper(c, t)
    chan = Channel(f)
    r = iterate(chan)
    if isnothing(r)
        return nothing
    end
    return (r[1], (chan, r[2]))
end


function Base.iterate(t::TestR, state)
    #println("Iterating.")
    r = iterate(state[1], state[2])
    if isnothing(r)
        return nothing
    end
    # I think (r[1], state) would also work for next
    # But safer to treat channel state as opaque.
    return (r[1], (state[1], r[2]))
end

function Base.length(t::TestR)
    sum(length(λs) for (_, λs) in t.requests)
end

function outside(t::TestR)
    r = Array{String}(undef, length(t))
    i = 1
    for x in t
        r[i] = string(x)
        i += 1
    end
    return r
end

myt = TestR([(exp, (1.4,)) ])

#using InteractiveUtils
#@code_warntype outside(myt)
println(outside(myt))
#=
for (i, f) in enumerate(myt)
    print(i, ": ", f(-1 ), ", ")
end
#println()
=#