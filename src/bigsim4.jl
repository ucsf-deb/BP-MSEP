#=
Conduct an enormous grid of simulations.
This is *not* a regular part of the MSEP package; it is a user of that package.

File: bigsim4.jl
Author: Ross Boylan
Created: 2023-02-08

The mission:
For logistic model.  Calculate MSEP for various values of abs(z)>tau (values below) for all the predictors (BP, SQ [lam= 0.2, 0.3, 0.4, 0.5], AB [lam=1.4, 1.6, 1.8], CT [lam=1.5, 1.75, 2]).   Do simulation for sigma=0.25, 0.5, 0.75, 1, and 1.25 and plot ratio of MSEP for z_BP divided by MSEP for each of the weighted predictors versus sigma and for each cutoff separately (tau=0, 1.28, 1.5, 1.645, 2, 2.33, and 2.5).  Repeat for cluster sizes of 5, 7, 20, and 100.   Repeat for mu=-1 and -2.   Perhaps just create an overall database with columns for MSEP for each predictor (so 11 columns plus columns to keep track of other parameters) and different rows for values of sigma, tau, cluster size and mu.  


So, we vary the following:
    1. predictor
    2. λ  for predictor in most causes
    3. Cutoff τ for MSEP
    4. σ
    5. cluster size (and number of clusters)
    6. μ

Chronologically, these fall into 3 groups:
    Data Generation: σ, μ, cluster size
    Prediction: predictor, λ
    Assessment: τ

Note the absence of covariates, which potentially could reduce this to the separable problem of
    1. compute predictor | Y for each possible y
    2a. compute distribution of z|Y.  Or even, for MSEP, just compute 
        ``p(z>\tau | Y)= p(Y|z)p(z)/p(Y)``.
    2b. Since computing  ``p(Y)`` is hard, just get z|Y numerically.

Also, we are only doing symmetric cases: no AS predictor, and no use of asymmetric measures of MSEP.
=#

using MSEP
using NamedArrays
using Statistics

#=
We need a way to loop over all possible generators.  The generators have slightly different parameters, since BP has no λ.  Then again, BP can be obtained from one of the other generators via zsimp, assuming the generator is not a cutoff generator.

There are 2 strategies.  The first is to focus only on the generator type and λ.  Return a function which takes the other values as arguments, and produces an evaluator.  The second is to take all the parameters as inputs and directly produce the evaluators.

This is not as simple as in Python because Julia [tends to use iterators rather than generators](https://itnext.io/generators-and-iterators-in-julia-and-python-6c9ace18fa93), and doesn't naturally do resumable coroutines.  The documentation describes `Tasks` as "coroutines" or ["one-shot continuations"](https://docs.julialang.org/en/v1/manual/asynchronous-programming/), but if you `wait` for one to return a value it will not return again, as one-shot implies.More simply, it lacks a `yield` operator *that returns a value*.  There are at least 2 packages that provide macros allowing one to use this style, [ResumableFunctions](https://github.com/BenLauwens/ResumableFunctions.jl) and [FGenerators](https://github.com/JuliaFolds/FGenerators.jl).  The former is only shown with a single `@yield` statement in the function body, and it is unclear to me if it is designed to handle more than 1.  It is also apparently [unmaintained](https://github.com/BenLauwens/ResumableFunctions.jl/issues/56#issuecomment-1401431437).  In contrast, `FGenerators` example has multiple `@yield` statements.  However, the result doesn't seem to be a simple iterator since the documentation says you need `FLoops` to iterate over the result.

It seems safest to stick with the facilities of the base language, which imply either using iterators or using `Channels` to communicate with `Tasks`.  Since direct iteration would require tracking multiple levels to get a location (type of evaluator at the top level, λ below it, possibly other parameters as well), it seems best to rely on `Channel`.  `ResumableFunctions` explicitly avoid `Channel` because of performance, but these are relatively high level loops that are not performance critical for me.

There is a constructor for `Channel` that uses a function taking only a `Channel` instance as an argument, and the result is itself iterable.  So something fairly clean should be possible.
=#

"Holds a list of pairs with an Evaluator Constructor as first argument and a list of λ values as the second"
struct EVRequests
    requests
    order::Int  # default order for quadrature
end

function Base.length(evr::EVRequests)
    sum(length(λs) for (_, λs) in evr.requests)
end

"return brief description of each iterator"
function names(evr::EVRequests)
    # easiest way to get names is to instantiate with random parameters
    r = Array{String}(undef, length(evr))
    # enumerate doesn't work on evr
    # I think it fails because it makes assuptions about the iteration state
    # that aren't true for evr, which uses a Channel.
    i = 1
    for f in evr
        ev = f(0.0, 0.0)
        nm = name(ev)
        if nm == "zBP"
            r[i] = nm
        else
            r[i] = nm * "(λ=" * string(ev.λ) * ")"
        end
        i += 1
    end
    return r            
end

function evrfeed(c::Channel, evr::EVRequests)
    for (ctor, λs) in evr.requests
        for λ in λs
            put!(c, (k, σ)->ctor(λ, k, σ, evr.order))
        end
    end
end

#= Implementation facts
  1. For the definitions to be effective one must define 
  `Base.iterate` not `iterate`.
  2. Previously I found explicit references to iterate must also use `Base.iterate`.  But now I find VSCode complains about them, and they do not seem
  necessary.
  3. I tried returning the `Channel`'s iterator directly from
  `Base.iterate(::EVRequests)`, hoping iteration would be
  delegated to the `Channel` iterator.  But it still looked
  for `Base.iterate(::EVRequest, state)` on the next iteration,
  hence the current design.

Earlier implementations attempted to pull items from
the channel use `take!` while guarding with `isopen`.  But the channel kept getting closed after `isopen` but before 
`take!`, throwing an error.  evrfeed executes in a
coroutine (Task) which does not guarantee an particular
sequencing of its code, the code that closes the channel after
the task exits, and  the tests in the iterator.

So instead I use a channel iterator to control the nastiness.
But the iteration framework will always call
iterate(::EVRequests, state) if I iterate on EVRequests, and so 
I can not simply return the channel, or an iterator on the channel.  And since iterating the channel requires the 
channel as well as the channel state, the iterator state
for EVRequests must include both.

See https://discourse.julialang.org/t/iterator-says-channel-is-closed/95702/3
and the test3.jl code for more on these issues.
=#
function Base.iterate(evr::EVRequests) 
    f(c::Channel) = evrfeed(c, evr)
    mychan = Channel(f)
    r = iterate(mychan)
    if isnothing(r)
        return nothing
    end
    return (r[1], (mychan, r[2]))
end

function Base.iterate(evr::EVRequests, state)
    r = iterate(state[1], state[2])
    if isnothing(r)
        return nothing
    end
    # I think (r[1], state) would also work for next
    # But safer to treat channel state as opaque.
    return (r[1], (state[1], r[2]))
end


#=
Here are the current constructor calls
"Default to zSQ evaluator"
function LogisticSimpleEvaluator(λ, k, σ, integration_order=7)

function LogisticCutoffEvaluator(λ, k, σ, integration_order=7)

### this makes the inner function that in turn makes the evaluator I want
### previous calls were to functions, while this calls the primary structure defntn
# note the capture of λ in the function i the argument of wDensity
function make_zAB_generator(; λ=1.6, k=-1.0, order=5)
    function (σ)
        LogisticSimpleEvaluator(λ, k, σ, order, wDensity((z, λ)-> λ*abs(z)), "zAB", 
        AgnosticAGK(order), "AGK", "Adaptive Gauss-Kronrod")
    end
end
=#

myr = EVRequests([
    # The first needs extra indirection to ignore λ
    ((λ, k, σ, order)->LogisticBPEvaluator(k, σ, order), (0.0)),
    (LogisticSimpleEvaluator, (0.2, 0.3, 0.4, 0.5)),
    (LogisticABEvaluator, (1.4, 1.6, 1.8)),
    (LogisticCutoffEvaluator, (1.5, 1.75, 2.0))
    ],
     7)

#=
for x in myr
    println(description(x(-1.0, 1.0)))
end
println("Total of ", length(myr), " estimands.")
=#

"""
Info at the level of a particular specification of the dataset
There will be multiple times the dataset is simulated
"""
mutable struct DatInfo
    "are all subvariants of estimators complete"
    done::Bool

    "should recompute done"
    checkDone::Bool

    "number of clusters in individual simulated dataset"
    nClusters::Int

    "numer of simulated datasets"
    nSim::Int
end

function DatInfo(nClusters::Int)
    DatInfo(false, false, nClusters, 0)
end

"""
Info at the level of a particular estimator, nested within data spec.
"""
mutable struct EstimInfo
    "are all variant of MSEP complete"
    done::Bool

    "should recompute done"
    checkDone::Bool

    "max number of times this evaluated"
    nCount::Int
end

function EstimInfo()
    EstimInfo(false, false, 0)
end

"""
Info for one particular measure of MSEP
"""
mutable struct MSEPInfo
    "sufficient precision achieved"
    done::Bool

    "next iteration at which to check if precision is sufficient"
    nextCheck::Int

    "each simulation contributes one result, an overall MSEP for relevant set"
    msep::Vector
end

function MSEPInfo(firstCheck=7)
    MSEPInfo(false, firstCheck, Vector())
end

"putting it all together"
mutable struct SimInfo
    "generating data"
    data::NamedArray{DatInfo,3}

    "computing estimates"
    zhat::NamedArray{EstimInfo,4}

    "measures of performance"
    msep::NamedArray{MSEPInfo,5}
end

function SimInfo(evr::EVRequests, μs, σs, τs, clusterSizes)
    estimNames = names(evr)
    dims = ("μ", "σ", "clsize", "zhat", "τ")
    # avoid using numbers as names, since they conflict
    # with indexing. Convert to string instead.
    dimnames = (string.(μs), string.(σs), string.(clusterSizes), estimNames,
            string.(τs))
    dimlen = length.(dimnames)
    # fill puts the same object in every cell
    # Comprehensions create distinct objects.
    dat = NamedArray(
        [ DatInfo(nClusters(nc)) for μ in μs, σ in σs, nc in clusterSizes],
        dimnames[1:3], dims[1:3])
    est = NamedArray(
        [EstimInfo() for μ in μs, σ in σs, nc in clusterSizes, zhat in estimNames],
        dimnames[1:4], dims[1:4])
    err = NamedArray(
        [MSEPInfo() for μ in μs, σ in σs, nc in clusterSizes, zhat in estimNames,
            τ in τs],
        dimnames[1:5], dims[1:5])
    SimInfo(dat, est, err)
end

"record that a computation has achieved sufficient accuracy"
function setDone!(si::SimInfo, i1, i2, i3, i4, i5)
    si.msep[i1, i2, i3, i4, i5].done = true
    si.zhat[i1, i2, i3, i4].checkDone = true
    si.data[i1, i2, i3].checkDone = true
end

"report whether computation at a certain level is done"
function isDone(si::SimInfo, i1, i2, i3, i4, i5)::Bool
    si.msep[i1, i2, i3, i4, i5].done
end

function isDone(si::SimInfo, i1, i2, i3, i4)::Bool
    if si.zhat[i1, i2, i3, i4].checkDone
        si.zhat[i1, i2, i3, i4].done = all(
            (i5)->isDone(si, i1, i2, i3, i4, i5), axes(si.msep, 5)
        )
        si.zhat[i1, i2, i3, i4].checkDone = false
    end
    return si.zhat[i1, i2, i3, i4, i5].done
end

function isDone(si::SimInfo, i1, i2, i3)::Bool
    println("isDone($i1, $i2, $i3)")
    if si.data[i1, i2, i3].checkDone
        si.data[i1, i2, i3].done = all(
            (i4)->isDone(si, i1, i2, i3, i4), axes(si.zhat, 4)
        )
        si.data[i1, i2, i3].checkDone = false
    end
    return si.data[i1, i2, i3].done
end

function isDone(si::SimInfo)
    all((ic)->isDone(si, Tuple(ic)...), CartesianIndices(si.data))
end

"computations finished for given indices"
function finished!(si:SimInfo, i1, i2, i3, i4, i5)
    # nothing to do
end

function finished!(si:SimInfo, i1, i2, i3, i4)
    si.zhat.nCount += 1
end

function finished!(si:SimInfo, i1, i2, i3)
    si.data.nSim += 1
end

"recommended number of clusters to simulate for given cluster size"
function nClusters(clusterSize)::Int
    nT = Threads.nthreads()  # maybe -1 to allow for coordination?
    targetPopulation = 4000
    maxClusters = round(500/nT)*nT
    fClusters = targetPopulation/clusterSize
    # round to nearest nT
    nClusters = round(fClusters/nT)*nT
    return clamp(nClusters, 5, maxClusters)
end


"""
Simulate over the range of scenarios given by the arguments,
which are orthogonal.
Return estimated MSEP and its sd.
Keep performing simulations until uncertainty in MSEP < maxsd
"""
function big4sim(evr::EVRequests; μs=[-1.0, -2.0],
    σs=[0.25, 0.5, 0.75, 1.0, 1.25], 
    τs=[0.0, 1.28, 1.5, 1.645, 2.0, 2.33, 2.5],
    clusterSizes=[5, 7, 20, 100],
    maxsd = 0.5)::SimInfo
    #= Top of loop and data structures concerns the generated
    datasets.
    =#
    siminfo = SimInfo(evr, μs, σs, τs, clusterSizes)
    nIter = 1
    while !isDone(siminfo)
        for (i1, μ) in enumerate(μs), (i2, σ) in enumerate(σs), (i3, ncs) in enumerate(clusterSizes)
            if isDone(siminfo, i1, i2, i3)
                continue
            end
            multi = maker(nclusters=dat[i1, i2, i3].nClusters, nclustersize=ncs, k=μ, σ=σ)
            for (i4, fest) in enumerate(evr)
                if isDone(siminfo, i1, i2, i3, i4)
                    continue
                end
                estiminfo = est[i1, i2, i3, i4]
                # do the estimation. results in multi
                ev = fest(μ, σ)
                for (i5, τ) in enumerate(τs)
                    msepinfo = siminfo.msep[i1, i2, i3, i4, i5]
                    # even if things are good enough, this is cheap to compute
                    push!(msepinfo.msep, msepabs(multi.clusters, τ))
                    # check if done
                    if msepinfo.done || nIter != msepinfo.nextCheck
                        finished!(siminfo, i1, i2, i3, i4, i5)
                        continue
                    end
                    sd = std(msepinfo.msep)
                    if sd/sqrt(nIter) ≤ maxsd
                        setDone!(siminfo, i1, i2, i3, i4, i5)
                    else
                        msepinfo.nextCheck = clamp((sd/maxsd)^2, nIter+5, nIter+100)
                    end
                    finished!(siminfo, i1, i2, i3, i4, i5)
                end
                finished!(siminfo, i1, i2, i3, i4)
            end
            finished!(siminfo, i1, i2, i3)
        end
        nIter += 1
    end
    return siminfo
end

d = big4sim(myr)
println(typeof(d))
println(size(d))
println(d)

