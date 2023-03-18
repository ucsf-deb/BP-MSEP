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

using Distributions  # only needed for stub code

using Dates
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

myr = EVRequests([
    # The first needs extra indirection to ignore λ
    ((λ, k, σ, order)->LogisticBPEvaluator(k, σ, order), (0.0)),
    (LogisticSimpleEvaluator, (0.2, 0.3, 0.4, 0.5)),
    (LogisticABEvaluator, (1.4, 1.6, 1.8)),
    (LogisticCutoffEvaluator, (1.5, 1.75, 2.0))
    ],
     7)

#=
We time various calculations, some nested within others.
We record the start time and the duration.
In making estimates of how long things take we most
commonly will want the duration, so we save that
rather then the end time.

WARNING: Nothing in the published documentation guarantees
the type of the duration.  In julia 1.8.5 on MS-Windows,
based on experiment and inspection of the code, it seems to
be Millisecond always.  In the interest of performance I use that,
rather than abstract type like TimePeriod.

For display, consider, e.g.,
canonicalize(round(duration), Minute(1))
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

    "start times of each iteration"
    starts::Vector{DateTime}

    "length of each iteration"
    durations::Vector{Millisecond}
end

function DatInfo(nClusters::Int)
    DatInfo(false, false, nClusters, 0, Vector{DateTime}(), Vector{Millisecond}())
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

    "start times of each iteration"
    starts::Vector{DateTime}

    "length of each iteration"
    durations::Vector{Millisecond}
end

function EstimInfo()
    EstimInfo(false, false, 0, Vector{DateTime}(), Vector{Millisecond}())
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

    "target sd of estimates of MSEP"
    maxSD::Float64
end

function SimInfo(evr::EVRequests, μs, σs, τs, clusterSizes, maxSD)
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
    SimInfo(dat, est, err, maxSD)
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
    return si.zhat[i1, i2, i3, i4].done
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

"computations starting for given indices"
function started!(si::SimInfo, i1, i2, i3, i4, i5)
    # nothing to do. Not worth timing
end

function started!(si::SimInfo, i1, i2, i3, i4)
    push!(si.zhat[i1, i2, i3, i4].starts, now())
end

function started!(si::SimInfo, i1, i2, i3)
    push!(si.data[i1, i2, i3].starts, now())
end

"computations finished for given indices"
function finished!(si::SimInfo, i1, i2, i3, i4, i5)
    # nothing to do
end

function finished!(si::SimInfo, i1, i2, i3, i4)
    push!(si.zhat[i1, i2, i3, i4].durations, now()-last(si.zhat[i1, i2, i3, i4].starts))
    si.zhat.nCount += 1
end

function finished!(si::SimInfo, i1, i2, i3)
    push!(si.data[i1, i2, i3].durations, now()-last(si.data[i1, i2, i3].starts))
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

##### Assessing how much work is done and how much remains
function stderr(si::SimInfo, i1, i2, i3, i4, i5)
    errs = si.msep[i1, i2, i3, i4, i5].msep
    return std(errs)/sqrt(length(errs))
end

"""
Given existing records, estimate the total number of iterations
required to achieve target standard error

To get remaining iterations, subtract the number of existing iterations.
"""
function estimated_iterations(si::SimInfo, i1, i2, i3, i4, i5)
    sd = std(si.msep[i1, i2, i3, i4, i5].msep)
    return round((sd/si.maxSD)^2)
end


function estimated_iterations(si::SimInfo, i1, i2, i3, i4)
    sdmax = max([std(si.msep.msep[i1, i2, i3, i4, i5].msep)] for i5 in axes(si.msep, 5))
    return round((sdmax/si.maxSD)^2)
end

function remaining_iterations(si::SimInfo, i1, i2, i3, i4)
    if isDone(si, i1, i2, i3, i4)
        return 0
    end
    return max(0, 
    estimated_iterations(si, i1, i2, i3, i4) - si.zhat.nCount)
end

function remaining_time(si::SimInfo, i1, i2, i3, i4)
    n = remaining_iterations(si, i1, i2, i3, i4)
    if n == 0
        return Millisecond(0)
    end
    return n*mean_duration(si, i1, i2, i3, i4)
end

# similar for the data level
function remaining_iterations(si:SimInfo, i1, i2, i3)
    if isDone(si, i1, i2, i3)
        return 0
    end
    return max([remaining_iterations(si, i1, i2, i3, i4) for i4 in axes(si.zhat, 4)])
end

"estimated remaining time for estimation.  Ignores data generation time, hence inner."
function remaining_inner_time(si:SimInfo, i1, i2, i3)
    if isDone(si, i1, i2, i3)
        return Millisecond(0)
    end
    return sum([remaining_time(si, i1, i2, i3, i4) for i4 in axes(si.zhat, 4)])
end

"estimate ratio of time including data generation to time without"
function expension_factor(si::SimInfo, i1, i2, i3)
    inner = sum([sum(si.zhat[i1, i2, i3, i4].durations) for i4 in axes(si.zhat, 4)])
    outer = sum(si.data[i1, i2, i3].durations)
    return max(1.0, outer/inner)
end

"estimate of remaining time including data generation overhead"
function remaining_time(si:SimInfo, i1, i2, i3)
    if isDone(si, i1, i2, i3)
        return Millisecond(0)
    end
    return remaining_inner_time(si, i1, i2, i3) *
        expansion_factor(si, i1, i2, i3)
end

function remaining_time(si::SimInfo)
    # CartesianIndices would be more elegant, but the functions above
    # would need to be extended to handle them.
    sum(remaining_time(si, i1, i2, i3) for i1 in axes(si.data, 1),
        i2 in axes(si.data, 2), i3 in axes(si.data, 3))
end

"Not a great guide to remaining work since number of estimates for each outer iteration will vary"
function remaining_iterations(si::SimInfo)
    max(remaining_itertions(si, i1, i2, i3) for  i1 in axes(si.data, 1),
    i2 in axes(si.data, 2), i3 in axes(si.data, 3))
end

"""
Return mean duration of the estimator at the given indices.

Since we always compute all MSEP (one level down) there
are no nasty issues about differing counts in the parts.
When we go up to the si.data level there are such issues.

There are problems computing the mean using `Dates`.  The
exact type of the result of subtracting 2 `DateTime`s is not
specified in the documentation, though it always seems to
be `Millisecond` under julia 1.8.5 on MS Win Server 2019, 64bit.
My type definitions for durations assure they will be
`Millisecond`s, so the code should just blow up if that's not
true.

`mean()` of a collection of `Duration`s, even if they are all
`Millisecond`, does not work reliably.  Unless the mean 
can be exactly converted to `Int64` an `InexactError` will be thrown.
The awkward code below gets around that.  Note that it is not
safe if durations other than `Millisecond` are present, and if any
of them are `CompoundPeriod`s it will throw an error since `value` is
not defined for that type.

For more, see
https://github.com/JuliaLang/julia/issues/15322
https://discourse.julialang.org/t/datetime-time-division/95285

Another approach would be to use the `Unitful` package or some lower-
level call to the system timer via time() (seconds since epoch, floating point)
"""
function mean_duration(si::SimInfo, i1, i2, i3, i4)
    return Millisecond(round(mean([d.value for d in si.zhat[i1, i2, i3, i4].durations])))
end

#= reporting needs a rethink.
The setup below is not really right for loops that accumulate information
as they go, which is what I planned to do.  The struct isn't mutable
and the conversion to hours and minutes is just a pain if I start adding
durations.

Also this seems likely to trigger a lot of redundant calculations.


"quick helper function for report"
struct remain
    minutes::Float64
    hours::Int
    iterations::Int
end

function remain(minutes::Float64, iterations::Int)
    (h, m) = divrem(minutes, 60)
    remain(m, h, iterations)
end

"""
Print a brief summary of estimated remaining work.
"""
function report(io::IO, si::SimInfo)
    remainingMinutes = remaining_time(si)
    (remainingHours, remainingMinutes) = divrem(remainingMinutes, 60)
    remainingIterations0 = remaining_iterations(si)
    
end
=#



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
            started!(siminfo, i1, i2, i3)
            multi = maker(nclusters=siminfo.data[i1, i2, i3].nClusters, nclustersize=ncs, k=μ, σ=σ)
            for (i4, fest) in enumerate(evr)
                if isDone(siminfo, i1, i2, i3, i4)
                    continue
                end
                started!(siminfo, i1, i2, i3, i4)
                estiminfo = siminfo.zhat[i1, i2, i3, i4]
                # do the estimation. results in multi
                ##STUB CODE FOR TESTING
                multi.clusters.zhat = multi.clusters.z .+ rand(Normal(0.0, 0.8), nrow(multi.clusters))
                ## END STUB. Start real code
                #ev = fest(μ, σ)
                for (i5, τ) in enumerate(τs)
                    started!(siminfo, i1, i2, i3, i4, i5)
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

si = big4sim(myr; σs=[0.25, 1.0], τs=[0.0, 1.25], clusterSizes=[5, 100])


