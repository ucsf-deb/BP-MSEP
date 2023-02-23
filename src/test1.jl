#=
for use in single-threaded version.
First step: see if it will run without further change.
=#
using DataFrames, Dates, MSEP, Statistics
Threads.nthreads()
now()
using Random
# try one nutty value
# from guts of big3sim
function testnut1()
    λ = 2.0
    k = -1.0
    order = 5
    σ = 1.25
    ev = LogisticCutoffEvaluator(λ, k, σ, order)
    nouter = 1
    Random.seed!(78093580)
    clust = bigsim(ev, nouter; nclusters=400, nclustersize=7)
    groups= groupby(clust, :∑Y)
    byY = combine(groups, :zhat=>mean=>name(ev), 
                :zhat=>std=>name_with_suffix("_sd", ev)
                )

    println("Summary predictors for " * description(ev))
    println(byY)
    println()
end

testnut1()


## check on types
function test1b()
    ev = LogisticSimpleEvaluator(0.4, -1.0, 1.0)
    ml::MultiLevel = maker(nclusters = 4, nclustersize = 5, k = ev.k, σ = ev.σ)
    nT = Threads.nthreads()
    command = Channel(4)
    #@code_warntype MSEP.worker(command, ml, ev)
    # launch workers
    #tasks = [Threads.@spawn worker(command, ml, ev) for i in 1:nT]
    # from inside worker
    wa = MSEP.WorkArea(ml.individuals, ev)
    #@code_warntype zhat(ev, wa)
    # following from inside zhat
    f(z) = ev.f(z, wa)
    wa.objective = MSEP.WZ
    #@code_warntype ev.integrator(f, segbuf=wa.segs)
    @code_warntype ev.f(0.0, wa)
end
test1b()