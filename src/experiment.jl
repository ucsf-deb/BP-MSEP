#= These are some early experiments, likely no longer relevant.
Uses MSEP but not part of the module.

Some of the code may need to be updated to work with current MSEP.
=#
using MSEP

"""
evaluate integrals of various functions with varying number of quadrature points
This is very special purpose code.
"""
struct Experiment
    "array of # of points to try"
    npoints
    "array of functions to integrate"
    funs
    "conditional likelihood(Y|z) within constant factor"
    condY

    # next terms computed rather than passed in
    "array of NormalGH for quadrature"
    quad_nodes

    "array of final functions to integrate over"
    fused_funs
end

"Experiment constructor"
function Experiment(npoints;
    funs=[z -> 1.0, identity, (z) -> exp(0.4 * z^2), (z) -> z * exp(0.4 * z^2)],
    condY=(z) -> logistic(z - 2))
    quad_nodes = [NormalGH(n) for n = npoints]
    fused = [z -> f(z) * condY(z) for f = funs]
    return Experiment(npoints, funs, condY, quad_nodes, fused)
end

function Experiment2(npoints;
    funs=[makezSQwd(0.4), z -> z * makezSQwd(0.4)(z)],#[z->1.0, identity, z->exp(0.4*z^2), z->z*exp(0.4*z^2)],
    condY=z -> logistic(z - 2))
    quad_nodes = [AgnosticAGK(n) for n = npoints]
    function maker(f)
        function inner(z)
            print("  @", z)
            a = f(z)
            print(" f(z)= ", a)
            b = condY(z)
            print(", Y|z= ", b)
            r = a * b
            println(" -> ", r)
            return r
        end
    end
    fused = [maker(f) for f = funs]
    return Experiment(npoints, funs, condY, quad_nodes, fused)
end
struct ExperimentResult
    "specification of Experiment"
    experiment::Experiment
    "rows are experiment.funs and columns experiment.npoints
    individual cells are computed quadrature values"
    result
end

function compute(experiment::Experiment)
    nfun = length(experiment.funs)
    nquad = length(experiment.npoints)
    # following seems to be only allowed call with type as first argument
    res = NamedArray(Real, nfun + 1, nquad)
    #setnames!(res, ["1", "z", "w", "wz", "zhat"], 1)
    setnames!(res, ["w", "wz", "zhat"], 1)
    setnames!(res, string.(experiment.npoints), 2)
    setdimnames!(res, "f", 1)
    setdimnames!(res, "npoints", 2)
    for i1 = 1:nquad
        println("order ", experiment.npoints[i1])
        for i0 = 1:nfun
            println("  Function # ", i0)
            res[i0, i1] = experiment.quad_nodes[i1](experiment.fused_funs[i0])
        end
        res[nfun+1, i1] = res["wz", i1] / res["w", i1]
    end
    return ExperimentResult(experiment, res)
end

function test()
    expt = Experiment2([1, 2, 3]) # [1, 3, 5, 7, 8, 9, 10, 15, 20])
    r = compute(expt)
    display(r.result)
    return r
end
