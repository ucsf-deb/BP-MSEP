module MSEP
using FastGaussQuadrature
using LinearAlgebra

"""
Use Gauss-Hermite quadrature to evaluate a function.
Construct it by giving the number of points to evaluate and,
optionally, the mean and standard deviation of the distribution.

Thereafter you can treat the object as a function and call it on 
a function f that takes a single argument x.

The result is the f integrated over the normal distribution.
"""
struct NormalGH
    npoints
    μ
    σ
    w
    x
end

"constructor"
function NormalGH(npoints; μ=0.0, σ=1.0)
    x, w = gausshermite(npoints)
    return NormalGH(npoints, μ, σ, w ./ √π, x)
end

"""evaluate the integral, treating `NormalGH` instance as a function
Integrate with respect to normal density.
"""
function (p::NormalGH)(f)
    return dot(p.w, f.(p.μ.+√2*p.σ*p.x))  
end


end
