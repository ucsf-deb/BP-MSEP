module MSEP
using Distributions
using FastGaussQuadrature
using LinearAlgebra
using StatsFuns

# NormalGH is a struct and function name.  Not sure what the next line does.
export NormalGH
# and then there's the function associate with NormalGH instances

export condzy1

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
    return NormalGH(npoints, μ, σ, w ./ sqrtπ, x)
end

"""evaluate the integral, treating `NormalGH` instance as a function
Integrate with respect to normal density.
"""
function (p::NormalGH)(f)
    return dot(p.w, f.(p.μ.+sqrt2*p.σ*p.x))  
end

"""
conditional likelihood z|Y=1
for a single outcome with constant term k
The variance of the random effects is 1.
"""
function condzy1(z, k)
    return logistic(k+z)*pdf(Normal(), z)
end

end
