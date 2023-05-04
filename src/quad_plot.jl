#=

Graph components of likelihood to understand screwy quadrature results.
=#

using Gadfly
using StatsFuns  # for logistic

"""
Model for likelihood and its components.
    Cribbed from the evaluator and zSQDensity.
    Unlike most of the code, this evaluates a complete set of outcomes.
    Like the rest of the code, there are no binomial terms, i.e., this is the
    likelihood of a given ordering of outcomes among individuals that have ∑Y.
"""
struct Like
    μ::Float64
    σ::Float64
    n::Int # cluster size
    λ::Float64  # assumed for zSQ
end

"weight at a given z"
function w(like::Like, z::Float64)
    exp(like.λ*z^2)
end

"normal densityish at z.  z is already normalized"
function ϕ(like::Like, z::Float64)
    exp(-z^2/2)
end

"densityish of Y, total number of successes, given z"
function Ydens(like::Like, z::Float64, Y::Int)
    η = like.μ + like.σ * z
    p = logistic(η)  # probability of individual success
    p^Y*(1-p)^(like.n-Y)
end

"term to be integrated for w"
function wdens(like::Like, z::Float64, Y::Int)
    # fuse normal and w density to avoid overflow
    exp((2*like.λ-1)*z^2/2)*Ydens(like, z, Y)
end

"term to be integrated for wz"
function wzdens(like::Like, z::Float64, Y::Int)
    z*wdens(like, z, Y)
end

function Base.show(io::IO, m::MIME"juliavscode/html", p::Gadfly.Plot)
    show(io, "text/html", p)
end

function look(Y=17)
    like = Like(-2.0, 0.5, 20, 0.4)
    w(z) = wdens(like, z, Y)
    wz(z) = wzdens(like, z, Y)
    nrm(z) = ϕ(like, z)  # much higher values
    p = plot([w, wz], -0.0, 12.0)
    img = SVG("quad_plot.svg", 30cm, 20cm)
    draw(img, p)
end

look()