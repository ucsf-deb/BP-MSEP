"""
Evaluator for conventional Best Predictor, or at least the clostest
obvious analogue for logistic.

The same estimate is available from zsimp for any LogisticSimpleEvaluator;
this class and c'tor provides one with a good description and
access to zBP via zhat for consistency with other evaluators.

**zsimp() is not defined for LogisticBPEvaluator.**

The generic worker() should work OK for this type.

This struct relies on the inner evaluator being setup correctly,
but does not enforce that.  Use the corresponding constructor,
immediately after the struct definition.
"""
struct LogisticBPEvaluator{TParam,TObjFn,TIntegrator} <: Evaluator
    inner::LogisticSimpleEvaluator{TParam,TObjFn,TIntegrator}
end

""" make ev.σ and ev.k work for LogisticBPEvaluator
zSQdensity() needs this.

On this implementation: 
https://discourse.julialang.org/t/is-it-possble-for-julia-to-overwrite-how-to-get-field/59507

Note that @forward from `Lazy` works on functions, not accessors:
https://github.com/MikeInnes/Lazy.jl/blob/6705a0aa95b2d479e4ca5b5593313b0d6a857966/src/macros.jl#L293
"""
function Base.getproperty(ev::LogisticBPEvaluator, v::Symbol)
    if v ∈ (:σ, :k)
        getfield(ev.inner, v)
    else
        getfield(ev, v)
    end
end

"""
Return Best Predictor evaluator (which does no weighting)
Note the absence of λ from the arguments.
"""
function LogisticBPEvaluator(k, σ, integration_order=7)
    LogisticBPEvaluator(LogisticSimpleEvaluator(0.0, k, σ, integration_order, zSQdensity, "zBP", 
    AgnosticAGK(integration_order), "AGK", "Adaptive Gauss-Kronrod"))
end


"""return fuller description of evaluator. full=true gives more detail
Must override default behavior since λ is not relevant.
"""
function description(ev::TBPEvaluator, full=false)::String where {TBPEvaluator <: LogisticBPEvaluator}
    rev = ev.inner #real evaluator
    des = "$(rev.targetName)(k=$(rev.k), σ=$(rev.σ))"
    if full
        return des * ". $(rev.integratorDescription), order $(rev.integration_order)."
    else
        return des * " $(rev.integratorName)($(rev.integration_order))"
    end
end

"return brief name of primary predictor being evaluated"
function name(ev::TBPEvaluator)::String where {TBPEvaluator <: LogisticBPEvaluator}
    name(ev.inner)
end

"return targetName with suffix"
function name_with_suffix(suf::String, ev::TBPEvaluator)::String where {TBPEvaluator <: LogisticBPEvaluator}
    return name_with_suffix(ev.inner, suf)
end

"zhat for BP is what we ordinarily get from zsimp"
function zhat(ev::LogisticBPEvaluator, wa::WorkArea) where {WorkArea}
    zsimp(ev.inner, wa)
end

function work(ev::LogisticBPEvaluator)
    return work(ev.inner)
end
