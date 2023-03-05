"""
Evaluator for conventional Best Predictor, or at least the clostest
obvious analogue for logistic.

The same estimate is available from zsimp for any LogisticSimpleEvaluator;
this class and c'tor provides one with a good description and
access to zBP via zhat for consistency with other evaluators.

**Neither zsimp() nor worker() are defined for LogisticBPEvaluator.**

This struct relies on the inner evaluator being setup correctly,
but does not enforce that.  Use the corresponding constructor,
immediately after the struct definition.
"""
struct LogisticBPEvaluator{TParam,TObjFn,TIntegrator} <: Evaluator
    inner::LogisticSimpleEvaluator{TParam,TObjFn,TIntegrator}
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
Must override default behavior since λ is not present.
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
