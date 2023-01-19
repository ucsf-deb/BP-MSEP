
"""
Evaluators compute a predictor for a single cluster.  Generally the value is
only good up to a constant of proportionality.  Some evaluators can be asked
to compute more than one kind of value, e.g., w*z or w alone.

Typically an Evaluator will include
   - a definition of the weight function
   - the underlying density of ``z``, usually std normal
   - the conditional density of the outcomes given the data
   - a method of numerical integration, including any parameters to set its 
   behavior

Expected protocol:
zhat(::Evaluator, ::WorkArea)
return an estimate of zhat for the cluster identified in the workarea

And the following functions, provided here with a default Implementation
that assumes certain instance variables are available.
"""
abstract type Evaluator end


"return brief name of primary predictor being evaluated"
# Maybe name(ev::T)  where {T <: Evaluator}::String ?
function name(ev::Evaluator)::String
    ev.targetName
end

"return fuller description of evaluator. full=true gives more detail"
function description(ev::Evaluator, full=false)::String
    des = "$(ev.targetName)(λ=$(ev.λ), k=$(ev.k), σ=$(ev.σ))"
    if full
        return des * ". $(ev.integratorDescription), order $(ev.integration_order)."
    else
        return des * " $(ev.integratorName)($(ev.integration_order))"
    end
end

"return targetName with suffix"
function name_with_suffix(suf::String, ev::Evaluator)::String
    return ev.targetName * suf
end

"""
Evaluators for binary outcomes.
"""
abstract type LogisticEvaluator <: Evaluator end

"""
Evaluators for Cutoff (CT) weighting schemes,
which are w=0 if |z|≤λ, else w=1.

This is unlike many others because
    1. The weight is not a term evaluated inside the exponential.
    2. The discontinuities do not work well with our typical numerical tools.

Logically Cutoff and Logistic are Orthogonal, but since julia is single inheritance,
for now we focus on the case we need.
"""
abstract type CutoffEvaluator <: LogisticEvaluator end
