"Return evaluator using |z| for the weight function"
function LogisticABEvaluator(λ, k, σ, integration_order=7)
    LogisticSimpleEvaluator(λ, k, σ, integration_order, wDensity((z, λ)-> λ*abs(z)), "zAB", 
    AgnosticAGK(integration_order), "AGK", "Adaptive Gauss-Kronrod")
end
