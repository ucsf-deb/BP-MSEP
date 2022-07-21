using MSEP
using Test
using FastGaussQuadrature

@testset "MSEP.jl" begin
    ngh = MSEP.NormalGH(5)
    f1(x)=1.0
    @test ngh(f1) ≈ (1.0)
    f2(x)=x
    @test ngh(f2) ≈ (0.0) atol=1e-14
    f3(x)=x^2
    @test ngh(f3) ≈ (1.0) 
end
