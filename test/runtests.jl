using MSEP
using Test
using FastGaussQuadrature

@testset "MSEP.jl standard normal" begin
    ngh = MSEP.NormalGH(5)
    f1(x)=1.0
    @test ngh(f1) ≈ (1.0)
    f2(x)=x
    @test ngh(f2) ≈ (0.0) atol=1e-14
    f3(x)=x^2
    @test ngh(f3) ≈ (1.0) 
end

@testset "MSEP Normal(-1, 2)" begin
    ngh = MSEP.NormalGH(5, μ=-1, σ=2)
    f1(x)=1.0
    @test ngh(f1) ≈ (1.0)
    f2(x)=x
    @test ngh(f2) ≈ (-1.0) atol=1e-14
    f3(x)=(x+1.0)^2
    @test ngh(f3) ≈ (4.0) rtol=1e-14
end