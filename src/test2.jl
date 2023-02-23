#=
Experiments with type system.
=#
struct Foo
    a::Int
    b::Int
end

struct Bar
    a
    b
end

function silly(f)
    f.a+f.b
end
@code_warntype silly(Foo(3, 4))

@code_warntype silly(Bar(1, 2))
function silly2(f::Foo)
    f.a+f.b
end

@code_warntype silly2(Foo(2, 9))

function silly3(f::T) where {T}
    f.a+f.b
end

@code_warntype silly3(Foo(2, 2))

s3=silly3

@code_warntype s3(Foo(1, 2))