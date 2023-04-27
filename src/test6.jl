using Random

Random.seed!(739587835)

function worker()
    sleep(0.2)
end

# launch workers
tasks = [Threads.@spawn worker() for i in 1:Threads.nthreads()]
println(Random.randn(10))
