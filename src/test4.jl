#illustrate problems getting average duration
using Dates
using Statistics
t1 = now()
sleep(2)
t2 = now()
sleep(1)
t3 = now()
d1 = t2-t1
d2 = t3-t2
durations = [d1, d2]
println(durations)

function meanDuration(ds)
    Millisecond(round(mean([d.value for d in ds])))
end
println("Manual method ", meanDuration(durations))

println(mean(durations))
