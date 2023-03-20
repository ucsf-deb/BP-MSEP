#test5: breakpoints on the far side of compiled code

function test5(x)
    a=10  #set breakpoint here
    return a+x
end

#print(sum(test5(x) for x in [3, 5])) no stop
#print(sum((test5(2), test5(10)))) stops
#print(sum((test5(x) for x in [3, 5]))) no stop
#print(sum([test5(x) for x in [4, 5]])) stops