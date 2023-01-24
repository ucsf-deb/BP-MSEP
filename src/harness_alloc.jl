# test harness for tracking memory allocations
using MSEP
using Profile
errs, errsBP = bigbigsim(1; nclusters=500);
Profile.clear_malloc_data()
errs, errsBP = bigbigsim(10; nclusters=500);

