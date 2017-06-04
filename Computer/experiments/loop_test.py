import time
previous_time = time.time()
while True:
    loop_time = time.time()
    print("single loop time:",loop_time-previous_time)
    prevous_time = loop_time
