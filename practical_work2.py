from numba import cuda, numba

gpus = cuda.gpus.lst
num_cores = numba.config.NUMBA_DEFAULT_NUM_THREADS
for gpu in gpus:
    with gpu:
        meminfo = cuda.current_context().get_memory_info()
        print("Device name: %s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
        print('Number of cores: ' + str(num_cores))