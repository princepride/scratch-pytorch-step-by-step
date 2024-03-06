from numba import cuda
import numpy as np
import math

THREAD_SIZE = 32

@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# 普通矩阵乘法
def matmul(A, B):
    
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))  # 结果矩阵C

    threadsperblock = (THREAD_SIZE, THREAD_SIZE)
    blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 启动kernel
    matmul_kernel[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

    # 将结果从GPU内存复制回主机内存
    return C_global_mem

@cuda.jit
def matmul_kernel_shared_memory(A, B, C):
    # 定义共享内存
    TILE_WIDTH = THREAD_SIZE
    shared_A = cuda.shared.array((TILE_WIDTH, TILE_WIDTH), dtype=np.float32)
    shared_B = cuda.shared.array((TILE_WIDTH, TILE_WIDTH), dtype=np.float32)

    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = 0.0
    for phase in range(math.ceil(A.shape[1] / TILE_WIDTH)):
        shared_A[tx, ty] = A[x, phase*TILE_WIDTH + ty]
        shared_B[tx, ty] = B[phase*TILE_WIDTH + tx, y]

        cuda.syncthreads()

        for k in range(TILE_WIDTH):
            tmp += shared_A[tx, k] * shared_B[k, ty]

        cuda.syncthreads()

    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp

def matmul_shared_memory(A, B):
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))  # 结果矩阵C

    threadsperblock = (THREAD_SIZE, THREAD_SIZE)
    blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 启动kernel
    matmul_kernel_shared_memory[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    return C_global_mem

def matmul_shared_memory_multistream(A, B):
    # 定义流的数量
    num_streams = 8
    streams = [cuda.stream() for _ in range(num_streams)]

    # 初始化结果矩阵C
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)

    # 计算每个流处理的行数
    rows_per_stream = math.ceil(A.shape[0] / num_streams)

    # 配置kernel
    threadsperblock = (THREAD_SIZE, THREAD_SIZE)
    
    for i in range(num_streams):
        # 计算每个流处理的矩阵A和C的部分
        start_row = i * rows_per_stream
        end_row = min((i + 1) * rows_per_stream, A.shape[0])
        A_part = A[start_row:end_row, :]
        
        C_part_global_mem = cuda.device_array((end_row - start_row, B.shape[1]), stream=streams[i])
        
        blockspergrid_x = int(np.ceil(A_part.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        A_global_mem = cuda.to_device(A_part, stream=streams[i])
        B_global_mem = cuda.to_device(B, stream=streams[i])
        
        matmul_kernel_shared_memory[blockspergrid, threadsperblock, streams[i]](A_global_mem, B_global_mem, C_part_global_mem)
        
        # 将结果从GPU内存复制回主机内存
        C_part = C_part_global_mem.copy_to_host(stream=streams[i])
        C[start_row:end_row, :] = C_part

    return C