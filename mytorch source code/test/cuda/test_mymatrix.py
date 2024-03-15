import numpy as np
import sys
sys.path.append('../../')
import mytorch.cuda.mymatrix as mymatrix

def test_matmul():
    MATRIX_SIZE = 2048
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    C = A@B
    my_C = mymatrix.matmul(A, B)
    assert np.allclose(my_C, C, atol=1e-6)

def test_matmul_shared_memory():
    MATRIX_SIZE = 2048
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    C = A@B
    my_C = mymatrix.matmul_shared_memory(A, B)
    print(my_C)
    assert np.allclose(my_C, C, atol=1e-6)

def test_matmul_shared_memory_multistream():
    MATRIX_SIZE = 2048
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    B = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    C = A@B
    my_C = mymatrix.matmul_shared_memory_multistream(A, B)
    assert np.allclose(my_C, C, atol=1e-6)

