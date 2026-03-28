# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

import torch
import tilelang
import tilelang.language as T


M = 16
N = 16
K = 16
DTYPE = "float32"
ACCUM_DTYPE = "float32"


@tilelang.jit(target="npuir")
def matmul():
    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((K, N), DTYPE),
        C: T.Tensor((M, N), DTYPE),
    ):
        with T.Kernel(1, is_npu=True) as (pid, sid):
            A_shared = T.alloc_shared((M, K), DTYPE)
            B_shared = T.alloc_shared((K, N), DTYPE)
            C_local = T.alloc_fragment((M, N), ACCUM_DTYPE)

            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, initC=True)
            T.copy(C_local, C[0, 0])

    return main


def main():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)

    torch.manual_seed(0)
    kernel = matmul()

    a = torch.randn((M, K), dtype=torch.float32).npu()
    b = torch.randn((K, N), dtype=torch.float32).npu()
    c = torch.empty((M, N), dtype=torch.float32).npu()

    kernel(a, b, c)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
