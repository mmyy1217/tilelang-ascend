# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

import torch
import tilelang
import tilelang.language as T


N = 16
DTYPE = "float32"


@tilelang.jit(target="npuir")
def vadd():
    @T.prim_func
    def main(
        A: T.Tensor((N), DTYPE),
        B: T.Tensor((N), DTYPE),
        C: T.Tensor((N), DTYPE),
    ):
        with T.Kernel(1, is_npu=True) as (pid, sid):
            A_vec = T.alloc_ub((N), DTYPE)
            B_vec = T.alloc_ub((N), DTYPE)
            C_vec = T.alloc_ub((N), DTYPE)

            T.copy(A[0:N], A_vec[0:N])
            T.copy(B[0:N], B_vec[0:N])
            T.vadd(A_vec, B_vec, C_vec)
            T.copy(C_vec[0:N], C[0:N])

    return main


def main():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    torch.manual_seed(0)

    kernel = vadd()

    a = torch.randn((N,), dtype=torch.float32).npu()
    b = torch.randn((N,), dtype=torch.float32).npu()
    c = torch.empty((N,), dtype=torch.float32).npu()

    kernel(a, b, c)

    ref_c = a + b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
