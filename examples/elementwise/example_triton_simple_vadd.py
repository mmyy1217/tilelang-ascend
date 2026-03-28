# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import triton
import triton.language as tl


N = 16
DTYPE = torch.float32


@triton.jit
def vadd_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)


def vadd(a, b, c):
    vadd_kernel[(1,)](
        a,
        b,
        c,
        BLOCK_SIZE=N,
    )


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)

    a = torch.randn((N,), dtype=DTYPE, device="npu")
    b = torch.randn((N,), dtype=DTYPE, device="npu")
    c = torch.empty((N,), dtype=DTYPE, device="npu")

    vadd(a, b, c)

    ref_c = a + b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
