# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import torch
import triton
import triton.language as tl


M = 128
N = 128
K = 128
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
DTYPE = torch.float16


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def matmul(a, b, c):
    matmul_kernel[(1,)](
        a,
        b,
        c,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)

    a = torch.randn((M, K), dtype=DTYPE, device="npu")
    b = torch.randn((K, N), dtype=DTYPE, device="npu")
    c = torch.empty((M, N), dtype=DTYPE, device="npu")

    matmul(a, b, c)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
