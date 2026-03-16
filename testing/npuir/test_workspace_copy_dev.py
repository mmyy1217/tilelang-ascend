# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import pytest
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


torch.npu.set_device(0)
tilelang.cache.clear_cache()
DTYPE = "float16"
TORCH_DTYPE = torch.float16
TEST_SHAPES = [
    (16, 16),
    (32, 64),
    (64, 32),
]


def workspace_copy_kernel(M, N):

    @T.prim_func
    def workspace_copy(
        A: T.Tensor((M, N), DTYPE),
        B: T.Tensor((M, N), DTYPE),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            UB_IN = T.alloc_fragment((M, N), DTYPE)
            WS = T.alloc_workspace((M, N), DTYPE)
            UB_OUT = T.alloc_fragment((M, N), DTYPE)

            # GM -> UB
            T.copy(A, UB_IN)
            # UB -> workspace
            T.copy(UB_IN, WS)
            # workspace -> GM (via UB bridge, because direct memref2memref is disabled)
            T.copy(WS, UB_OUT)
            T.copy(UB_OUT, B)

    return workspace_copy


def workspace_fixpipe_loop_kernel():

    @T.prim_func
    def workspace_fixpipe_loop(
        A: T.Tensor((32, 32), DTYPE),
        B: T.Tensor((32, 32), DTYPE),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_ub = T.alloc_shared((32, 32), DTYPE)
            B_ub = T.alloc_shared((32, 32), DTYPE)
            Acc = T.alloc_fragment((32, 32), "float32")
            WS = T.alloc_workspace((32, 32), "float32")
            Out_ub = T.alloc_fragment((32, 32), "float32")

            T.copy(A, A_ub)
            T.copy(B, B_ub)

            for _ in T.serial(2):
                T.gemm(A_ub, B_ub, Acc, initC=True)
                T.copy(Acc, WS)
                T.copy(WS, Out_ub)

            T.copy(Out_ub, C)

    return workspace_fixpipe_loop


@pytest.mark.parametrize("M, N", TEST_SHAPES)
def test_workspace_copy_dev(M, N):
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"

    func = workspace_copy_kernel(M, N)
    compiled_kernel = tilelang.compile(func, target="npuir")

    src = torch.randn(size=[M, N], dtype=TORCH_DTYPE).npu()
    dst = torch.zeros(size=[M, N], dtype=TORCH_DTYPE).npu()
    ref = src.clone()

    compiled_kernel(src, dst)
    torch.testing.assert_close(dst, ref, rtol=1e-2, atol=1e-2)


def test_workspace_fixpipe_loop_compile_dev():
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    tilelang.compile(workspace_fixpipe_loop_kernel(), target="npuir")
