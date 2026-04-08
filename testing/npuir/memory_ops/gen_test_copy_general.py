# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

DTYPES = ["float16", "float32"]

DIRECTIONS = [
    "GM_UB_GM",
    "UB_UB",
]

# fmt: off
SHAPE_RANK_CASES = [
    # 1D-5D (Dense Same Rank)
    ((1024, ), (1024, ), "0:1024", "0:1024", "dense_1d"),
    ((32, 64), (32, 64), "0:32, 0:64", "0:32, 0:64", "dense_2d"),
    ((8, 16, 32), (8, 16, 32), "0:8, 0:16, 0:32", "0:8, 0:16, 0:32", "dense_3d"),
    ((2, 4, 16, 32), (2, 4, 16, 32), "0:2, 0:4, 0:16, 0:32", "0:2, 0:4, 0:16, 0:32", "dense_4d"),
    ((2, 2, 4, 16, 32), (2, 2, 4, 16, 32), "0:2, 0:2, 0:4, 0:16, 0:32", "0:2, 0:2, 0:4, 0:16, 0:32", "dense_5d"),

    # Singletons (Dense Rank-Reduced)
    ((1, 32, 64), (1, 32, 64), "0, 0:32, 0:64", "0, 0:32, 0:64", "dense_rank_reduced_3d_highest_1"),
    ((4, 1, 64), (4, 1, 64), "0:4, 0, 0:64", "0:4, 0, 0:64", "dense_rank_reduced_3d_middle_1"),
    ((4, 32, 1), (4, 32, 1), "0:4, 0:32, 0", "0:4, 0:32, 0", "dense_rank_reduced_3d_lowest_1"),
    ((1, 4, 1, 32, 1), (1, 4, 1, 32, 1), "0, 0:4, 0, 0:32, 0", "0, 0:4, 0, 0:32, 0", "dense_rank_reduced_5d_multi_1"),

    # True Strided
    ((32, 128), (16, 64), "16:32, 32:96", "0:16, 0:64", "true_strided_2d"),
    ((4, 16, 64), (2, 8, 32), "1:3, 4:12, 16:48", "0:2, 0:8, 0:32", "true_strided_3d"),

    # Squeeze
    ((16, 128), (128, ), "5, 0:128", "0:128", "dense_squeeze_2d_to_1d"),
    ((4, 16, 64), (16, 64), "2, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_3d_to_2d"),
    ((2, 4, 16, 64), (16, 64), "1, 2, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_4d_to_2d"),
    ((2, 2, 4, 16, 64), (16, 64), "1, 0, 3, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_5d_to_2d"),
    ((2, 1, 64), (1, 64), "1, 0:1, 0:64", "0:1, 0:64", "dense_squeeze_3d_mid_1_to_2d"),
    ((4, 32, 128), (16, 64), "2, 8:24, 32:96", "0:16, 0:64", "true_strided_squeeze_3d_to_2d"),

    # Expand
    ((128, ), (16, 128), "0:128", "5, 0:128", "dense_expand_1d_to_2d_row"),
    ((16, 64), (4, 16, 64), "0:16, 0:64", "2, 0:16, 0:64", "dense_expand_2d_to_3d_plane"),
    ((16, 64), (2, 4, 16, 64), "0:16, 0:64", "1, 2, 0:16, 0:64", "dense_expand_2d_to_4d"),
    ((1, ), (1, 128), "0:1", "0, 0:1", "dense_expand_1d_1_to_2d"),
    ((1, 64), (2, 4, 1, 64), "0:1, 0:64", "1, 2, 0:1, 0:64", "dense_expand_2d_1_to_4d"),
    ((16, 32), (4, 64, 128), "0:16, 0:32", "2, 16:32, 32:64", "true_strided_expand_2d_to_3d_inner"),
]
# fmt: on

DYNAMIC_SCENARIOS = [
    "dynamic_from_func_args",
    "dynamic_from_kernel_cid",
    "dynamic_from_for_loop_iterator",
    "dynamic_from_min_max_arithmetic",
    "dynamic_loaded_from_ub_buffer",
]

SYNTAX_STYLES = [
    "full_tensor",
    "implicit_src_scalar",
    "implicit_dst_scalar",
]


def simplify_slice(slice_str, shape_tuple):
    parts = [p.strip() for p in slice_str.split(",")]
    if len(parts) != len(shape_tuple):
        return slice_str

    simplified_parts = []
    for part, dim_len in zip(parts, shape_tuple):
        if ":" in part:
            start, end = part.split(":")
            start = start.strip()
            end = end.strip()

            if start == "0" and end == str(dim_len):
                simplified_parts.append(":")
            elif start == "0":
                simplified_parts.append(f":{end}")
            elif end == str(dim_len):
                simplified_parts.append(f"{start}:")
            else:
                simplified_parts.append(part)
        else:
            simplified_parts.append(part)

    if all(p == ":" for p in simplified_parts):
        return ""

    return ", ".join(simplified_parts)


def format_access(tensor_name, slice_str):
    if not slice_str:
        return tensor_name
    return f"{tensor_name}[{slice_str}]"


def indent(lines, spaces=4):
    prefix = " " * spaces
    return [prefix + line if line else line for line in lines]


def get_human_description(name):
    mappings = {
        "dense_same_rank": "Tests basic dense contiguous memory copy without slicing.",
        "dense_rank_reduced_3d_highest_1": "Tests dense static copy with a rank-reduced slice on the highest dimension.",
        "dense_rank_reduced_3d_middle_1": "Tests dense static copy with a rank-reduced slice on the middle dimension.",
        "dense_rank_reduced_3d_lowest_1": "Tests dense static copy with a rank-reduced slice on the lowest dimension.",
        "dense_rank_reduced_5d_multi_1": "Tests dense static copy with multiple discontinuous rank-reduced dimensions.",
        "true_strided_2d": "Tests true non-contiguous 2D strided block memory copy.",
        "true_strided_3d": "Tests true non-contiguous 3D strided block memory copy.",
        "dense_squeeze_2d_to_1d": "Tests dimensionality reduction from 2D dense to 1D flat buffer.",
        "dense_squeeze_3d_to_2d": "Tests contiguous block fetch causing squeeze from 3D to 2D.",
        "dense_squeeze_4d_to_2d": "Tests contiguous block fetch squeezing 4D into 2D.",
        "dense_squeeze_5d_to_2d": "Tests contiguous block fetch squeezing 5D into 2D.",
        "dense_squeeze_3d_mid_1_to_2d": "Tests squeeze behavior when extracting a block from a middle singleton dimension.",
        "true_strided_squeeze_3d_to_2d": "Tests true non-contiguous strided copy across rank mismatch (3D squeezed to 2D).",
        "dense_expand_1d_to_2d_row": "Tests broadcasting/expanding a 1D vector into a 2D matrix row.",
        "dense_expand_2d_to_3d_plane": "Tests embedding a 2D matrix into a 3D volume.",
        "dense_expand_2d_to_4d": "Tests embedding a 2D matrix into a deeply nested 4D tensor.",
        "dense_expand_1d_1_to_2d": "Tests expanding a scalar/singleton 1D to a 2D matrix row.",
        "dense_expand_2d_1_to_4d": "Tests expanding a 2D matrix into a deeply nested 4D tensor starting from singleton.",
        "true_strided_expand_2d_to_3d_inner": "Tests expanding a 2D matrix into a 3D inner volume with true non-contiguous strided indexing.",
        "dynamic_from_func_args": "Tests dynamic memory lengths passed directly as runtime arguments.",
        "dynamic_from_kernel_cid": "Tests dynamic memory indexing natively dependent on NPU block CID (Core ID).",
        "dynamic_from_for_loop_iterator": "Tests dynamic row indexing derived from internal for-loop induction variables.",
        "dynamic_from_min_max_arithmetic": "Tests arithmetic boundaries (min/max) for dynamic tail-block processing.",
        "dynamic_loaded_from_ub_buffer": "Tests scalar indices loaded dynamically from UB for data-dependent gather memory fetching.",
        "full_tensor": "Tests baseline whole-tensor copy syntax.",
        "implicit_src_scalar": "Tests TileLang syntax sugar where source tensors have implicit trailing scalar bounds.",
        "implicit_dst_scalar": "Tests TileLang syntax sugar where target tensors have implicit trailing scalar bounds.",
    }
    return mappings.get(name, f"Tests {name.replace('_', ' ')} functionality.")


def gen_shape_rank_kernel_str(
    src_shape, dst_shape, src_slice, dst_slice, direction, func_name
):
    src_slice_simp = simplify_slice(src_slice, src_shape)
    dst_slice_simp = simplify_slice(dst_slice, dst_shape)

    base_name = (
        func_name.replace("kernel_", "").replace("_GM_UB_GM", "").replace("_UB_UB", "")
    )
    desc = get_human_description(base_name)
    s_slice_str = src_slice_simp if src_slice_simp else ":"
    d_slice_str = dst_slice_simp if dst_slice_simp else ":"
    description_str = f'        """\n        {desc}\n        Example Instantiation: A[{s_slice_str}] (Shape: {src_shape}) -> B[{d_slice_str}] (Shape: {dst_shape})\n        """'

    if direction == "GM_UB_GM":
        lines = [
            f"def get_{func_name}(src_shape, dst_shape, dtype):",
            "    @T.prim_func",
            f"    def {func_name}(",
            "        A: T.Tensor(src_shape, dtype), B: T.Tensor(src_shape, dtype)",
            "    ):",
            description_str,
            "        with T.Kernel(1, is_npu=True):",
            "            A_UB = T.alloc_ub(dst_shape, dtype)",
            f"            T.copy({format_access('A', src_slice_simp)}, {format_access('A_UB', dst_slice_simp)})",
            f"            T.copy({format_access('A_UB', dst_slice_simp)}, {format_access('B', src_slice_simp)})",
            "    ",
            "    def ref_func(inp, out):",
            "        expected_out = torch.zeros_like(out)",
            f"        expected_out{format_access('', src_slice_simp) if src_slice_simp else '[:]'} = inp{format_access('', src_slice_simp) if src_slice_simp else '[:]'} ",
            "        return expected_out",
            "",
            f"    return {func_name}, ref_func",
        ]
    elif direction == "UB_UB":
        lines = [
            f"def get_{func_name}(src_shape, dst_shape, dtype):",
            "    @T.prim_func",
            f"    def {func_name}(",
            "        A: T.Tensor(src_shape, dtype), B: T.Tensor(dst_shape, dtype)",
            "    ):",
            description_str,
            "        with T.Kernel(1, is_npu=True):",
            "            A_UB1 = T.alloc_ub(src_shape, dtype)",
            "            A_UB2 = T.alloc_ub(dst_shape, dtype)",
            "            T.copy(A, A_UB1)",
            f"            T.copy({format_access('A_UB1', src_slice_simp)}, {format_access('A_UB2', dst_slice_simp)})",
            "            T.copy(A_UB2, B)",
            "    ",
            "    def ref_func(inp, out):",
            "        expected_out = torch.zeros_like(out)",
            f"        expected_out{format_access('', dst_slice_simp) if dst_slice_simp else '[:]'} = inp{format_access('', src_slice_simp) if src_slice_simp else '[:]'}",
            "        return expected_out",
            "",
            f"    return {func_name}, ref_func",
        ]

    return "\n".join(lines)


def gen_dynamic_kernel_str(dyn_case, direction, func_name, src_shape, ub_shape):
    lines = []
    kernel_threads = 1
    core_logic = []

    if dyn_case == "dynamic_from_func_args":
        args = (
            "A: T.Tensor((1024,), dtype), B: T.Tensor((1024,), dtype), shape_N: T.int32"
        )
        core_logic = ["length = shape_N", "offset = 0"]
    elif dyn_case == "dynamic_from_kernel_cid":
        args = "A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)"
        kernel_threads = 4
        core_logic = ["row = cid", "offset = 0", "length = 256"]
    elif dyn_case == "dynamic_from_for_loop_iterator":
        args = "A: T.Tensor((8, 128), dtype), B: T.Tensor((8, 128), dtype)"
        kernel_threads = 2
        core_logic = [
            "for i in T.serial(4):",
            "    row = cid * 4 + i",
            "    offset = 0",
            "    length = 128",
        ]

    elif dyn_case == "dynamic_from_min_max_arithmetic":
        kernel_threads = 4
        core_logic = [
            "block_size = 256",
            "offset = cid * 256",
            "remain = 1000 - offset",
            "length = T.min(block_size, remain)",
        ]
    elif dyn_case == "dynamic_loaded_from_ub_buffer":
        kernel_threads = 4
        core_logic = [
            "I_UB = T.alloc_ub((4,), 'int32')",
            "T.copy(indices, I_UB)",
            "row = I_UB[cid]",
            "offset = 0",
            "length = 256",
        ]

    arg_str_a = "A: T.Tensor(src_shape, dtype)"
    arg_str_b = "B: T.Tensor(dst_shape, dtype)"
    if dyn_case == "dynamic_loaded_from_ub_buffer":
        args = f"{arg_str_a}, indices: T.Tensor((4,), 'int32'), {arg_str_b}"
    elif dyn_case == "dynamic_from_func_args":
        args = f"{arg_str_a}, {arg_str_b}, length: T.int32"
    else:
        args = f"{arg_str_a}, {arg_str_b}"

    desc = get_human_description(dyn_case)
    description_str = f'        """\n        {desc}\n        Example Instantiation: src_shape={src_shape}, ub_shape={ub_shape}\n        """'

    lines = [
        f"def get_{func_name}(src_shape, dst_shape, ub_shape, dtype):",
        "    @T.prim_func",
        f"    def {func_name}({args}):",
        description_str,
        f"        with T.Kernel({kernel_threads}, is_npu=True) as (cid, _):",
    ]

    is_2d = dyn_case in (
        "dynamic_from_kernel_cid",
        "dynamic_from_for_loop_iterator",
        "dynamic_loaded_from_ub_buffer",
    )
    indention = (
        "                "
        if dyn_case == "dynamic_from_for_loop_iterator"
        else "            "
    )
    src_slice = "row, offset:offset+length" if is_2d else "offset:offset+length"
    ub_slice = "0:length"
    b_target = f"B[{src_slice}]" if is_2d else "B"

    for line in core_logic:
        lines.append("            " + line)

    if direction == "GM_UB_GM":
        lines.extend(
            [
                f"{indention}A_UB = T.alloc_ub(ub_shape, dtype)",
                f"{indention}T.copy(A[{src_slice}], A_UB[{ub_slice}])",
                f"{indention}T.copy(A_UB[{ub_slice}], {b_target})",
            ]
        )
        ref_lines = [
            "    def ref_func(inp, out, indices=None):",
            "        expected_out = torch.zeros_like(out)",
        ]
        if dyn_case == "dynamic_from_kernel_cid":
            ref_lines.extend(
                [
                    "        for cid in range(4):",
                    "            expected_out[cid, 0:256] = inp[cid, 0:256]",
                ]
            )
        elif dyn_case == "dynamic_from_min_max_arithmetic":
            ref_lines.extend(
                [
                    "        for cid in range(4):",
                    "            offset = cid * 256",
                    "            remain = 1000 - offset",
                    "            length = min(256, remain)",
                    "            expected_out[offset:offset+length] = inp[offset:offset+length]",
                ]
            )
        elif dyn_case == "dynamic_from_func_args":
            ref_lines.extend(["        expected_out[0:1000] = inp[0:1000]"])
        elif dyn_case == "dynamic_from_for_loop_iterator":
            ref_lines.extend(
                [
                    "        for cid in range(2):",
                    "            for i in range(4):",
                    "                row = cid * 4 + i",
                    "                expected_out[row, 0:128] = inp[row, 0:128]",
                ]
            )
        elif dyn_case == "dynamic_loaded_from_ub_buffer":
            ref_lines.extend(
                [
                    "        for i in range(4):",
                    "            expected_out[indices[i].item(), :] = inp[indices[i].item(), :]",
                ]
            )
        ref_lines.append("        return expected_out")
    else:
        a_target = "A[row, :]" if is_2d else "A"
        lines.extend(
            [
                f"{indention}A_UB1 = T.alloc_ub(ub_shape, dtype)",
                f"{indention}A_UB2 = T.alloc_ub(ub_shape, dtype)",
                f"{indention}T.copy({a_target}, A_UB1)",
                f"{indention}T.copy(A_UB1[{ub_slice}], A_UB2[{ub_slice}])",
                f"{indention}T.copy(A_UB2, {b_target})",
            ]
        )
        ref_lines = [
            "    def ref_func(inp, out, indices=None):",
            "        expected_out = torch.zeros_like(out)",
            "        expected_out[...] = inp[...]",
            "        return expected_out",
        ]
    lines.extend(ref_lines)
    lines.append(f"    return {func_name}, ref_func")
    return "\n".join(lines)


def gen_syntax_kernel_str(syntax, func_name, src_shape):
    desc = get_human_description(syntax)
    description_str = f'        """\n        {desc}\n        Example Instantiation: src_shape={src_shape}\n        """'
    if syntax == "full_tensor":
        return f"""def get_{func_name}(src_shape, dst_shape, dtype):
    @T.prim_func
    def {func_name}(
        A: T.Tensor(src_shape, dtype), B: T.Tensor(dst_shape, dtype)
    ):
{description_str}
        with T.Kernel(1, is_npu=True) as (cid, _):
            A_UB = T.alloc_ub(dst_shape, dtype)
            T.copy(A, A_UB)
            T.copy(A_UB, B)
    def ref_func(inp, out):
        expected_out = out.clone()
        expected_out[...] = inp[...]
        return expected_out
    return {func_name}, ref_func"""
    else:
        if syntax == "implicit_src_scalar":
            copies = "            T.copy(A[cid, 0:128], A_UB[0:128])\n            T.copy(A_UB[0:128], B[cid, 0:128])"
        else:
            copies = "            T.copy(A[cid, 0:128], A_UB)\n            T.copy(A_UB, B[cid, 0:128])"

        return f"""def get_{func_name}(src_shape, dst_shape, dtype):
    @T.prim_func
    def {func_name}(
        A: T.Tensor(src_shape, dtype), B: T.Tensor(dst_shape, dtype)
    ):
{description_str}
        with T.Kernel(8, is_npu=True) as (cid, _):
            A_UB = T.alloc_ub((128,), dtype)
{copies}
    def ref_func(inp, out):
        expected_out = out.clone()
        expected_out[...] = inp[...]
        return expected_out
    return {func_name}, ref_func"""


def main():
    out_file = os.path.join(os.path.dirname(__file__), "test_copy_general.py")

    with open(out_file, "w") as f:
        f.write("# Copyright (c) Huawei Technologies Co., Ltd. 2025.\n")
        f.write(
            "# This file is AUTO-GENERATED by gen_test_copy_general.py. DO NOT EDIT DIRECTLY.\n"
        )
        f.write("import pytest\n")
        f.write("import torch\n")
        f.write("import torch_npu  # noqa: F401\n")
        f.write("import tilelang\n")
        f.write("import tilelang.language as T\n")
        f.write("from testcommon import assert_close, gen_tensor\n\n")

        f.write("pytestmark = [\n")
        f.write("    pytest.mark.op('copy_general'),\n")
        f.write("    pytest.mark.mode('Expert'),\n")
        f.write("]\n\n")

        f.write("DTYPES = ['float16', 'float32']\n\n")

        # 1. Shape Rank
        shape_cases_list = []
        generated_func_names = set()
        for direction in DIRECTIONS:
            for (
                src_shape,
                dst_shape,
                src_slice,
                dst_slice,
                description,
            ) in SHAPE_RANK_CASES:
                src_slice_simp = simplify_slice(src_slice, src_shape)
                dst_slice_simp = simplify_slice(dst_slice, dst_shape)

                # Deduplicate identical kernel generation for entirely dense accesses without slices
                if src_slice_simp == "" and dst_slice_simp == "":
                    base_name = "dense_same_rank"
                else:
                    base_name = description

                func_name = f"kernel_{base_name}_{direction}"
                if func_name not in generated_func_names:
                    f.write(
                        gen_shape_rank_kernel_str(
                            src_shape,
                            dst_shape,
                            src_slice,
                            dst_slice,
                            direction,
                            func_name,
                        )
                        + "\n\n"
                    )
                    generated_func_names.add(func_name)
                shape_cases_list.append(
                    f'    ("{description}_{direction}", "{direction}", {src_shape}, {dst_shape}, get_{func_name}),\n'
                )

        f.write("# fmt: off\n")
        f.write("SHAPE_RANK_PARAMS = [\n")
        for c in shape_cases_list:
            f.write(c)
        f.write("]\n")
        f.write("# fmt: on\n\n")

        f.write("""@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("desc, direction, src_shape, dst_shape, get_kernel_func", SHAPE_RANK_PARAMS)
def test_shape_rank_mismatch(desc, direction, src_shape, dst_shape, get_kernel_func, dtype):
    kernel_func, ref_func = get_kernel_func(src_shape, dst_shape, dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    inp = gen_tensor(src_shape, dtype, kind='randn')
    out = gen_tensor(src_shape if direction == "GM_UB_GM" else dst_shape, dtype, kind='zeros')
    compiled(inp, out)
    expected_out = ref_func(inp, out)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)

""")

        # 2. Dynamic
        dynamic_cases_list = []
        for direction in DIRECTIONS:
            for dyn_case in DYNAMIC_SCENARIOS:
                func_name = f"kernel_{dyn_case}_{direction}"
                if dyn_case == "dynamic_from_kernel_cid":
                    src_shape, ub_shape = (4, 256), (256,)
                elif dyn_case == "dynamic_from_for_loop_iterator":
                    src_shape, ub_shape = (8, 128), (128,)
                elif dyn_case in ("dynamic_loaded_from_ub_buffer"):
                    src_shape, ub_shape = (4, 256), (256,)
                elif dyn_case == "dynamic_from_min_max_arithmetic":
                    src_shape, ub_shape = (1000,), (256,)
                else:
                    src_shape, ub_shape = (1024,), (1024,)

                f.write(
                    gen_dynamic_kernel_str(
                        dyn_case, direction, func_name, src_shape, ub_shape
                    )
                    + "\n\n"
                )

                dynamic_cases_list.append(
                    f'    ("{dyn_case}", "{direction}", {src_shape}, {src_shape}, {ub_shape}, get_{func_name}),\n'
                )

        f.write("# fmt: off\n")
        f.write("DYNAMIC_PARAMS = [\n")
        for c in dynamic_cases_list:
            f.write(c)
        f.write("]\n")
        f.write("# fmt: on\n\n")

        f.write("""@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dyn_case, direction, src_shape, dst_shape, ub_shape, get_kernel_func", DYNAMIC_PARAMS)
def test_dynamic_scenarios(dyn_case, direction, src_shape, dst_shape, ub_shape, get_kernel_func, dtype):
    kernel_func, ref_func = get_kernel_func(src_shape, dst_shape, ub_shape, dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    inp = gen_tensor(src_shape, dtype, kind='randn')
    out = gen_tensor(src_shape if direction == "GM_UB_GM" else dst_shape, dtype, kind='zeros')
    if dyn_case == "dynamic_from_func_args":
        compiled(inp, out, 1000)
        expected_out = ref_func(inp, out)
    elif dyn_case == "dynamic_loaded_from_ub_buffer":
        indices = torch.tensor([3, 2, 1, 0], dtype=torch.int32).npu()
        compiled(inp, indices, out)
        expected_out = ref_func(inp, out, indices=indices)
    else:
        compiled(inp, out)
        expected_out = ref_func(inp, out)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)

""")

        # 3. Syntax
        syntax_cases_list = []
        for syntax_case in SYNTAX_STYLES:
            func_name = f"kernel_syntax_{syntax_case}"
            src_shape = (128,) if syntax_case == "full_tensor" else (8, 128)
            f.write(gen_syntax_kernel_str(syntax_case, func_name, src_shape) + "\n\n")
            syntax_cases_list.append(
                f'    ("{syntax_case}", {src_shape}, {src_shape}, get_{func_name}),\n'
            )

        f.write("# fmt: off\n")
        f.write("SYNTAX_PARAMS = [\n")
        for c in syntax_cases_list:
            f.write(c)
        f.write("]\n")
        f.write("# fmt: on\n\n")

        f.write("""@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("syntax_case, src_shape, dst_shape, get_kernel_func", SYNTAX_PARAMS)
def test_syntax_scenarios(syntax_case, src_shape, dst_shape, get_kernel_func, dtype):
    kernel_func, ref_func = get_kernel_func(src_shape, dst_shape, dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    inp = gen_tensor(src_shape, dtype, kind='randn')
    out = gen_tensor(dst_shape, dtype, kind='zeros')
    compiled(inp, out)
    expected_out = ref_func(inp, out)
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
""")


if __name__ == "__main__":
    main()
