# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os

DTYPES = ["float16", "float32"]

DIRECTIONS = [
    "GM_UB_GM",
    "UB_UB",
]

SHAPE_RANK_CASES = [
    # Same Rank Dense
    ((1024,), (1024,), "0:1024", "0:1024", "dense_1d"),
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
    ((16, 128), (128,), "5, 0:128", "0:128", "dense_squeeze_2d_to_1d"),
    ((4, 16, 64), (16, 64), "2, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_3d_to_2d"),
    ((2, 4, 16, 64), (16, 64), "1, 2, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_4d_to_2d"),
    ((2, 2, 4, 16, 64), (16, 64), "1, 0, 3, 0:16, 0:64", "0:16, 0:64", "dense_squeeze_5d_to_2d"),
    ((2, 1, 64), (1, 64), "1, 0:1, 0:64", "0:1, 0:64", "dense_squeeze_3d_mid_1_to_2d"),
    ((4, 32, 128), (16, 64), "2, 8:24, 32:96", "0:16, 0:64", "true_strided_squeeze_3d_to_2d"),

    # Expand
    ((128,), (16, 128), "0:128", "5, 0:128", "dense_expand_1d_to_2d_row"),
    ((16, 64), (4, 16, 64), "0:16, 0:64", "2, 0:16, 0:64", "dense_expand_2d_to_3d_plane"),
    ((16, 64), (2, 4, 16, 64), "0:16, 0:64", "1, 2, 0:16, 0:64", "dense_expand_2d_to_4d"),
    ((1,), (1, 128), "0:1", "0, 0:1", "dense_expand_1d_1_to_2d"),
    ((1, 64), (2, 4, 1, 64), "0:1, 0:64", "1, 2, 0:1, 0:64", "dense_expand_2d_1_to_4d"),
    ((16, 32), (4, 64, 128), "0:16, 0:32", "2, 16:32, 32:64", "true_strided_expand_2d_to_3d_inner"),
]

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
    parts = [p.strip() for p in slice_str.split(',')]
    if len(parts) != len(shape_tuple):
        return slice_str
    
    simplified_parts = []
    for part, dim_len in zip(parts, shape_tuple):
        if ':' in part:
            start, end = part.split(':')
            start = start.strip()
            end = end.strip()
            
            if start == "0" and end == str(dim_len):
                simplified_parts.append(':')
            elif start == "0":
                simplified_parts.append(f':{end}')
            elif end == str(dim_len):
                simplified_parts.append(f'{start}:')
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


def gen_shape_rank_kernel_str(src_shape, dst_shape, src_slice, dst_slice, direction, func_name):
    src_slice_simp = simplify_slice(src_slice, src_shape)
    dst_slice_simp = simplify_slice(dst_slice, dst_shape)
    
    if direction == "GM_UB_GM":
        lines = [
            f"def get_{func_name}(src_shape, dst_shape, dtype):",
            "    @T.prim_func",
            f"    def generated_kernel(A: T.Tensor(src_shape, dtype), B: T.Tensor(src_shape, dtype)):",
            "        with T.Kernel(1, is_npu=True):",
            f"            A_UB = T.alloc_ub(dst_shape, dtype)",
            f"            T.copy({format_access('A', src_slice_simp)}, {format_access('A_UB', dst_slice_simp)})",
            f"            T.copy({format_access('A_UB', dst_slice_simp)}, {format_access('B', src_slice_simp)})",
            "    return generated_kernel"
        ]
    elif direction == "UB_UB":
        lines = [
            f"def get_{func_name}(src_shape, dst_shape, dtype):",
            "    @T.prim_func",
            f"    def generated_kernel(A: T.Tensor(src_shape, dtype), B: T.Tensor(dst_shape, dtype)):",
            "        with T.Kernel(1, is_npu=True):",
            f"            A_UB1 = T.alloc_ub(src_shape, dtype)",
            f"            A_UB2 = T.alloc_ub(dst_shape, dtype)",
            "            T.copy(A, A_UB1)",
            f"            T.copy({format_access('A_UB1', src_slice_simp)}, {format_access('A_UB2', dst_slice_simp)})",
            "            T.copy(A_UB2, B)",
            "    return generated_kernel"
        ]
        
    return "\n".join(lines)


def gen_dynamic_kernel_str(dynamic_origin, direction, func_name):
    args = ""
    kernel_threads = 1
    core_logic = []

    if dynamic_origin == "dynamic_from_func_args":
        args = f"A: T.Tensor((1024,), dtype), B: T.Tensor((1024,), dtype), shape_N: T.int32"
        core_logic = [
            "length = shape_N",
            "offset = 0"
        ]
    elif dynamic_origin == "dynamic_from_kernel_cid":
        args = f"A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)"
        kernel_threads = 4
        core_logic = [
            "row = cid",
            "offset = 0",
            "length = 256"
        ]
    elif dynamic_origin == "dynamic_from_for_loop_iterator":
        args = f"A: T.Tensor((8, 128), dtype), B: T.Tensor((8, 128), dtype)"
        kernel_threads = 2
        core_logic = [
            "for i in T.serial(4):",
            "    row = cid * 4 + i",
            "    offset = 0",
            "    length = 128"
        ]
    elif dynamic_origin == "dynamic_from_min_max_arithmetic":
        args = f"A: T.Tensor((1000,), dtype), B: T.Tensor((1000,), dtype)"
        kernel_threads = 4
        core_logic = [
            "block_size = 256",
            "offset = cid * 256",
            "remain = 1000 - offset",
            "length = T.min(block_size, remain)"
        ]
    elif dynamic_origin == "dynamic_loaded_from_ub_buffer":
        args = f"A: T.Tensor((4, 256), dtype), Indices: T.Tensor((4,), 'int32'), B: T.Tensor((4, 256), dtype)"
        kernel_threads = 4
        core_logic = [
            "I_UB = T.alloc_ub((4,), 'int32')",
            "T.copy(Indices, I_UB)",
            "row = I_UB[cid]",
            "offset = 0",
            "length = 256"
        ]

    lines = [
        f"def get_{func_name}(dtype):",
        "    @T.prim_func",
        f"    def generated_kernel({args}):",
        f"        with T.Kernel({kernel_threads}, is_npu=True) as (cid, _):"
    ]
    for line in core_logic:
        lines.append("            " + line)

    is_2d = dynamic_origin in ["dynamic_from_kernel_cid", "dynamic_from_for_loop_iterator", "dynamic_loaded_from_ub_buffer"]
    is_for_loop = dynamic_origin == "dynamic_from_for_loop_iterator"
    indention = "                " if is_for_loop else "            "
    
    if is_2d:
        src_slice = "row, offset:offset+length"
        dst_slice = "row, offset:offset+length"
        ub_shape = "(128,)" if is_for_loop else "(256,)"
        ub_slice = "0:length"
        b_out_target = f"B[{src_slice}]"
        b_args_target = "B"
        b_target = "B[row, :]"
    else:
        src_slice = "offset:offset+length"
        dst_slice = "offset:offset+length"
        ub_shape = "(1024,)" if dynamic_origin == "dynamic_from_func_args" else "(256,)"
        ub_slice = "0:length"
        b_out_target = f"B[{src_slice}]"
        b_args_target = "B"
        b_target = "B"
    
    if direction == "GM_UB_GM":
        if is_2d:
            b_args_target = f"B: T.Tensor((4, 256), dtype)" if dynamic_origin == "dynamic_from_kernel_cid" else f"B: T.Tensor((8, 128), dtype)"
            if dynamic_origin == "dynamic_loaded_from_ub_buffer":
                b_args_target = f"B: T.Tensor((4, 256), dtype)"
            b_out_target = f"B[{src_slice}]"
        else:
            b_args_target = f"B: T.Tensor((1024,), dtype)" if dynamic_origin == "dynamic_from_func_args" else f"B: T.Tensor((1000,), dtype)"
            b_out_target = f"B[{src_slice}]"

    if direction == "GM_UB_GM":
        if is_2d:
            b_args_target = f"B: T.Tensor((4, 256), dtype)" if dynamic_origin == "dynamic_from_kernel_cid" else f"B: T.Tensor((8, 128), dtype)"
            if dynamic_origin == "dynamic_loaded_from_ub_buffer":
                b_args_target = f"B: T.Tensor((4, 256), dtype)"
        else:
            b_args_target = f"B: T.Tensor((1024,), dtype)" if dynamic_origin == "dynamic_from_func_args" else f"B: T.Tensor((1000,), dtype)"
        # we need custom replace because we previously hardcoded args
        # actually, the easiest way is to just replace B's arg string universally inside lines
        # But wait, lines[2] is exactly f"    def generated_kernel({args}):"
        
        lines.extend([
            f"{indention}A_UB = T.alloc_ub({ub_shape}, dtype)",
            f"{indention}T.copy(A[{src_slice}], A_UB[{ub_slice}])",
            f"{indention}T.copy(A_UB[{ub_slice}], {b_out_target})"
        ])
    elif direction == "UB_UB":
        a_target = "A[row, :]" if is_2d else "A"
        lines.extend([
            f"{indention}A_UB1 = T.alloc_ub({ub_shape}, dtype)",
            f"{indention}A_UB2 = T.alloc_ub({ub_shape}, dtype)",
            f"{indention}T.copy({a_target}, A_UB1)",
            f"{indention}T.copy(A_UB1[{ub_slice}], A_UB2[{ub_slice}])",
            f"{indention}T.copy(A_UB2, {b_target})"
        ])

    lines.append("    return generated_kernel")
    return "\n".join(lines)


def gen_syntax_kernel_str(syntax, func_name):
    lines = []
    lines = [f"def get_{func_name}(dtype):"]
    if syntax == "full_tensor":
        lines.extend([
            "    @T.prim_func",
            f"    def generated_kernel(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype)):",
            "        with T.Kernel(1, is_npu=True) as (cid, _):",
            f"            A_UB = T.alloc_ub((128,), dtype)",
            "            T.copy(A, A_UB)",
            "            T.copy(A_UB, B)"
        ])
    else:
        lines.extend([
            "    @T.prim_func",
            f"    def generated_kernel(A: T.Tensor((8, 128), dtype), B: T.Tensor((8, 128), dtype)):",
            "        with T.Kernel(8, is_npu=True) as (cid, _):",
            f"            A_UB = T.alloc_ub((128,), dtype)"
        ])
        if syntax == "implicit_src_scalar":
            lines.extend([
                "            T.copy(A[cid, 0:128], A_UB[0:128])",
                "            T.copy(A_UB[0:128], B[cid, 0:128])"
            ])
        elif syntax == "implicit_dst_scalar":
            lines.extend([
                "            T.copy(A[cid, 0:128], A_UB)",
                "            T.copy(A_UB, B[cid, 0:128])"
            ])
    lines.append("    return generated_kernel")
    return "\n".join(lines)


def main():
    out_file = os.path.join(os.path.dirname(__file__), "test_copy_general.py")
    
    with open(out_file, "w") as f:
        f.write("# Copyright (c) Huawei Technologies Co., Ltd. 2025.\n")
        f.write("# This file is AUTO-GENERATED by gen_test_copy_general.py. DO NOT EDIT DIRECTLY.\n")
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
        f.write("# " + "-"*76 + "\n")
        f.write("# 1. Shape, Rank Mismatch & Memory Continuity Kernels\n")
        f.write("# " + "-"*76 + "\n\n")

        # Generate shape case arrays (Only call string builders ONCE for structure, no dtype needed)
        shape_cases_list = []
        generated_func_names = set()
        
        for direction in DIRECTIONS:
            for src_shape, dst_shape, src_slice, dst_slice, description in SHAPE_RANK_CASES:
                src_slice_simp = simplify_slice(src_slice, src_shape)
                dst_slice_simp = simplify_slice(dst_slice, dst_shape)
                
                # Deduplicate identical kernel generation for entirely dense accesses without slices
                if src_slice_simp == "" and dst_slice_simp == "":
                    base_name = "dense_same_rank"
                else:
                    base_name = description
                    
                func_name = f"kernel_{base_name}_{direction}"
                
                if func_name not in generated_func_names:
                    kernel_str = gen_shape_rank_kernel_str(src_shape, dst_shape, src_slice, dst_slice, direction, func_name)
                    f.write(kernel_str + "\n\n")
                    generated_func_names.add(func_name)
                
                shape_cases_list.append(f'    ("{description}_{direction}", "{direction}", {src_shape}, {dst_shape}, "{src_slice_simp}", "{dst_slice_simp}", get_{func_name}),\n')

        f.write("SHAPE_RANK_PARAMS = [\n")
        for c in shape_cases_list:
            f.write(c)
        f.write("]\n\n")

        f.write('''@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("desc, direction, src_shape, dst_shape, src_slice_str, dst_slice_str, get_kernel_func", SHAPE_RANK_PARAMS)
def test_shape_rank_mismatch(desc, direction, src_shape, dst_shape, src_slice_str, dst_slice_str, get_kernel_func, dtype):
    kernel_func = get_kernel_func(src_shape, dst_shape, dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    
    inp = gen_tensor(src_shape, dtype, kind='randn')
    out_shape = src_shape if direction == "GM_UB_GM" else dst_shape
    out = gen_tensor(out_shape, dtype, kind='zeros')
    compiled(inp, out)
    
    expected_out = torch.zeros_like(out)
    # Use exec to dynamically slice via the string
    
    if direction == "GM_UB_GM":
        src_access = f"[{src_slice_str}]" if src_slice_str else "[:]"
        exec(f"expected_out{src_access} = inp{src_access}")
    else:
        dst_access = f"[{dst_slice_str}]" if dst_slice_str else "[:]"
        src_access = f"[{src_slice_str}]" if src_slice_str else "[:]"
        exec(f"expected_out{dst_access} = inp{src_access}")
        
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)

''')

        f.write("# " + "-"*76 + "\n")
        f.write("# 2. Dynamic Runtime Bounds Kernels\n")
        f.write("# " + "-"*76 + "\n\n")

        dynamic_cases_list = []
        for direction in DIRECTIONS:
            for dyn_case in DYNAMIC_SCENARIOS:
                func_name = f"kernel_{dyn_case}_{direction}"
                kernel_str = gen_dynamic_kernel_str(dyn_case, direction, func_name)
                f.write(kernel_str + "\n\n")
                # Removed the complex UB_to_GM skipping since we use GM_UB_GM natively
                dynamic_cases_list.append(f'    ("{dyn_case}", "{direction}", get_{func_name}, False),\n')

        f.write("DYNAMIC_PARAMS = [\n")
        for c in dynamic_cases_list:
            f.write(c)
        f.write("]\n\n")

        f.write('''@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dyn_case, direction, get_kernel_func, skip", DYNAMIC_PARAMS)
def test_dynamic_scenarios(dyn_case, direction, get_kernel_func, skip, dtype):
    if skip:
        pytest.skip("Dynamic UB->GM tests skipped due to unified memory overflow bugs in simple AST mocking")
        
    kernel_func = get_kernel_func(dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    
    if dyn_case == "dynamic_from_func_args":
        inp = gen_tensor((1024,), dtype, kind='randn')
        out = gen_tensor((1024,), dtype, kind='zeros')
        shape_N = 620
        compiled(inp, out, shape_N)
        expected_out = torch.zeros_like(out)
        expected_out[0:shape_N] = inp[0:shape_N]
    elif dyn_case in ("dynamic_from_kernel_cid", "dynamic_from_min_max_arithmetic"):
        shape = (4, 256) if dyn_case == "dynamic_from_kernel_cid" else (1000,)
        inp = gen_tensor(shape, dtype, kind='randn')
        out = gen_tensor(shape, dtype, kind='zeros')
        compiled(inp, out)
        expected_out = torch.zeros_like(out)
        if direction == "GM_UB_GM":
            if dyn_case == "dynamic_from_kernel_cid":
                for cid in range(4):
                    expected_out[cid, 0:256] = inp[cid, 0:256]
            else:
                for cid in range(4):
                    offset = cid * 256
                    remain = 1000 - offset
                    length = min(256, remain)
                    expected_out[offset:offset+length] = inp[offset:offset+length]
        else:
            expected_out[...] = inp[...]
            
    elif dyn_case == "dynamic_from_for_loop_iterator":
        inp = gen_tensor((8, 128), dtype, kind='randn')
        out = gen_tensor((8, 128), dtype, kind='zeros')
        compiled(inp, out)
        expected_out = torch.zeros_like(out)
        if direction == "GM_UB_GM":
            for cid in range(2):
                for i in range(4):
                    row = cid * 4 + i
                    expected_out[row, 0:128] = inp[row, 0:128]
        else:
            expected_out[...] = inp[...]
            
    elif dyn_case == "dynamic_loaded_from_ub_buffer":
        inp = gen_tensor((4, 256), dtype, kind='randn')
        out = gen_tensor((4, 256), dtype, kind='zeros')
        indices = torch.tensor([3, 2, 1, 0], dtype=torch.int32).npu()
        compiled(inp, indices, out)
        expected_out = torch.zeros_like(out)
        for i in range(4):
            if direction == "GM_UB_GM":
                expected_out[indices[i].item(), :] = inp[indices[i].item(), :]
            else:
                expected_out[i, :] = inp[indices[i].item(), :]
            
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)

''')

        f.write("# " + "-"*76 + "\n")
        f.write("# 3. Explicit vs Implicit Range Syntaxes Kernels\n")
        f.write("# " + "-"*76 + "\n\n")

        syntax_cases_list = []
        for syntax in SYNTAX_STYLES:
            func_name = f"kernel_syntax_{syntax}"
            kernel_str = gen_syntax_kernel_str(syntax, func_name)
            f.write(kernel_str + "\n\n")
            syntax_cases_list.append(f'    ("{syntax}", get_{func_name}),\n')

        f.write("SYNTAX_PARAMS = [\n")
        for c in syntax_cases_list:
            f.write(c)
        f.write("]\n\n")
        
        f.write('''@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("syntax, get_kernel_func", SYNTAX_PARAMS)
def test_syntax_styles(syntax, get_kernel_func, dtype):
    kernel_func = get_kernel_func(dtype)
    compiled = tilelang.compile(kernel_func, target='npuir')
    shape = (128,) if syntax == "full_tensor" else (8, 128)
    inp = gen_tensor(shape, dtype, kind='randn')
    out = gen_tensor(shape, dtype, kind='zeros')
    compiled(inp, out)
    expected_out = inp.clone()
    assert_close(out.cpu(), expected_out.cpu(), dtype=dtype, rtol=1e-2, atol=1e-2)
''')
                  
    print(f"Generated {out_file} successfully.")

if __name__ == '__main__':
    main()
