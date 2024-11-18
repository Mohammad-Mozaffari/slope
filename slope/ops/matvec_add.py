import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['d', 'bs'],
)
@triton.jit
def matvec_add_kernel(
        matrix_ptr,  # Pointer to the matrix
        vector_ptr,  # Pointer to the vector
        result_ptr,  # Pointer to the result matrix
        bs,  # Batch size (number of rows in the matrix)
        d,  # Dimension size (number of columns in the matrix)
        BLOCK_SIZE: tl.constexpr  # Block size (tuning parameter)
):
    # Compute the row and column indices
    row_idx = tl.program_id(0)  # 1D grid: program_id(0) gives the row index
    col_offset = tl.arange(0, BLOCK_SIZE)  # column indices within the block

    for col_idx in range(0, d, BLOCK_SIZE):
        # Compute the pointers to the matrix and vector elements
        row_ptr = matrix_ptr + row_idx * d + col_offset + col_idx
        vec_ptr = vector_ptr + col_offset + col_idx

        # Mask out-of-bounds accesses (in case d is not a multiple of BLOCK_SIZE)
        mask = col_idx + col_offset < d

        # Load matrix row and vector elements
        matrix_val = tl.load(row_ptr, mask=mask)
        vector_val = tl.load(vec_ptr, mask=mask)

        # Perform the matrix-vector addition
        result_val = matrix_val + vector_val

        # Store the result
        tl.store(result_ptr + row_idx * d + col_offset + col_idx, result_val, mask=mask)


def matvec_add(matrix, vector):
    if matrix.dim() == 2:
        bs, d = matrix.shape
    elif matrix.dim() == 3:
        b, s, d = matrix.shape
        bs = b * s
    else:
        raise NotImplementedError("Only 2D and 3D matrices are supported")

    # Ensure that vector length matches the number of columns in the matrix
    assert vector.shape[0] == d

    # Create an empty tensor for the result
    result = torch.empty_like(matrix)

    # Launch the kernel with 1D grid of `bs` (number of rows in matrix)
    grid = (bs,)

    matvec_add_kernel[grid](
        matrix, vector, result, bs, d,
    )

    return result


if __name__ == "__main__":
    # Example usage
    bs, seq, d = 2, 512, 1024  # Batch size and dimension
    matrix = torch.randn(bs, seq, d, device='cuda')
    vector = torch.randn(d, device='cuda')

    result = matvec_add(matrix, vector)

    result2 = matrix + vector  # Verify the result

    # Check the correctness
    print(torch.norm(result.float() - result2.float()) / torch.norm(result2.float()))

# import triton
# import triton.language as tl
# import torch

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 8}, num_warps=1),
#         triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 16}, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16}, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16}, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64}, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8),
#     ],
#     key=['bs', 'd'],
# )
# @triton.jit
# def matvec_add_kernel(
#     matrix_ptr,   # Pointer to the matrix
#     vector_ptr,   # Pointer to the vector
#     result_ptr,   # Pointer to the result matrix
#     bs,           # Batch size (number of rows in the matrix)
#     d,            # Dimension size (number of columns in the matrix)
#     BLOCK_SIZE_M: tl.constexpr,  # Number of rows per block
#     BLOCK_SIZE_N: tl.constexpr,  # Number of columns per block
# ):
#     # Compute the row and column indices
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(bs, BLOCK_SIZE_M)
#     pid_m = pid % num_pid_m
#     pid_n = pid // num_pid_m

#     # Compute the starting row for this thread block
#     row_start = pid_m * BLOCK_SIZE_M
#     row_offsets = tl.arange(0, BLOCK_SIZE_M)
#     row_indices = row_start + row_offsets

#     # Compute the starting column for this thread block
#     col_start = pid_n * BLOCK_SIZE_N

#     # Mask for rows
#     row_mask = row_indices < bs

#     col_offsets = tl.arange(0, BLOCK_SIZE_N)
#     # Iterate over columns
#     for col in range(0, d, BLOCK_SIZE_N):
#         col_indices = col + col_offsets

#         # Mask for columns
#         col_mask = col_indices < d

#         # Combined mask
#         mask = row_mask[:, None] & col_mask[None, :]

#         # Load matrix elements
#         matrix_ptrs = matrix_ptr + row_indices[:, None] * d + col_indices[None, :]
#         matrix_vals = tl.load(matrix_ptrs, mask=mask)

#         # Load vector elements
#         vector_ptrs = vector_ptr + col_indices
#         vector_vals = tl.load(vector_ptrs, mask=col_mask)

#         # Perform the addition
#         result_vals = matrix_vals + vector_vals[None, :]

#         # Store the result
#         result_ptrs = result_ptr + row_indices[:, None] * d + col_indices[None, :]
#         tl.store(result_ptrs, result_vals, mask=mask)

# def matvec_add(matrix, vector):
#     if matrix.dim() == 2:
#         bs, d = matrix.shape
#     elif matrix.dim() == 3:
#         b, s, d = matrix.shape
#         bs = b * s
#     else:
#         raise NotImplementedError("Only 2D and 3D matrices are supported")

#     # Ensure that vector length matches the number of columns in the matrix
#     assert vector.shape[0] == d

#     # Create an empty tensor for the result
#     result = torch.empty_like(matrix)

#     # Calculate grid size
#     grid = lambda meta: (
#         triton.cdiv(bs, meta['BLOCK_SIZE_M']),
#     )

#     # Launch the kernel
#     matvec_add_kernel[grid](
#         matrix, vector, result,
#         bs, d,
#     )

#     return result

# if __name__ == "__main__":
#     # Example usage
#     bs, seq, d = 2, 512, 1024  # Batch size and dimension
#     matrix = torch.randn(bs, seq, d, device='cuda')
#     vector = torch.randn(d, device='cuda')
#     result = matvec_add(matrix, vector)
#     result2 = matrix + vector  # Verify the result
#     # Check the correctness
#     print(torch.norm(result.float() - result2.float()) / torch.norm(result2.float()))