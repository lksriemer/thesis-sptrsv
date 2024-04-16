// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>


#include <cuda.h>
#include <cusparse.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/memory.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace solver {


// struct SolveStruct {
//     virtual ~SolveStruct() = default;
// };


}  // namespace solver


namespace kernels {
namespace cuda {

// TODO: This namespace commented out for ffi
// namespace {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11031))


template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    cusparseHandle_t handle;
    cusparseSpSMDescr_t spsm_descr;
    cusparseSpMatDescr_t descr_a;
    size_type num_rhs;

    // Implicit parameter in spsm_solve, therefore stored here.
    array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper, bool unit_diag)
        : handle{exec->get_cusparse_handle()},
          spsm_descr{},
          descr_a{},
          num_rhs{num_rhs},
          work{exec}
    {
        if (num_rhs == 0) {
            return;
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        spsm_descr = cusparse::create_spsm_descr();
        descr_a = cusparse::create_csr(
            matrix->get_size()[0], matrix->get_size()[1],
            matrix->get_num_stored_elements(),
            const_cast<IndexType*>(matrix->get_const_row_ptrs()),
            const_cast<IndexType*>(matrix->get_const_col_idxs()),
            const_cast<ValueType*>(matrix->get_const_values()));
        cusparse::set_attribute<cusparseFillMode_t>(
            descr_a, CUSPARSE_SPMAT_FILL_MODE,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        cusparse::set_attribute<cusparseDiagType_t>(
            descr_a, CUSPARSE_SPMAT_DIAG_TYPE,
            unit_diag ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT);

        const auto rows = matrix->get_size()[0];
        // workaround suggested by NVIDIA engineers: for some reason
        // cusparse needs non-nullptr input vectors even for analysis
        auto descr_b = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAD));
        auto descr_c = cusparse::create_dnmat(
            dim<2>{matrix->get_size()[0], num_rhs}, matrix->get_size()[1],
            reinterpret_cast<ValueType*>(0xDEAF));

        auto work_size = cusparse::spsm_buffer_size(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(), descr_a,
            descr_b, descr_c, CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        work.resize_and_reset(work_size);

        cusparse::spsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                one<ValueType>(), descr_a, descr_b, descr_c,
                                CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr,
                                work.get_data());

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    void solve(const matrix::Csr<ValueType, IndexType>*,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        if (input->get_size()[1] != num_rhs) {
            throw gko::ValueMismatch{
                __FILE__,
                __LINE__,
                __FUNCTION__,
                input->get_size()[1],
                num_rhs,
                "the dimensions of the multivector do not match the value "
                "provided at generation time. Check the value specified in "
                ".with_num_rhs(...)."};
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        auto descr_b = cusparse::create_dnmat(
            input->get_size(), input->get_stride(),
            const_cast<ValueType*>(input->get_const_values()));
        auto descr_c = cusparse::create_dnmat(
            output->get_size(), output->get_stride(), output->get_values());

        cusparse::spsm_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, one<ValueType>(),
                             descr_a, descr_b, descr_c,
                             CUSPARSE_SPSM_ALG_DEFAULT, spsm_descr);

        cusparse::destroy(descr_b);
        cusparse::destroy(descr_c);
    }

    ~CudaSolveStruct()
    {
        if (descr_a) {
            cusparse::destroy(descr_a);
            descr_a = nullptr;
        }
        if (spsm_descr) {
            cusparse::destroy(spsm_descr);
            spsm_descr = nullptr;
        }
    }

    CudaSolveStruct(const SolveStruct&) = delete;

    CudaSolveStruct(SolveStruct&&) = delete;

    CudaSolveStruct& operator=(const SolveStruct&) = delete;

    CudaSolveStruct& operator=(SolveStruct&&) = delete;
};


#elif (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))

template <typename ValueType, typename IndexType>
struct CudaSolveStruct : gko::solver::SolveStruct {
    std::shared_ptr<const gko::CudaExecutor> exec;
    cusparseHandle_t handle;
    int algorithm;
    csrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    size_type num_rhs;
    mutable array<char> work;

    CudaSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* matrix,
                    size_type num_rhs, bool is_upper, bool unit_diag)
        : exec{exec},
          handle{exec->get_cusparse_handle()},
          algorithm{},
          solve_info{},
          policy{},
          factor_descr{},
          num_rhs{num_rhs},
          work{exec}
    {
        if (num_rhs == 0) {
            return;
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        factor_descr = cusparse::create_mat_descr();
        solve_info = cusparse::create_solve_info();
        cusparse::set_mat_fill_mode(
            factor_descr,
            is_upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        cusparse::set_mat_diag_type(
            factor_descr,
            unit_diag ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT);
        algorithm = 0;
        policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

        size_type work_size{};

        cusparse::buffer_size_ext(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            &work_size);

        // allocate workspace
        work.resize_and_reset(work_size);

        cusparse::csrsm2_analysis(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), one<ValueType>(), factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs, solve_info, policy,
            work.get_data());
    }

    void solve(const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* input,
               matrix::Dense<ValueType>* output, matrix::Dense<ValueType>*,
               matrix::Dense<ValueType>*) const
    {
        if (input->get_size()[1] != num_rhs) {
            throw gko::ValueMismatch{
                __FILE__,
                __LINE__,
                __FUNCTION__,
                input->get_size()[1],
                num_rhs,
                "the dimensions of the multivector do not match the value "
                "provided at generation time. Check the value specified in "
                ".with_num_rhs(...)."};
        }
        cusparse::pointer_mode_guard pm_guard(handle);
        dense::copy(exec, input, output);
        cusparse::csrsm2_solve(
            handle, algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
            output->get_stride(), matrix->get_num_stored_elements(),
            one<ValueType>(), factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            output->get_values(), output->get_stride(), solve_info, policy,
            work.get_data());
    }

    ~CudaSolveStruct()
    {
        if (factor_descr) {
            cusparse::destroy(factor_descr);
            factor_descr = nullptr;
        }
        if (solve_info) {
            cusparse::destroy(solve_info);
            solve_info = nullptr;
        }
    }

    CudaSolveStruct(const CudaSolveStruct&) = delete;

    CudaSolveStruct(CudaSolveStruct&&) = delete;

    CudaSolveStruct& operator=(const CudaSolveStruct&) = delete;

    CudaSolveStruct& operator=(CudaSolveStruct&&) = delete;
};


#endif


constexpr int default_block_size = 512;
constexpr int fallback_block_size = 32;


__device__ __forceinline__ void atom_red_add_relaxed(float* ptr, float add)
{
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(ptr), "f"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_relaxed(double* ptr, double add)
{
    asm volatile("red.relaxed.gpu.global.add.f64 [%0], %1;" ::"l"(ptr), "d"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_relaxed(thrust::complex<float>* ptr, thrust::complex<float> add)
{
    // unimplemented
}

__device__ __forceinline__ void atom_red_add_relaxed(thrust::complex<double>* ptr, thrust::complex<double> add)
{
    // unimplemented
}

__device__ __forceinline__ void atom_red_add_release(float* ptr, float add)
{
    asm volatile("red.release.gpu.global.add.f32 [%0], %1;" ::"l"(ptr), "f"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_release(double* ptr, double add)
{
    asm volatile("red.release.gpu.global.add.f64 [%0], %1;" ::"l"(ptr), "d"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_release(thrust::complex<float>* ptr, thrust::complex<float> add)
{
    // unimplemented
}

__device__ __forceinline__ void atom_red_add_release(thrust::complex<double>* ptr, thrust::complex<double> add)
{
    // unimplemented
}

__device__ __forceinline__ void atom_red_add_relaxed(uint32* ptr, uint32 add)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_relaxed(uint64* ptr, uint64 add)
{
    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(ptr), "l"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_release(uint32* ptr, uint32 add)
{
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_release(uint64* ptr, uint64 add)
{
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(ptr), "l"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_relaxed(int32* ptr, int32 add)
{
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(add)
                 : "memory");
}

__device__ __forceinline__ void atom_red_add_release(int32* ptr, int32 add)
{
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(add)
                 : "memory");
}

template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_csc_balanced_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const vals, 
    const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, 
    int32 *const row_counts,
    const IndexType *const col_assignments,
    const int32 *const col_assignment_sizes,
    const size_type vn, const size_type nrhs, bool unit_diag)
{
        const auto full_gid = thread::get_thread_id_flat();

        const auto rhs = full_gid % nrhs;
        const auto gid = full_gid / nrhs;
        const auto vcol = is_upper ? vn - 1 - gid : gid;

        
        
        if(gid >= vn){
            return;
        }
        
        const auto col = col_assignments[vcol];
        const auto col_assignment_size = col_assignment_sizes[col + 1] - col_assignment_sizes[col];
        const auto vid = gid - col_assignment_sizes[col];

        const auto col_begin = colptrs[col] + 1;
        const auto col_end = colptrs[col + 1];

        const auto b_val = b[col];
        const auto diag = vals[colptrs[col]];

        while(load_relaxed(row_counts + col) > zero<IndexType>()) { }

        const auto val = -load_relaxed(x + col) + b_val;

        // if(col == 0){
        //     printf("vcol %d col %d vid %d \n", (int32)vcol, (int32)col, (int32)vid);
        // }

        for(auto i = col_begin + vid; i < col_end; i += col_assignment_size){
            
            // if(col_end - i >= 4) {
            //     const auto row_1 = rowidxs[i];
            //     const auto row_2 = rowidxs[i + 1];
            //     const auto row_3 = rowidxs[i + 2];
            //     const auto row_4 = rowidxs[i + 3];
            //     const auto factor_1 = vals[i];
            //     const auto factor_2 = vals[i + 1];
            //     const auto factor_3 = vals[i + 2];
            //     const auto factor_4 = vals[i + 3];

            //     atom_red_add_relaxed(x + row_1, val * factor_1);
            //     atom_red_add_relaxed(x + row_2, val * factor_2);
            //     atom_red_add_relaxed(x + row_3, val * factor_3);
            //     atom_red_add_relaxed(x + row_4, val * factor_4);
                
            //     i += 4;
            //     // __threadfence();

            //     atom_red_add_release(row_counts + row_1, -1);
            //     atom_red_add_release(row_counts + row_2, -1);
            //     atom_red_add_release(row_counts + row_3, -1);
            //     atom_red_add_release(row_counts + row_4, -1);
                
            // }else{
                const auto row = rowidxs[i];
                const auto factor = vals[i];

                atomic_add(x + row, val * factor);
                // __threadfence();
                atomic_add(row_counts + row, -1);

                // ++i;
            // }
            
            
        }

        // __threadfence();

        // for(auto i = col_begin; i < col_end; ++i){
        //     const auto row = rowidxs[i];

        //     atom_red_add_relaxed(row_counts + row, -1);
        // }

        const auto vcol_count = atomic_add(row_counts + col, -1);
        if(vcol_count - 1 == -col_assignment_size){ // Only the last one should do this store
            store_relaxed(x + col, val); 
        }
        
}

template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const vals, 
    const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, 
    int32 *const row_counts,
    const size_type n, const size_type nrhs, bool unit_diag)
{
        const auto full_gid = thread::get_thread_id_flat();

        const auto rhs = full_gid % nrhs;
        const auto gid = full_gid / nrhs;
        const auto col = is_upper ? n - 1 - gid : gid;

        
        
        if(gid >= n){
            return;
        }

        const auto col_begin = colptrs[col] + 1;
        const auto col_end = colptrs[col + 1];

        const auto b_val = b[col];
        const auto diag = vals[colptrs[col]];

        while(load_relaxed(row_counts + col) > zero<IndexType>()) { }

        const auto val = -load_relaxed(x + col) + b_val;

        // if(col == 0){
        //     printf("vcol %d col %d vid %d \n", (int32)vcol, (int32)col, (int32)vid);
        // }

        for(auto i = col_begin; i < col_end; ++i){
            const auto row = rowidxs[i];
            const auto factor = vals[i];

            atomic_add(x + row, val * factor);
            // __threadfence();
            atomic_add(row_counts + row, -1);
        }
        
        store_relaxed(x + col, val); 
}

template <typename IndexType>
__global__ void sptrsv_csc_rowcounts_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    int32 *const row_counts, const size_type n) 
{
        const auto full_gid = thread::get_thread_id_flat();
        const auto row = full_gid;

        if(full_gid >= n){
            return;
        }
        
        const auto this_nnz = rowptrs[row + 1] - rowptrs[row];
        row_counts[row] = (int32)(this_nnz - 1);
}


template <typename IndexType>
__global__ void sptrsv_csc_colsizes_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    int32 *const assigned_vwarp_sizes, const size_type n) {
        const auto full_gid = thread::get_thread_id_flat();
        const auto col = full_gid;

        if(full_gid >= n){
            return;
        }

        const auto this_nnz = colptrs[col + 1] - colptrs[col];

        const auto avg_nnz_per_col = (float)(colptrs[n]) / (float)n;

        assigned_vwarp_sizes[col] = (int32)max(1, min(__float2int_rn(((float)this_nnz) / avg_nnz_per_col), 32));
    }

template <typename IndexType>
__global__ void sptrsv_csc_writeassignments_kernel(
    IndexType *const assignments, const int32 *const assigned_vwarp_sizes, 
    const size_type n, const size_type vn) {
        const auto full_gid = thread::get_thread_id_flat();
        const auto col = full_gid;

        if(full_gid >= n){
            return;
        }

        for(auto vcol = assigned_vwarp_sizes[col]; vcol < assigned_vwarp_sizes[col + 1]; ++vcol){
            assignments[vcol] = col;
        }
    }


template <typename ValueType, typename IndexType>
struct SptrsvebccbnSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> scaled_m;
    std::unique_ptr<matrix::Dense<ValueType>> new_b;

    array<int32> assigned_sizes;
    array<IndexType> assignments;
    int32 vn;

    array<int32> row_counts;
    mutable array<int32> row_counts_clone;
    
    int num_sms;
    int max_threads_per_sm;

    SptrsvebccbnSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          diag{matrix->extract_diagonal()},
          row_counts{exec, matrix->get_size()[0]},
          row_counts_clone{exec, matrix->get_size()[0]},
          assigned_sizes{exec, matrix->get_size()[0] + 1}
    {
        const auto is_fallback_required = exec->get_major_version() < 7;
        const auto n = matrix->get_size()[0];

        cudaDeviceGetAttribute(&max_threads_per_sm, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, exec->get_device_id());
        num_sms = exec->get_num_multiprocessor();

        scaled_m = matrix::Csr<ValueType, IndexType>::create(exec);
        scaled_m->copy_from(matrix);
        diag->inverse_apply(matrix, scaled_m.get());

        // Get row counts kernel (could be done in parallel with unit_diag create kernel)
        const dim3 block_size(default_block_size, 1,1);
        const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
        sptrsv_csc_rowcounts_kernel<<<grid_size, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(), 
            row_counts.get_data(), n);

        scaled_m = gko::as<matrix::Csr<ValueType, IndexType>>(scaled_m->transpose());

        sptrsv_csc_colsizes_kernel<<<grid_size, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(), 
            assigned_sizes.get_data(), n);

        components::prefix_sum_nonnegative(exec, assigned_sizes.get_data(), n + 1);

        // printf("at n: %d at n+1: %d\n", (int32)exec->copy_val_to_host(assigned_sizes.get_const_data() + n - 1), 
        //     (int32)exec->copy_val_to_host(assigned_sizes.get_const_data() + n));

        vn = (int32)exec->copy_val_to_host(assigned_sizes.get_const_data() + n);
        
        assignments = array<IndexType>{exec, vn};

        const dim3 vn_block_size(default_block_size, 1,1);
        const dim3 vn_grid_size(ceildiv(vn, block_size.x), 1, 1);
        sptrsv_csc_writeassignments_kernel<<<vn_grid_size, vn_block_size>>>(
            assignments.get_data(), assigned_sizes.get_const_data(), n, vn);

        new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];

        cudaMemset(x->get_values(), 0, n * sizeof(ValueType));
        exec->copy(n, row_counts.get_const_data(), row_counts_clone.get_data());

        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());

        const dim3 block_size(default_block_size, 1,1);
        const dim3 grid_size(ceildiv(vn, block_size.x), 1, 1);

        sptrsv_csc_balanced_kernel<false><<<grid_size, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(),
            as_cuda_type(scaled_m->get_const_values()), 
            as_cuda_type(new_b->get_const_values()), new_b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), 
            row_counts_clone.get_data(), 
            assignments.get_const_data(),
            assigned_sizes.get_const_data(),
            vn, 1, unit_diag);
        
        const auto dbg_5353 = 0;
    }
};


template <typename ValueType, typename IndexType>
struct SptrsvebccnSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> scaled_m;
    std::unique_ptr<matrix::Dense<ValueType>> new_b;

    array<int32> row_counts;
    mutable array<int32> row_counts_clone;
    
    int num_sms;
    int max_threads_per_sm;

    SptrsvebccnSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          diag{matrix->extract_diagonal()},
          row_counts{exec, matrix->get_size()[0]},
          row_counts_clone{exec, matrix->get_size()[0]}
    {
        const auto is_fallback_required = exec->get_major_version() < 7;
        const auto n = matrix->get_size()[0];

        cudaDeviceGetAttribute(&max_threads_per_sm, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, exec->get_device_id());
        num_sms = exec->get_num_multiprocessor();

        scaled_m = matrix::Csr<ValueType, IndexType>::create(exec);
        scaled_m->copy_from(matrix);
        diag->inverse_apply(matrix, scaled_m.get());

        // Get row counts kernel (could be done in parallel with unit_diag create kernel)
        const dim3 block_size(default_block_size, 1,1);
        const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
        sptrsv_csc_rowcounts_kernel<<<grid_size, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(), 
            row_counts.get_data(), n);

        scaled_m = gko::as<matrix::Csr<ValueType, IndexType>>(scaled_m->transpose());

        new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];

        cudaMemset(x->get_values(), 0, n * sizeof(ValueType));
        exec->copy(n, row_counts.get_const_data(), row_counts_clone.get_data());

        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());

        const dim3 block_size(default_block_size, 1,1);
        const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);

        sptrsv_csc_kernel<false><<<grid_size, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(),
            as_cuda_type(scaled_m->get_const_values()), 
            as_cuda_type(new_b->get_const_values()), new_b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), 
            row_counts_clone.get_data(),
            n, 1, unit_diag);
        
        const auto dbg_987 = 0;
    }
};


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;


    const auto full_gid = thread::get_thread_id_flat<IndexType>();
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        int sleep = 0;
        while (is_nan(x)) {
            __nanosleep(sleep);
            x = load_relaxed(x_p);
            if (sleep < 4000) {
                sleep += 2;
            }
        }

        sum += x * vals[i];
    }

    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];

    const auto r = (b[row * b_stride + rhs] - sum) / diag;

    store_relaxed_shared(x_s + self_shid, r);
    store_relaxed_noL1(x + row * x_stride + rhs, r);

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store_relaxed_shared(x_s + self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <bool is_upper, bool spin, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_warpperrow_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    using WarpReduce = cub::WarpReduce<ValueType, 32>;

    __shared__ typename WarpReduce::TempStorage cub_red_storage[default_block_size / 32];

    const auto full_gid = thread::get_thread_id_flat<IndexType>();
    const auto tid = full_gid % 32;
    const auto warp_gid = full_gid / 32;
    const auto rhs = warp_gid % nrhs;
    const auto gid = warp_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -32 : 32;

    auto sum = zero<ValueType>();
    auto i = row_begin + tid;
    for (; i < row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        ValueType x = *x_p;
        int sleep = 0;
        while (is_nan(x)) {
            if constexpr (spin){
                __nanosleep(sleep);
            }
            
            x = load_relaxed(x_p);
            if constexpr (spin){
                    if (sleep < 4000) {
                    sleep += 2;
                }
            }
        }

        sum += x * vals[i];
    }
    
    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];

    const auto aggr_sum = WarpReduce(cub_red_storage[warp_gid]).Sum(sum);

    const auto r = (b[row * b_stride + rhs] - aggr_sum) / diag;

    if(tid == 0){
        store_relaxed(x + row * x_stride + rhs, r);

        // This check to ensure no infinte loops happen.
        if (is_nan(r)) {
            x[row * x_stride + rhs] = zero<ValueType>();
            *nan_produced = true;
        }
    }
    
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void __launch_bounds__(512) sptrsv_naive_caching_multirow_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, ValueType* const x,
    const size_type n, const size_type nrhs, bool unit_diag, bool* nan_produced)
{
    constexpr int mrowc = 6;
    // uninitialized_array<ValueType, mrowc * default_block_size> sums_arr;
    // const auto sums = sums_arr + threadIdx.x * mrowc;

    const auto full_gid = thread::get_thread_id_flat<IndexType>();
    auto row_1 = is_upper ? n - 1 - mrowc * full_gid : mrowc * full_gid;
    auto row_2 = is_upper ? row_1 - 1 : row_1 + 1;
    auto row_3 = is_upper ? row_1 - 2 : row_1 + 2;
    auto row_4 = is_upper ? row_1 - 3 : row_1 + 3;
    const auto row_5 = is_upper ? row_1 - 4 : row_1 + 4;
    const auto row_6 = is_upper ? row_1 - 5 : row_1 + 5;
    // const auto row_7 = is_upper ? row_1 - 6 : row_1 + 6;
    // const auto row_8 = is_upper ? row_1 - 7 : row_1 + 7;

    if (full_gid >= (n + mrowc - 1) / mrowc) {
        return;
    }

    // store_relaxed_shared(sums + 0, zero<ValueType>());
    // store_relaxed_shared(sums + 1, zero<ValueType>());
    // store_relaxed_shared(sums + 2, zero<ValueType>());
    // store_relaxed_shared(sums + 3, zero<ValueType>());
    // store_relaxed_shared(sums + 4, zero<ValueType>());
    // store_relaxed_shared(sums + 5, zero<ValueType>());
    // store_relaxed_shared(sums + 6, zero<ValueType>());
    // store_relaxed_shared(sums + 7, zero<ValueType>());

    int done_count = 0;

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    auto i_1 =
        is_upper ? rowptrs[row_1 + 1] - 1 : (row_1 < n ? rowptrs[row_1] : 0);
    auto row_end_1 = is_upper ? rowptrs[row_1] - 1
                              : (row_1 < n ? (rowptrs[row_1 + 1] - 1) : 0);
    auto i_2 =
        is_upper ? rowptrs[row_2 + 1] - 1 : (row_2 < n ? rowptrs[row_2] : 0);
    auto row_end_2 = is_upper ? rowptrs[row_2] - 1
                              : (row_2 < n ? (rowptrs[row_2 + 1] - 1) : 0);
    auto i_3 =
        is_upper ? rowptrs[row_3 + 1] - 1 : (row_3 < n ? rowptrs[row_3] : 0);
    auto row_end_3 = is_upper ? rowptrs[row_3] - 1
                              : (row_3 < n ? (rowptrs[row_3 + 1] - 1) : 0);
    auto i_4 =
        is_upper ? rowptrs[row_4 + 1] - 1 : (row_4 < n ? rowptrs[row_4] : 0);
    auto row_end_4 = is_upper ? rowptrs[row_4] - 1
                              : (row_4 < n ? (rowptrs[row_4 + 1] - 1) : 0);
    auto i_5 =
        is_upper ? rowptrs[row_5 + 1] - 1 : (row_5 < n ? rowptrs[row_5] : 0);
    const auto row_end_5 = is_upper
                               ? rowptrs[row_5] - 1
                               : (row_5 < n ? (rowptrs[row_5 + 1] - 1) : 0);
    auto i_6 =
        is_upper ? rowptrs[row_6 + 1] - 1 : (row_6 < n ? rowptrs[row_6] : 0);
    const auto row_end_6 = is_upper
                               ? rowptrs[row_6] - 1
                               : (row_6 < n ? (rowptrs[row_6 + 1] - 1) : 0);
    // auto i_7 = is_upper ? rowptrs[row_7 + 1] - 1 : (row_7 < n ?
    // rowptrs[row_7] : 0); const auto row_end_7 = is_upper ? rowptrs[row_7] - 1
    // : (row_7 < n ? (rowptrs[row_7 + 1] - 1) : 0); auto i_8 = is_upper ?
    // rowptrs[row_8 + 1] - 1 : (row_8 < n ? rowptrs[row_8] : 0); const auto
    // row_end_8 = is_upper ? rowptrs[row_8] - 1 : (row_8 < n ? (rowptrs[row_8 +
    // 1] - 1) : 0); constexpr int row_step = 1; auto i_1 = (row_1 < n ?
    // rowptrs[row_1] : 0); const auto row_end_1 = (row_1 < n ? (rowptrs[row_1 +
    // 1] - 1) : 0); auto i_2 = (row_1 + 1 < n ? rowptrs[row_1 + 1] : 0); const
    // auto row_end_2 = (row_1 + 1 < n ? (rowptrs[row_1 + 2] - 1) : 0); auto i_3
    // = (row_1 + 2 < n ? rowptrs[row_1 + 2] : 0); const auto row_end_3 = (row_1
    // + 2 < n ? (rowptrs[row_1 + 3] - 1) : 0); auto i_4 = (row_1 + 3 < n ?
    // rowptrs[row_1 + 3] : 0); const auto row_end_4 = (row_1 + 3 < n ?
    // (rowptrs[row_1 + 4] - 1) : 0); auto i_5 = (row_1 + 4 < n ? rowptrs[row_1
    // + 4] : 0); const auto row_end_5 = (row_1 + 4 < n ? (rowptrs[row_1 + 5] -
    // 1) : 0); auto i_6 = (row_1 + 5 < n ? rowptrs[row_1 + 5] : 0); const auto
    // row_end_6 = (row_1 + 5 < n ? (rowptrs[row_1 + 6] - 1) : 0);
    // // auto i_7 = (row_1 + 6 < n ? rowptrs[row_1 + 6] : 0);
    // // const auto row_end_7 = (row_1 + 6 < n ? (rowptrs[row_1 + 7] - 1) : 0);
    // // auto i_8 = (row_1 + 7 < n ? rowptrs[row_1 + 7] : 0);
    // // const auto row_end_8 = (row_1 + 7 < n ? (rowptrs[row_1 + 8] - 1) : 0);
    constexpr int row_step = 1;

    bool done_1 = i_1 >= row_end_1;
    bool done_2 = i_2 >= row_end_2;
    bool done_3 = i_3 >= row_end_3;
    bool done_4 = i_4 >= row_end_4;
    bool done_5 = i_5 >= row_end_5;
    bool done_6 = i_6 >= row_end_6;
    // bool done_7 = i_7 >= row_end_7;
    // bool done_8 = i_8 >= row_end_8;

    if (done_1) {
        if (row_1 < n) {
            store_relaxed(x + row_1, b[row_1] / vals[row_end_1]);
        }
        ++done_count;
        // row_1 = row_5;
        // i_1 = i_5;
        // row_end_1 = row_end_5;
        // done_1 = done_5;
    }
    if (done_2) {
        if (row_2 < n) {
            store_relaxed(x + row_2, b[row_2] / vals[row_end_2]);
        }
        ++done_count;
        // row_2 = row_6;
        // i_2 = i_6;
        // row_end_2 = row_end_6;
        // done_2 = done_6;
    }
    if (done_3) {
        if (row_3 < n) {
            store_relaxed(x + row_3, b[row_3] / vals[row_end_3]);
        }
        ++done_count;
        // row_3 = row_7;
        // i_3 = i_7;
        // row_end_3 = row_end_7;
        // done_3 = done_7;
    }
    if (done_4) {
        if (row_4 < n) {
            store_relaxed(x + row_4, b[row_4] / vals[row_end_4]);
        }
        ++done_count;
        // row_4 = row_8;
        // i_4 = i_8;
        // row_end_4 = row_end_8;
        // done_4 = done_8;
    }
    if (done_5) {
        if (row_5 < n) {
            store_relaxed(x + row_5, b[row_5] / vals[row_end_5]);
        }
        ++done_count;
    }
    if (done_6) {
        if (row_6 < n) {
            store_relaxed(x + row_6, b[row_6] / vals[row_end_6]);
        }
        ++done_count;
    }
    // if(done_7){
    //     if(row_7 < n){
    //         store_relaxed(x + row_7, b[row_7] / vals[row_end_7]);
    //     }
    //     ++done_count;
    // }
    // if(done_8){
    //     if(row_8 < n){
    //         store_relaxed(x + row_8, b[row_8] / vals[row_end_8]);
    //     }
    //     ++done_count;
    // }

    auto dependency_1 = colidxs[i_1];
    auto dependency_2 = colidxs[i_2];
    auto dependency_3 = colidxs[i_3];
    auto dependency_4 = colidxs[i_4];
    auto dependency_5 = colidxs[i_5];
    auto dependency_6 = colidxs[i_6];
    // auto dependency_7 = colidxs[i_7];

    auto sum_1 = zero<ValueType>();
    auto sum_2 = zero<ValueType>();
    auto sum_3 = zero<ValueType>();
    auto sum_4 = zero<ValueType>();
    auto sum_5 = zero<ValueType>();
    auto sum_6 = zero<ValueType>();
    // auto sum_7 = zero<ValueType>();

    // auto spin = 0;
    while (done_count < mrowc) {
        // ++spin;
        // if(spin == 1000000){
        //     printf("Gid %d row %d-%d keeps on spinning because %d<8\n",
        //     (int32)full_gid, (int32)row_1, (int32)row_2, (int32)done_count);
        // }

        // if(done_1){
        //     goto after_1;
        // }
        if (!done_1) {
            auto load = load_relaxed(x + dependency_1);
            if (!is_nan(load)) {
                sum_1 += load * vals[i_1];
                // store_relaxed_shared(sums + 0, load_relaxed_shared(sums + 0)
                // + load * vals[i_1]);
                i_1 += row_step;
                dependency_1 = colidxs[i_1];
                done_1 = i_1 >= row_end_1;

                if (done_1) {
                    const auto r_1 = (b[row_1] - sum_1) / vals[i_1];
                    // const auto r_1 = (b[row_1] - load_relaxed_shared(sums +
                    // 0)) / vals[i_1];
                    store_relaxed(x + row_1, r_1);

                    ++done_count;
                    // if(row_1 < row_5){
                    //     row_1 = row_5;
                    //     i_1 = i_5;
                    //     dependency_1 = colidxs[i_5];
                    //     row_end_1 = row_end_5;
                    //     sum_1 = zero<ValueType>();
                    //     done_1 = done_5;
                    // }
                }
            }
        }
        // after_1:

        // if(done_2){
        //     goto after_2;
        // }
        if (!done_2) {
            auto load = load_relaxed(x + dependency_2);
            if (!is_nan(load)) {
                sum_2 += load * vals[i_2];
                // store_relaxed_shared(sums + 1, load_relaxed_shared(sums + 1)
                // + load * vals[i_2]);
                i_2 += row_step;
                dependency_2 = colidxs[i_2];
                done_2 = i_2 >= row_end_2;

                if (done_2) {
                    const auto r_2 = (b[row_2] - sum_2) / vals[i_2];
                    // const auto r_2 = (b[row_2] - load_relaxed_shared(sums +
                    // 1)) / vals[i_2];
                    store_relaxed(x + row_2, r_2);
                    ++done_count;
                    // if(row_2 < row_6){
                    //     row_2 = row_6;
                    //     i_2 = i_6;
                    //     dependency_2 = colidxs[i_6];
                    //     row_end_2 = row_end_6;
                    //     sum_2 = zero<ValueType>();
                    //     done_2 = done_6;
                    // }
                }
            }
        }
        // after_2:

        // if(done_3){
        //     goto after_3;
        // }
        if (!done_3) {
            auto load = load_relaxed(x + dependency_3);
            if (!is_nan(load)) {
                sum_3 += load * vals[i_3];
                // store_relaxed_shared(sums + 2, load_relaxed_shared(sums + 2)
                // + load * vals[i_3]);
                i_3 += row_step;
                dependency_3 = colidxs[i_3];
                done_3 = i_3 >= row_end_3;

                if (done_3) {
                    const auto r_3 = (b[row_3] - sum_3) / vals[i_3];
                    // const auto r_3 = (b[row_3] - load_relaxed_shared(sums +
                    // 2)) / vals[i_3];
                    store_relaxed(x + row_3, r_3);
                    ++done_count;
                    // if(row_3 < row_7){
                    //     row_3 = row_7;
                    //     i_3 = i_7;
                    //     dependency_3 = colidxs[i_7];
                    //     row_end_3 = row_end_7;
                    //     sum_3 = zero<ValueType>();
                    //     done_3 = done_7;
                    // }
                }
            }
        }
        // after_3:

        // if(done_4){
        //     goto after_4;
        // }
        if (!done_4) {
            auto load = load_relaxed(x + dependency_4);
            if (!is_nan(load)) {
                sum_4 += load * vals[i_4];
                // store_relaxed_shared(sums + 3, load_relaxed_shared(sums + 3)
                // + load * vals[i_4]);
                i_4 += row_step;
                dependency_4 = colidxs[i_4];
                done_4 = i_4 >= row_end_4;

                if (done_4) {
                    const auto r_4 = (b[row_4] - sum_4) / vals[i_4];
                    // const auto r_4 = (b[row_4] - load_relaxed_shared(sums +
                    // 3)) / vals[i_4];
                    store_relaxed(x + row_4, r_4);
                    ++done_count;
                    // if(row_4 < row_8){
                    //     row_4 = row_8;
                    //     i_4 = i_8;
                    //     dependency_4 = colidxs[i_8];
                    //     row_end_4 = row_end_8;
                    //     sum_4 = zero<ValueType>();
                    //     done_4 = done_8;
                    // }
                }
            }
        }
        // after_4:

        // if(done_5){
        //     goto after_5;
        // }
        if (!done_5) {
            auto load = load_relaxed(x + dependency_5);
            if (!is_nan(load)) {
                sum_5 += load * vals[i_5];
                // store_relaxed_shared(sums + 4, load_relaxed_shared(sums + 4)
                // + load * vals[i_5]);
                i_5 += row_step;
                dependency_5 = colidxs[i_5];
                done_5 = i_5 >= row_end_5;

                if (done_5) {
                    const auto r_5 = (b[row_5] - sum_5) / vals[i_5];
                    // const auto r_5 = (b[row_5] - load_relaxed_shared(sums +
                    // 4)) / vals[i_5];
                    store_relaxed(x + row_5, r_5);
                    ++done_count;
                }
            }
        }
        // after_5:

        // if(done_6){
        //     goto after_6;
        // }
        if (!done_6) {
            auto load = load_relaxed(x + dependency_6);
            if (!is_nan(load)) {
                sum_6 += load * vals[i_6];
                // store_relaxed_shared(sums + 5, load_relaxed_shared(sums + 5)
                // + load * vals[i_6]);
                i_6 += row_step;
                dependency_6 = colidxs[i_6];
                done_6 = i_6 >= row_end_6;

                if (done_6) {
                    const auto r_6 = (b[row_6] - sum_6) / vals[i_6];
                    // const auto r_6 = (b[row_6] - load_relaxed_shared(sums +
                    // 5)) / vals[i_6];
                    store_relaxed(x + row_6, r_6);
                    ++done_count;
                }
            }
        }
        // after_6:

        // if(done_7){
        //     goto after_7;
        // }
        // if(!done_7)
        // {
        //     auto load = load_relaxed(x + dependency_7);
        //     if(!is_nan(load)){
        //         sum_7 += load * vals[i_7];
        //         // store_relaxed_shared(sums + 6, load_relaxed_shared(sums +
        //         6) + load * vals[i_7]); i_7 += row_step; dependency_7 =
        //         colidxs[i_7]; done_7 = i_7 >= row_end_7;

        //         if(done_7){
        //             const auto r_7 = (b[row_1 + 6] - sum_7) / vals[i_7];
        //             // const auto r_7 = (b[row_1 + 6] -
        //             load_relaxed_shared(sums + 6)) / vals[i_7];
        //             store_relaxed(x + row_1 + 6, r_7);
        //             ++done_count;
        //         }
        //     }
        // }
        // after_7:

        // if(done_8){
        //     goto after_8;
        // }
        // if(!done_8){
        //     auto load = load_relaxed(x + dependency_8);
        //     if(!is_nan(load)){
        //         sum_8 += load * vals[i_8];
        //         // store_relaxed_shared(sums + 7, load_relaxed_shared(sums +
        //         7) + load * vals[i_8]); i_8 += row_step; dependency_8 =
        //         colidxs[i_8]; done_8 = i_8 >= row_end_8;

        //         if(done_8){
        //             const auto r_8 = (b[row_1 + 7] - sum_8) / vals[i_8];
        //             // const auto r_8 = (b[row_1 + 7] -
        //             load_relaxed_shared(sums + 7)) / vals[i_8];
        //             store_relaxed(x + row_1 + 7, r_8);
        //             ++done_count;
        //         }
        //     }
        // }
        // after_8:
    }

    // The first entry past the triangular part will be the diagonal


    // This check to ensure no infinte loops happen.
    // if (is_nan(r)) {
    //     store_relaxed_shared(x_s + self_shid, zero<ValueType>());
    //     x[row * x_stride + rhs] = zero<ValueType>();
    //     *nan_produced = true;
    // }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_pt_firstrun_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* schedule_counter, IndexType* row_recordings)
{
    const auto initial_gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto lane = threadIdx.x & 31;
    const auto warp = initial_gid / 32;

    auto gid = initial_gid;

    while (gid < nrhs * n) {
        const auto rhs = gid % nrhs;
        const auto row = is_upper ? n - 1 - gid / nrhs : gid / nrhs;

        row_recordings[gid] = initial_gid;

        // lower tri matrix: start at beginning, run forward until last entry,
        // (row_end - 1) which is the diagonal entry
        // upper tri matrix: start at last entry (row_end - 1), run backward
        // until first entry, which is the diagonal entry
        const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
        const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
        const int row_step = is_upper ? -1 : 1;

        auto sum = zero<ValueType>();
        auto i = row_begin;
        for (; i != row_end; i += row_step) {
            const auto dependency = colidxs[i];
            if (is_upper ? dependency <= row : dependency >= row) {
                break;
            }
            auto x_p = &x[dependency * x_stride + rhs];

            ValueType x = *x_p;
            while (is_nan(x)) {
                x = load_relaxed(x_p);
            }

            sum += x * vals[i];
        }

        // The first entry past the triangular part will be the diagonal
        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r = (b[row * b_stride + rhs] - sum) / diag;

        x[row * x_stride + rhs] = r;

        // This check to ensure no infinte loops happen.
        if (is_nan(r)) {
            x[row * x_stride + rhs] = zero<ValueType>();
            *nan_produced = true;
        }

        // const auto partial_warp_mask = 0xFFFF0000U * (lane > 15) +
        // 0x0000FFFFU * (lane < 16); uint32 mask; if (lane > 15){
        //    mask = __ballot_sync(0xFFFF0000U, 1);
        // }else{
        //     mask = __ballot_sync(0x0000FFFFU, 1);
        // }
        const auto mask = __ballot_sync(0xFFFFFFFFU, 1);

        const auto pop = __popc(mask);
        const auto lz = __clz(mask);


        // if (initial_gid < 32){
        //     printf("initial_gid %d (now %d) got mask %d pop %d lz %d\n",
        //     (int)initial_gid, gid, mask, pop, lz);
        // }

        if (lane == lz) {
            gid = atomic_add(schedule_counter, (IndexType)pop);
            // if(initial_gid < 32){
            //      printf("new gid %d (mask %d pop %d lz %d) at initial %d\n",
            //      (int)gid, mask, pop, lz, (int)initial_gid);
            // }
        }

        // if (lane > 15){
        //     const auto rgid = __shfl_sync(mask, gid, 31);
        //     gid = rgid + lane - 31;
        // }else{
        //     const auto rgid = __shfl_sync(mask, gid, 15);
        //     gid = rgid + lane - 15;
        // }
        const auto rgid = __shfl_sync(mask, gid, lz);
        gid = rgid + lane - lz;
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_pt_secondrun_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* schedule_counter, const IndexType* const row_mappings,
    const IndexType* const row_mapping_starts, int num_pthreads
    // ,int64* row_start_times, int64* row_end_times
)
{
    const auto initial_gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto lane = threadIdx.x & 31;
    const auto warp = initial_gid / 32;

    if(initial_gid >= n){
        return;
    }

    const auto row_mapping_start = row_mapping_starts[initial_gid];

    const auto rhs = 0;

    // TIMING
    // int64 time = clock();

    const auto row_count =
        (initial_gid == (num_pthreads < n ? (num_pthreads - 1) : (n - 1)))
            ? n - row_mapping_start
            : row_mapping_starts[initial_gid + 1] - row_mapping_start;
    for (auto row_counter = 0; row_counter < row_count; ++row_counter) {
        const auto row = row_mappings[row_mapping_start + row_counter];


        // TIMING
        // row_start_times[row] = time;


        // lower tri matrix: start at beginning, run forward until last entry,
        // (row_end - 1) which is the diagonal entry
        // upper tri matrix: start at last entry (row_end - 1), run backward
        // until first entry, which is the diagonal entry
        const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
        const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
        const int row_step = is_upper ? -1 : 1;

        auto sum = zero<ValueType>();
        auto i = row_begin;
        for (; i != row_end; i += row_step) {
            const auto dependency = colidxs[i];
            if (is_upper ? dependency <= row : dependency >= row) {
                break;
            }
            auto x_p = &x[dependency * x_stride + rhs];

            ValueType x = *x_p;
            while (is_nan(x)) {
                x = load_relaxed(x_p);
            }

            sum += x * vals[i];
        }

        // The first entry past the triangular part will be the diagonal
        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r = (b[row * b_stride + rhs] - sum) / diag;

        x[row * x_stride + rhs] = r;

        // This check to ensure no infinte loops happen.
        // if (is_nan(r)) {
        //     x[row * x_stride + rhs] = zero<ValueType>();
        //     *nan_produced = true;
        // }


        // Not needed for safety, but for perf
        // const auto mask = __ballot_sync(0xFFFFFFFFU, 1);
        __syncwarp();

        // TIMING
        // time = clock();
        // row_end_times[row] = time;
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_pt_check_wellposedness_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, ValueType* const x, const size_type n,
    bool* nan_produced,
    int32* considerably_spinning_count,  // init to 0
    uint32* wellposed,                   // init to 1
    IndexType* schedule_counter, const IndexType* const row_mappings,
    const IndexType* const row_mapping_starts)
{
    const auto initial_gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto lane = threadIdx.x & 31;
    const auto warp = initial_gid / 32;
    const auto row_mapping_start = row_mapping_starts[initial_gid];

    auto gid = initial_gid;

    const auto row_count =
        initial_gid == 20479
            ? n - row_mapping_start
            : row_mapping_starts[initial_gid + 1] - row_mapping_start;

    // DEBUG
    // if(initial_gid < 10){
    //     printf("Initial gid %d has %d row_count", (int32)initial_gid,
    //     (int32)row_count);
    // }

    for (auto row_counter = 0; row_counter < row_count; ++row_counter) {
        const auto row = row_mappings[row_mapping_start + row_counter];
        const auto rhs = 0;

        // lower tri matrix: start at beginning, run forward until last entry,
        // (row_end - 1) which is the diagonal entry
        // upper tri matrix: start at last entry (row_end - 1), run backward
        // until first entry, which is the diagonal entry
        const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
        const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
        const int row_step = is_upper ? -1 : 1;

        auto sum = zero<ValueType>();
        auto i = row_begin;
        for (; i != row_end; i += row_step) {
            const auto dependency = colidxs[i];
            if (is_upper ? dependency <= row : dependency >= row) {
                break;
            }

            auto x_p = &x[dependency];

            uint32 spincount = 0;
            bool inced = false;
            ValueType x = *x_p;
            while (is_nan(x)) {
                ++spincount;
                if (spincount == 1000) {
                    // Safe to say, if every thread has been spinning
                    // for at least 1000 iterations: we're stuck.
                    const auto spinning_threads_count =
                        atomic_add(considerably_spinning_count, 1);
                    inced = true;

                    if (spinning_threads_count >= 20479) {
                        // printf("I (%d, %d, %d) am setting non-wellposed,
                        // because I spinned 100000 times, and so did %d
                        // others\n", (int32)initial_gid, (int32)row, (int32)
                        // dependency, (int32)spinning_threads_count);
                        store_relaxed(wellposed, 0);
                    }
                }

                x = load_relaxed(x_p);

                if (!load_relaxed(wellposed)) {
                    // printf("I'm exiting because wellposed isn't set\n");
                    return;
                }
            }

            // This is theoretically not guaranteed.
            // Could be spuriosuly receiving at spin 1001, and then
            // its to late. Seems unlikely though.
            if (inced) {
                atomic_add(considerably_spinning_count, -1);
            }

            sum += x * vals[i];
        }

        // The first entry past the triangular part will be the diagonal
        const auto diag = one<ValueType>();
        const auto r = sum;

        x[row] = r;

        // This check to ensure no infinte loops happen.
        if (is_nan(r)) {
            x[row] = zero<ValueType>();
            *nan_produced = true;
        }

        __syncwarp();
    }

    atomic_add(considerably_spinning_count, 1);
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_nonpt_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;

    const auto full_gid = threadIdx.x + blockDim.x * blockIdx.x;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load_relaxed(x_p);
        }

        sum += x * vals[i];
    }

    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;

    store_relaxed_shared(x_s + self_shid, r);
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store_relaxed_shared(x_s + self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_legacy_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool unit_diag, bool* nan_produced,
    IndexType* atomic_counter)
{
    __shared__ IndexType block_base_idx;
    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * fallback_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward
    // upper tri matrix: start at last entry (row_end - 1), run backward
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto j = row_begin;
    auto col = colidxs[j];
    while (j != row_end) {
        auto x_val = load_relaxed(x + (col * x_stride + rhs));
        while (!is_nan(x_val)) {
            sum += vals[j] * x_val;
            j += row_step;
            col = colidxs[j];
            x_val = load_relaxed(x + (col * x_stride + rhs));
        }
        // to avoid the kernel hanging on matrices without diagonal,
        // we bail out if we are past the triangle, even if it's not
        // the diagonal entry. This may lead to incorrect results,
        // but prevents an infinite loop.
        if (is_upper ? row >= col : row <= col) {
            // assert(row == col);
            auto diag = unit_diag ? one<ValueType>() : vals[j];
            const auto r = (b[row * b_stride + rhs] - sum) / diag;
            store_relaxed(x + (row * x_stride + rhs), r);
            // after we encountered the diagonal, we are done
            // this also skips entries outside the triangle
            j = row_end;
            if (is_nan(r)) {
                store_relaxed(x + (row * x_stride + rhs), zero<ValueType>());
                *nan_produced = true;
            }
        }
    }
}


template <typename IndexType>
__global__ void sptrsv_init_kernel(bool* const nan_produced,
                                   IndexType* const atomic_counter)
{
    *nan_produced = false;
    *atomic_counter = IndexType{};
}

template <typename IndexType>
__global__ void sptrsv_pt_start_indices_write_kernel(
    const IndexType* const row_records, IndexType* const start_indices,
    IndexType* const start_indices_writeidx, const size_type n)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();

    if (gid >= n) {
        return;
    }

    const auto is_first = gid == 0;

    const auto is_start =
        is_first ? true : (row_records[gid - 1] < row_records[gid]);

    if (is_start) {
        start_indices[row_records[gid]] = gid;
    }
}

template <typename ValueType, typename IndexType>
struct SptrsvebcrnSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;

    int num_sms;
    int max_threads_per_sm;

    array<IndexType> row_mappings;
    array<IndexType> row_mapping_starts;

    void sync_to_gpu(std::shared_ptr<const gko::CudaExecutor> exec)
    {
        row_mappings.set_executor(exec);
        row_mapping_starts.set_executor(exec);
    }

    SptrsvebcrnSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          row_mappings{exec, matrix->get_size()[0]}
    {
        const auto is_fallback_required = exec->get_major_version() < 7;
        const auto n = matrix->get_size()[0];
        const auto nrhs = 1;

        cudaDeviceGetAttribute(&max_threads_per_sm, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, exec->get_device_id());
        num_sms = exec->get_num_multiprocessor();

        row_mapping_starts = array<IndexType>{exec, (gko::size_type)std::min(num_sms * max_threads_per_sm, (int)n) + 1};


        auto x = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        auto b = gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        dense::fill(exec, x.get(), nan<ValueType>());
        dense::fill(exec, b.get(), one<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

        array<IndexType> schedule_counter(exec, {num_sms * max_threads_per_sm});
        array<IndexType> row_recordings(exec, n);
        sptrsv_naive_pt_firstrun_kernel<false><<<num_sms, max_threads_per_sm>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), n, 1, unit_diag,
            nan_produced.get_data(), schedule_counter.get_data(),
            row_recordings.get_data());

        thrust::sequence(thrust::device, row_mappings.get_data(),
                         row_mappings.get_data() + n);
        thrust::stable_sort_by_key(thrust::device, row_recordings.get_data(),
                                   row_recordings.get_data() + n,
                                   row_mappings.get_data());

        // Write kernel to:
        // Go through this vec
        // If this is a downhill step, add the index to some global array (needs
        // sync) Then call trust to sort this global vec of indices There you
        // have the start indices of all processors, proc[i] at indices[i]
        array<IndexType> row_mapping_starts_writeidx(exec, 1);
        cudaMemset(row_mapping_starts_writeidx.get_data(), 0,
                   sizeof(IndexType));

        sptrsv_pt_start_indices_write_kernel<<<grid_size, block_size>>>(
            row_recordings.get_const_data(), row_mapping_starts.get_data(),
            row_mapping_starts_writeidx.get_data(), n);

        printf("GOT HERE 1\n");

        // thrust::sort(thrust::device, row_mapping_starts.get_data(),
        //              row_mapping_starts.get_data()  + num_sms * max_threads_per_sm);


        // HACKED WRITEOUT
        // row_mappings.set_executor(exec->get_master());
        // std::basic_ofstream<char> output;
        // output.open("delaunayn22.input.pt.p");
        // for (size_type i = 0; i < n; i++) {
        //     output << i << ',' << row_mappings.get_const_data()[i];
        //     output << '\n';
        // }
        // output.close();
        // row_mappings.set_executor(exec);


        // HACKED WRITEOUT
        // row_mapping_starts.set_executor(exec->get_master());
        // std::basic_ofstream<char> output_2;
        // output_2.open("delaunayn22.input.pt.s");
        // for (size_type i = 0; i < 20480; i++) {
        //     output_2 << i << ',' << row_mapping_starts.get_const_data()[i];
        //     output_2 << '\n';
        // }
        // output_2.close();
        // row_mapping_starts.set_executor(exec);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];


        // DEBUG
        // check_solvability(exec, matrix);


        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(ceildiv(32 * n * nrhs, block_size.x), 1, 1); // FIXME here for no-warp-sized

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsv_naive_caching_warpperrow_kernel<true, true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                // sptrsv_naive_caching_kernel<false><<<grid_size, block_size>>>(
                //     matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                //     as_cuda_type(matrix->get_const_values()),
                //     as_cuda_type(b->get_const_values()), b->get_stride(),
                //     as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                //     unit_diag, nan_produced.get_data(),
                //     atomic_counter.get_data());

                // const dim3 block_size(512, 1, 1);
                // const dim3 grid_size(ceildiv((n + 5) / 6, block_size.x), 1,
                // 1); sptrsv_naive_caching_multirow_kernel<false><<<grid_size,
                // block_size>>>(
                //     matrix->get_const_row_ptrs(),
                //     matrix->get_const_col_idxs(),
                //     as_cuda_type(matrix->get_const_values()),
                //     as_cuda_type(b->get_const_values()),
                //     as_cuda_type(x->get_values()), n, nrhs,
                //     unit_diag, nan_produced.get_data());

                // array<int64> begin_times(exec, n);
                // array<int64> end_times(exec, n);   

                array<IndexType> schedule_counter(exec, {max_threads_per_sm * num_sms});
                sptrsv_naive_pt_secondrun_kernel<false><<<num_sms, max_threads_per_sm>>>(
                    matrix->get_const_row_ptrs(),
                    matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    schedule_counter.get_data(),
                    row_mappings.get_const_data(),
                    row_mapping_starts.get_const_data(),
                    num_sms * max_threads_per_sm
                    // ,begin_times.get_data(), end_times.get_data()
                );

                // HACKED WRITEOUT
                // std::basic_ofstream<char> output;
                // output.open("delaunayn22.input.pt.xs");
                // begin_times.set_executor(exec->get_master());
                // end_times.set_executor(exec->get_master());
                // for (size_type i = 0; i < n; i++) {
                //     output << i << ',' << begin_times.get_const_data()[i] <<
                //     ',' << end_times.get_const_data()[i]; output << '\n';
                // }

                //     output.close();
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }

    bool check_solvability(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix) const
    {
        const auto n = matrix->get_size()[0];

        // Initialize x to all NaNs.
        array<ValueType> x(exec, n);
        x.fill(nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        array<int32> considerably_spinning_count(exec, 1);
        array<uint32> wellposed(exec, 1);
        cudaMemset(considerably_spinning_count.get_data(), 0, sizeof(int32));
        cudaMemset(wellposed.get_data(), 1, sizeof(uint32));

        array<IndexType> schedule_counter(exec, {20480});
        sptrsv_pt_check_wellposedness_kernel<false><<<40, 512>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(x.get_data()), n, nan_produced.get_data(),
            considerably_spinning_count.get_data(), wellposed.get_data(),
            schedule_counter.get_data(), row_mappings.get_const_data(),
            row_mapping_starts.get_const_data());

        return (bool)exec->copy_val_to_host(wellposed.get_const_data());
    }
};


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_ud_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const size_type nrhs, bool* nan_produced, IndexType* atomic_counter)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = b[row * b_stride + rhs];
    auto i = row_begin;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load_relaxed(x_p);
        }

        sum -= x * vals[i];
    }

    store_relaxed_shared(x_s + self_shid, sum);
    x[row * x_stride + rhs] = sum;

    // This check to ensure no infinte loops happen.
    if (is_nan(sum)) {
        store_relaxed_shared(x_s + self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebcrnuSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_m;


    SptrsvebcrnuSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* matrix,
                            size_type, bool is_upper)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}
    {
        scaled_m = matrix::Csr<ValueType, IndexType>::create(exec);
        scaled_m->copy_from(matrix);
        diag->inverse_apply(matrix, scaled_m.get());
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        const auto new_b = matrix::Dense<ValueType>::create(exec);
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());

        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    true, nan_produced.get_data(), atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    true, nan_produced.get_data(), atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsv_naive_caching_ud_kernel<true><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data());
            } else {
                sptrsv_naive_caching_ud_kernel<false>
                    <<<grid_size, block_size>>>(
                        scaled_m->get_const_row_ptrs(),
                        scaled_m->get_const_col_idxs(),
                        as_cuda_type(scaled_m->get_const_values()),
                        as_cuda_type(new_b->get_const_values()),
                        b->get_stride(), as_cuda_type(x->get_values()),
                        x->get_stride(), n, nrhs, nan_produced.get_data(),
                        atomic_counter.get_data());
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsmebrcnm_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b, size_type b_stride,
    ValueType* const x, size_type x_stride, const size_type n,
    const IndexType nrhs, bool* nan_produced, IndexType* atomic_counter,
    IndexType m, bool unit_diag)
{
    __shared__ IndexType block_base_idx;

    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = (full_gid / m) % nrhs;
    const auto gid = full_gid / (m * nrhs);
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n || rhs >= nrhs || full_gid % m != 0) {
        return;
    }

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_diag = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_diag; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto x_p = &x[dependency * x_stride + rhs];


        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load_relaxed(x_p);
        }

        sum += x * vals[i];
    }

    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebcrnmSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    IndexType m;
    bool unit_diag;

    SptrsvebcrnmSolveStruct(std::shared_ptr<const gko::CudaExecutor>,
                            const matrix::Csr<ValueType, IndexType>*, size_type,
                            bool is_upper, bool unit_diag, uint8 m)
        : is_upper{is_upper}, m{m}, unit_diag{unit_diag}
    {}

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(
            ceildiv(n * (is_fallback_required ? 1 : m) * nrhs, block_size.x), 1,
            1);

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    unit_diag, nan_produced.get_data(),
                    atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsmebrcnm_kernel<true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data(), m,
                    unit_diag);
            } else {
                sptrsmebrcnm_kernel<false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data(), m,
                    unit_diag);
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvelcr_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b,
    const size_type b_stride, ValueType* const x, const size_type x_stride,
    const IndexType* const levels, const IndexType sweep, const IndexType n,
    const IndexType nrhs, bool unit_diag)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid / nrhs;
    const auto rhs = gid % nrhs;

    if (row >= n) {
        return;
    }

    if (levels[row] != sweep) {
        return;
    }

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const auto row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    IndexType i = row_start;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        sum += x[dependency * x_stride + rhs] * vals[i];
    }

    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;
    x[row * x_stride + rhs] = r;
}


template <typename IndexType, bool is_upper>
__global__ void level_generation_kernel(const IndexType* const rowptrs,
                                        const IndexType* const colidxs,
                                        IndexType* const levels,
                                        IndexType* const height,
                                        const IndexType n,
                                        IndexType* const atomic_counter)
{
    __shared__ uninitialized_array<IndexType, default_block_size> level_s_array;
    // __shared__ IndexType block_base_idx;

    // if (threadIdx.x == 0) {
    //     block_base_idx =
    //         atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    // }
    // __syncthreads();
    // const auto full_gid = static_cast<IndexType>(threadIdx.x) +
    // block_base_idx;
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // const auto self_shmem_id = full_gid / default_block_size;
    // const auto self_shid = full_gid % default_block_size;

    // IndexType* level_s = level_s_array;
    // level_s[self_shid] = -1;

    // __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    IndexType level = -one<IndexType>();
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto l_p = &levels[dependency];

        // const auto dependency_gid = is_upper ? n - 1 - dependency :
        // dependency; const bool shmem_possible =
        //     (dependency_gid / default_block_size) == self_shmem_id;
        // if (shmem_possible) {
        //     const auto dependency_shid = dependency_gid % default_block_size;
        //     l_p = &level_s[dependency_shid];
        // }

        IndexType l = *l_p;
        while (l == -one<IndexType>()) {
            l = load_relaxed(l_p);
        }

        level = max(l, level);
    }

    // store(level_s, self_shid, level + 1);
    levels[row] = level + 1;

    atomic_max((IndexType*)height, level + 1);
}


template <typename IndexType, bool is_upper>
__global__ void _sptrsvrdpi_level_generation_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    IndexType* const levels, IndexType* const height, const IndexType n,
    IndexType* const atomic_counter)
{
    __shared__ uninitialized_array<IndexType, default_block_size> level_s_array;
    // __shared__ IndexType block_base_idx;

    // if (threadIdx.x == 0) {
    //     block_base_idx =
    //         atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    // }
    // __syncthreads();
    // const auto full_gid = static_cast<IndexType>(threadIdx.x) +
    // block_base_idx;
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // const auto self_shmem_id = full_gid / default_block_size;
    // const auto self_shid = full_gid % default_block_size;

    // IndexType* level_s = level_s_array;
    // level_s[self_shid] = -1;

    // __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    IndexType level = -one<IndexType>();
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto l_p = &levels[dependency];

        // const auto dependency_gid = is_upper ? n - 1 - dependency :
        // dependency; const bool shmem_possible =
        //     (dependency_gid / default_block_size) == self_shmem_id;
        // if (shmem_possible) {
        //     const auto dependency_shid = dependency_gid % default_block_size;
        //     l_p = &level_s[dependency_shid];
        // }

        IndexType l = *l_p;
        while (l == -one<IndexType>()) {
            l = load_relaxed(l_p);
        }

        level = max(l, level);
    }

    // store(level_s, self_shid, level + 1);
    levels[row] = level + 1;

    atomic_max((IndexType*)height, level + 1);
}


template <typename IndexType>
__global__ void sptrsv_level_counts_kernel(
    const IndexType* const levels, volatile IndexType* const level_counts,
    IndexType* const lperm, const IndexType n)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = gid;

    if (row >= n) {
        return;
    }

    auto level = levels[row];

    // TODO: Make this a parallel reduction from n -> #levels
    const auto i = atomic_add((IndexType*)(level_counts + level), (IndexType)1);

    lperm[row] = i;
}


template <typename IndexType>
__global__ void sptrsv_lperm_finalize_kernel(
    const IndexType* const levels, const IndexType* const level_counts,
    IndexType* const lperm, const IndexType n)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = gid;

    if (row >= n) {
        return;
    }

    lperm[row] += level_counts[levels[row]];
}


template <typename ValueType, typename IndexType>
struct SptrsvlrSolveStruct : solver::SolveStruct {
    bool is_upper;
    array<IndexType> levels;
    IndexType height;
    bool unit_diag;

    SptrsvlrSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* matrix,
                        size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          levels{exec, matrix->get_size()[0]}
    {
        const IndexType n = matrix->get_size()[0];
        cudaMemset(levels.get_data(), 0xFF, n * sizeof(IndexType));

        array<IndexType> height_d(exec, 1);
        cudaMemset(height_d.get_data(), 0, sizeof(IndexType));

        array<IndexType> atomic_counter(exec, 1);
        cudaMemset(atomic_counter.get_data(), 0, sizeof(IndexType));

        const auto block_size = default_block_size;
        const auto block_count = (n + block_size - 1) / block_size;

        if (is_upper) {
            level_generation_kernel<IndexType, true>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    levels.get_data(), height_d.get_data(), n,
                    atomic_counter.get_data());
        } else {
            level_generation_kernel<IndexType, false>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    levels.get_data(), height_d.get_data(), n,
                    atomic_counter.get_data());
        }

        height = exec->copy_val_to_host(height_d.get_const_data()) + 1;

        array<IndexType> level_counts(exec, height);
        cudaMemset(level_counts.get_data(), 0, height * sizeof(IndexType));

        array<IndexType> lperm(exec, n);

        sptrsv_level_counts_kernel<<<block_count, block_size>>>(
            levels.get_const_data(), level_counts.get_data(), lperm.get_data(),
            n);

        components::prefix_sum_nonnegative(exec, level_counts.get_data(),
                                           height);
    }

    void solve(std::shared_ptr<const CudaExecutor>,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        for (IndexType done_for = 0; done_for < height; ++done_for) {
            const dim3 block_size(default_block_size, 1, 1);
            const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

            if (is_upper) {
                sptrsvelcr_kernel<decltype(as_cuda_type(ValueType{})),
                                  IndexType, true><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(),
                    levels.get_const_data(), done_for, n, nrhs, unit_diag);
            } else {
                sptrsvelcr_kernel<decltype(as_cuda_type(ValueType{})),
                                  IndexType, false><<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    as_cuda_type(b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(),
                    levels.get_const_data(), done_for, n, nrhs, unit_diag);
            }
        }
    }
};


// Values other than 32 don't work.
constexpr int32 warp_inverse_size = 32;


template <typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_generate_prep_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    IndexType* const row_skip_counts, const size_type n)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid;

    if (row >= n) {
        return;
    }

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end =
        is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];  // Includes diagonal
    const auto row_step = is_upper ? -1 : 1;
    const auto local_inv_count =
        is_upper ? warp_inverse_size - (row % warp_inverse_size) - 1
                 : row % warp_inverse_size;

    // TODO: Evaluate end-to-start iteration with early break optimization
    // Note: This optimization is only sensible when a hint
    //       "does this use compact storage" is set to false.
    // FIXME: Document a requirement of sorted indices, then
    //        break on first hit in the diagonal box, calculating
    //        the number of not-visited entries. That is more
    //        efficient for compact storage schemes.
    IndexType row_skip_count = 0;
    for (IndexType i = row_start; i != row_end; i += row_step) {
        const auto dep = colidxs[i];

        if (is_upper) {
            // Includes diagonal, entries from the other factor evaluate to
            // negative
            if (dep - row <= local_inv_count) {
                ++row_skip_count;
            }
        } else {
            if (row - dep <= local_inv_count) {
                ++row_skip_count;
            }
        }
    }

    row_skip_counts[row] = row_skip_count;
}


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_generate_inv_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, IndexType* const row_skip_counts,
    ValueType* const band_inv,  // zero initialized
    uint32* const masks, const size_type n, const bool unit_diag)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto inv_block = gid / warp_inverse_size;
    const auto rhs = gid % warp_inverse_size;

    if (gid >= n) {
        return;
    }
    int activemask = __ballot_sync(0xFFFFFFFF, 1);

    const auto local_start_row = is_upper ? warp_inverse_size - 1 : 0;
    const auto local_end_row = is_upper ? -1 : warp_inverse_size;
    const auto local_step_row = is_upper ? -1 : 1;

#pragma unroll
    for (IndexType _i = local_start_row; _i != local_end_row;
         _i += local_step_row) {
        const auto row = (gid / warp_inverse_size) * warp_inverse_size + _i;

        // Skips entries beyond matrix size, in the last/first block
        if (row >= n) {
            continue;
        }

        // Go though all block-internal dependencies of the row

        const auto row_start = is_upper
                                   ? rowptrs[row] + row_skip_counts[row] - 1
                                   : rowptrs[row + 1] - row_skip_counts[row];
        const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
        const auto row_step = is_upper ? -1 : 1;

        auto sum = zero<ValueType>();
        IndexType i = row_start;
        for (; i != row_end; i += row_step) {
            const auto dep = colidxs[i];

            // To skip out-of-triangle entries for compressed storage
            if (dep == row) {
                break;
            }

            sum +=
                band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         dep % warp_inverse_size + rhs * warp_inverse_size] *
                vals[i];
        }

        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r =
            ((rhs == _i ? one<ValueType>() : zero<ValueType>()) - sum) / diag;

        band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                 row % warp_inverse_size + rhs * warp_inverse_size] = r;
    }

    const auto local_row = rhs;
    const auto row = gid;

    // Discover connected components.

    // Abuse masks as intermediate storage for component descriptors
    store_relaxed(masks + row, local_row);
    __syncwarp(activemask);

    for (IndexType _i = 0; _i < warp_inverse_size; ++_i) {
        uint32 current_min = local_row;

        const auto h_start = is_upper ? local_row + 1 : 0;
        const auto h_end = is_upper ? warp_inverse_size : local_row;
        const auto v_start = is_upper ? 0 : local_row + 1;
        const auto v_end = is_upper ? local_row : warp_inverse_size;

        for (IndexType i = h_start; i < h_end; ++i) {
            if (band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         local_row + i * warp_inverse_size] != 0.0) {
                const auto load1 = load_relaxed(masks + (row - local_row + i));
                if (current_min > load1) {
                    current_min = load1;
                }
            }
        }
        for (IndexType i = v_start; i < v_end; ++i) {
            if (band_inv[inv_block * (warp_inverse_size * warp_inverse_size) +
                         i + local_row * warp_inverse_size] != 0.0) {
                const auto load2 = load_relaxed(masks + (row - local_row + i));
                if (current_min > load2) {
                    current_min = load2;
                }
            }
        }

        // That was one round of fixed-point min iteration.
        store_release(masks + row, current_min);
        __syncwarp(activemask);
    }

    // Now translate that into masks.
    uint32 mask = 0b0;
    const auto component = load_relaxed(masks + row);
    for (IndexType i = 0; i < warp_inverse_size; ++i) {
        if (load_relaxed(masks + (row - local_row + i)) == component) {
            mask |= (0b1 << (is_upper ? warp_inverse_size - i - 1 : i));
        }
    }

    __syncwarp(activemask);

    masks[row] = mask;
}


template <typename ValueType, typename IndexType, bool is_upper>
__global__ void sptrsvebcrwi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const IndexType* const row_skip_counts, const ValueType* const vals,
    const ValueType* const b, const size_type b_stride, ValueType* const x,
    const size_type x_stride, const ValueType* const band_inv,
    const uint32* const masks, const size_type n, const size_type nrhs,
    bool* nan_produced)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row =
        is_upper ? ((IndexType)n + blockDim.x - 1) / blockDim.x * blockDim.x -
                       gid - 1
                 : gid;
    const auto rhs = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= n) {
        return;
    }
    if (rhs >= nrhs) {
        return;
    }

    const int self_shid = row % default_block_size;
    const auto skip_count = row_skip_counts[row];

    const auto row_start = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end =
        is_upper ? rowptrs[row] + skip_count - 1
                 : rowptrs[row + 1] - skip_count;  // no -1, as skip_count >= 1
    const auto row_step = is_upper ? -1 : 1;

    ValueType sum = 0.0;
    for (IndexType i = row_start; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        auto x_p = &x[dependency * x_stride + rhs];

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load_relaxed(x_p);
        }

        sum += x * vals[i];
    }

    __shared__ uninitialized_array<ValueType, default_block_size> b_s_array;
    ValueType* b_s = b_s_array;
    store_relaxed_shared(b_s + self_shid, b[row * b_stride + rhs] - sum);

    // Now sync all necessary threads before going into the mult.
    // Inactive threads can not have a sync bit set.
    const auto syncmask = masks[row];
    __syncwarp(syncmask);

    const auto band_inv_block =
        band_inv +
        (warp_inverse_size * warp_inverse_size) * (row / warp_inverse_size) +
        row % warp_inverse_size;
    const auto local_offset = row % warp_inverse_size;

    ValueType inv_sum = zero<ValueType>();
    for (int i = 0; i < warp_inverse_size; ++i) {
        inv_sum += band_inv_block[i * warp_inverse_size] *
                   load_relaxed(b_s + (self_shid - local_offset + i));
    }

    const auto r = inv_sum;
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebrwiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;
    array<ValueType> band_inv;
    array<IndexType> row_skip_counts;
    array<uint32> masks;

    SptrsvebrwiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          band_inv{exec, static_cast<uint64>(warp_inverse_size) *
                             static_cast<uint64>(warp_inverse_size) *
                             ceildiv(matrix->get_size()[0],
                                     static_cast<uint64>(warp_inverse_size))},
          row_skip_counts{exec, matrix->get_size()[0]},
          masks{exec, matrix->get_size()[0]}
    {
        const auto n = matrix->get_size()[0];
        const auto inv_blocks_count = ceildiv(n, warp_inverse_size);

        cudaMemset(band_inv.get_data(), 0,
                   warp_inverse_size * warp_inverse_size * inv_blocks_count *
                       sizeof(ValueType));
        cudaMemset(masks.get_data(), 0, n * sizeof(uint32));

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);

        if (is_upper) {
            sptrsvebcrwi_generate_prep_kernel<IndexType, true>
                <<<grid_size, block_size>>>(matrix->get_const_row_ptrs(),
                                            matrix->get_const_col_idxs(),
                                            row_skip_counts.get_data(), n);
            sptrsvebcrwi_generate_inv_kernel<
                decltype(as_cuda_type(ValueType{})), IndexType, true>
                <<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    row_skip_counts.get_data(),
                    as_cuda_type(band_inv.get_data()), masks.get_data(), n,
                    unit_diag);
        } else {
            sptrsvebcrwi_generate_prep_kernel<IndexType, false>
                <<<grid_size, block_size>>>(matrix->get_const_row_ptrs(),
                                            matrix->get_const_col_idxs(),
                                            row_skip_counts.get_data(), n);
            sptrsvebcrwi_generate_inv_kernel<
                decltype(as_cuda_type(ValueType{})), IndexType, false>
                <<<grid_size, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    as_cuda_type(matrix->get_const_values()),
                    row_skip_counts.get_data(),
                    as_cuda_type(band_inv.get_data()), masks.get_data(), n,
                    unit_diag);
        }
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        // TODO: Optimize for multiple rhs, by calling to a device gemm.

        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, {false});

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(n, block_size.x), nrhs, 1);
        if (is_upper) {
            sptrsvebcrwi_kernel<decltype(as_cuda_type(ValueType{})), IndexType,
                                true><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                row_skip_counts.get_const_data(),
                as_cuda_type(matrix->get_const_values()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                as_cuda_type(band_inv.get_const_data()), masks.get_const_data(),
                n, nrhs, nan_produced.get_data());
        } else {
            sptrsvebcrwi_kernel<decltype(as_cuda_type(ValueType{})), IndexType,
                                false><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                row_skip_counts.get_const_data(),
                as_cuda_type(matrix->get_const_values()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                as_cuda_type(band_inv.get_const_data()), masks.get_const_data(),
                n, nrhs, nan_produced.get_data());
        }
    }
};


template <bool is_upper, typename IndexType>
__global__ void sptrsvebcrwvs_write_vwarp_ids(
    const IndexType* const vwarp_offsets, IndexType* const vwarp_ids,
    const IndexType num_vwarps, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= num_vwarps) {
        return;
    }

    const auto vwarp_start = vwarp_offsets[gid];
    const auto vwarp_end = vwarp_offsets[gid + 1];

    for (IndexType i = vwarp_start; i < vwarp_end; ++i) {
        vwarp_ids[i] = is_upper ? n - gid - 1 : gid;
    }
}


template <bool is_upper, typename IndexType>
__global__ void sptrsvebcrwvs_generate_assigned_sizes(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const double avg_threads_per_row, IndexType* const assigned_sizes,
    IndexType* const entry_counts, const IndexType n, const IndexType nnz)
{
    const IndexType gid = thread::get_thread_id_flat();
    const auto row = is_upper ? n - gid - 1 : gid;
    const auto row_write_location = gid;
    const int32 thread = threadIdx.x;

    if (gid >= n) {
        return;
    }

    const auto diag_pos = binary_search(
        zero<IndexType>(), rowptrs[row + 1] - rowptrs[row],
        [&](IndexType i) { return *(colidxs + rowptrs[row] + i) == row; });
    const auto valid_entry_count =
        is_upper ? rowptrs[row + 1] - rowptrs[row] - diag_pos : diag_pos + 1;
    entry_counts[row] = valid_entry_count;

    const double avg_nnz = (double)nnz / n;
    const double perfect_size =
        (valid_entry_count)*avg_threads_per_row / avg_nnz;
    const IndexType assigned_size = std::max(
        std::min((IndexType)__double2int_rn(perfect_size), (IndexType)32),
        (IndexType)1);

    volatile __shared__ int32 block_size_assigner[1];
    volatile __shared__ int32 block_size_assigner_lock[1];

    *block_size_assigner = 0;
    *block_size_assigner_lock = -1;

    __syncthreads();

    while (*block_size_assigner_lock != thread - 1) {
    }

    const auto prev_offset = *block_size_assigner;
    *block_size_assigner += assigned_size;

    int32 shrinked_size = 0;
    if ((prev_offset + assigned_size) / 32 > prev_offset / 32) {
        shrinked_size = ((prev_offset + assigned_size) / 32) * 32 - prev_offset;
        *block_size_assigner += shrinked_size - assigned_size;
    }

    __threadfence();
    *block_size_assigner_lock = thread;

    assigned_sizes[row_write_location] =
        shrinked_size == 0 ? assigned_size : shrinked_size;

    // This part to ensure each assigner block starts on 32*k, meaning the cuts
    // are well-placed.
    if (thread == default_block_size - 1) {
        if ((prev_offset +
             (shrinked_size == 0 ? assigned_size : shrinked_size)) %
                32 !=
            0) {
            assigned_sizes[row_write_location] =
                (prev_offset / 32 + 1) * 32 - prev_offset;
        }
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsvebcrwvs_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const IndexType* const vwarp_ids,
    const IndexType* const vwarp_offsets, const IndexType* const entry_counts,
    const ValueType* const b, const size_type b_stride, ValueType* const x,
    const size_type x_stride, bool* const nan_produced,
    const IndexType num_vthreads, const IndexType n, const IndexType nrhs,
    const bool unit_diag)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto thread = threadIdx.x;
    const auto rhs = blockIdx.y * blockDim.y + threadIdx.y;

    if (gid >= num_vthreads) {
        return;
    }
    if (rhs >= nrhs) {
        return;
    }

    const auto vwarp = vwarp_ids[gid];
    const auto vwarp_start = vwarp_offsets[is_upper ? n - vwarp - 1 : vwarp];
    const auto vwarp_end = vwarp_offsets[is_upper ? n - vwarp : vwarp + 1];
    const auto vwarp_size = vwarp_end - vwarp_start;
    const auto row = vwarp;
    const IndexType vthread = gid - vwarp_start;

    if (row >= n) {
        return;
    }

    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const IndexType row_step = is_upper ? -1 : 1;

    const auto valid_entry_count = entry_counts[row];
    const auto start_offset = (valid_entry_count - 1) % vwarp_size;

    auto sum = zero<ValueType>();
    // i is adjusted for vthread 0 to hit the diagonal
    IndexType i =
        unit_diag
            ? row_begin + row_step * vthread
            : row_begin +
                  ((row_step * vthread + row_step * start_offset) % vwarp_size);
    for (; (is_upper && i > row_end) || (!is_upper && i < row_end);
         i += row_step * vwarp_size) {
        const auto dependency = colidxs[i];

        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        volatile auto x_p = &x[x_stride * dependency + rhs];

        auto l = *x_p;
        while (is_nan(l)) {
            l = load_relaxed(x_p);
        }


        sum += l * vals[i];
    }

    uint32 syncmask = ((1 << vwarp_size) - 1) << (vwarp_start & 31);

    ValueType total = sum;
    for (int offset = 1; offset < vwarp_size; ++offset) {
        auto a = real(sum);
        const auto received_a = __shfl_down_sync(syncmask, a, offset);
        const auto should_add = (syncmask >> ((thread & 31) + offset)) & 1 == 1;
        total += should_add * received_a;
        if (gko::is_complex<ValueType>()) {
            auto b = imag(sum);
            const auto received_b = __shfl_down_sync(syncmask, b, offset);
            auto ptotal =
                (thrust::complex<gko::remove_complex<ValueType>>*)&total;
            *ptotal += should_add * received_b *
                       (thrust::complex<gko::remove_complex<ValueType>>)
                           unit_root<ValueType>(4);
        }
    }

    if (vthread == 0) {
        const auto diag = unit_diag ? one<ValueType>() : vals[i];
        const auto r = (b[row * b_stride + rhs] - total) / diag;
        x[row * x_stride + rhs] = r;

        // This check to ensure no infinte loops happen.
        if (is_nan(r)) {
            x[row * x_stride + rhs] = zero<ValueType>();
            *nan_produced = true;
        }
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebrwvSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;
    IndexType vthread_count;
    array<IndexType> vwarp_ids;
    array<IndexType> vwarp_offsets;
    array<IndexType> entry_counts;

    SptrsvebrwvSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          vwarp_offsets{exec, matrix->get_size()[0] + 1},
          entry_counts{exec, matrix->get_size()[0]},
          vwarp_ids{exec}
    {
        const auto desired_avg_threads_per_row = 1.0;

        const IndexType n = matrix->get_size()[0];
        const IndexType nnz = matrix->get_num_stored_elements();

        array<IndexType> assigned_sizes(exec, n);

        const auto block_size = default_block_size;
        const auto block_count = (n + block_size - 1) / block_size;

        if (is_upper) {
            sptrsvebcrwvs_generate_assigned_sizes<true>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    desired_avg_threads_per_row, assigned_sizes.get_data(),
                    entry_counts.get_data(), n, nnz);
        } else {
            sptrsvebcrwvs_generate_assigned_sizes<false>
                <<<block_count, block_size>>>(
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    desired_avg_threads_per_row, assigned_sizes.get_data(),
                    entry_counts.get_data(), n, nnz);
        }

        cudaMemcpy(vwarp_offsets.get_data(), assigned_sizes.get_const_data(),
                   n * sizeof(IndexType), cudaMemcpyDeviceToDevice);
        components::prefix_sum_nonnegative(exec, vwarp_offsets.get_data(),
                                           n + 1);

        cudaMemcpy(&vthread_count, vwarp_offsets.get_const_data() + n,
                   sizeof(IndexType), cudaMemcpyDeviceToHost);

        vwarp_ids.resize_and_reset(vthread_count);
        const auto block_size_vwarped = default_block_size;
        const auto block_count_vwarped =
            (n + block_size_vwarped - 1) / block_size_vwarped;
        if (is_upper) {
            sptrsvebcrwvs_write_vwarp_ids<true>
                <<<block_count_vwarped, block_size_vwarped>>>(
                    vwarp_offsets.get_const_data(), vwarp_ids.get_data(), n, n);
        } else {
            sptrsvebcrwvs_write_vwarp_ids<false>
                <<<block_count_vwarped, block_size_vwarped>>>(
                    vwarp_offsets.get_const_data(), vwarp_ids.get_data(), n, n);
        }
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        // TODO: Optimize for multiple rhs.

        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, {false});

        const dim3 block_size(default_block_size, 1024 / default_block_size, 1);
        const dim3 grid_size(ceildiv(vthread_count, block_size.x),
                             ceildiv(nrhs, block_size.y), 1);

        if (is_upper) {
            sptrsvebcrwvs_kernel<true><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_cuda_type(matrix->get_const_values()),
                vwarp_ids.get_const_data(), vwarp_offsets.get_const_data(),
                entry_counts.get_const_data(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                nan_produced.get_data(), vthread_count, n, nrhs, unit_diag);
        } else {
            sptrsvebcrwvs_kernel<false><<<grid_size, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                as_cuda_type(matrix->get_const_values()),
                vwarp_ids.get_const_data(), vwarp_offsets.get_const_data(),
                entry_counts.get_const_data(),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(x->get_values()), x->get_stride(),
                nan_produced.get_data(), vthread_count, n, nrhs, unit_diag);
        }

#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


template <typename ValueType, typename IndexType>
__global__ void partial_inversion_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const IndexType* const inversion_ids,
    const IndexType* const inversion_idxs,  // 2 entrys for each id, start and
                                            // stop, end is exclusive
    ValueType* const new_deps,  // n spots for each inversion_id, already filled
                                // (later optimze to dynamic slab/arena alloc)
    ValueType* const
        new_b_coeffs,  // the accompanying b coeffs, initialized to
                       // all zeros. Diagonal handled implicitly. Colmaj.
    const IndexType inversion_count, const IndexType n, const IndexType nnz)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    // First draft only parallelizes over inversion_ids
    // TODO: Use (multiple) warps for each inversion_id

    // One idea:
    // Have each warp work a "sector" of the row, marks sectors as diretied
    // (meaning in need of another pass) when a coeff is inserted there

    // Other idea: Have a warp work in lockstep, shuffling intra-warp results to
    // each other

    // Other idea: Something with odd-even reductions

    if (gid >= inversion_count) {
        return;
    }

    const auto inv_row = inversion_ids[gid];
    const auto inv_start = inversion_idxs[2 * gid];
    const auto inv_end = inversion_idxs[2 * gid + 1];
    const auto row_start = rowptrs[inv_row];
    const auto row_end = rowptrs[inv_row + 1] - 1;

    // DEBUG METRIC
    IndexType nonzero_count = 0;
    IndexType reduction_ops = 0;

    // TODO
    // Optimize by starting at the first nonzero
    const auto start = min(inv_row - 1, inv_end - 1);  // max for upper, i guess

    // DEBUG
    // printf("Row %d (gid %d) reducing from idx %d down to %d\n",
    // (int32)inv_row, (int32)gid, (int32)start, (int32)inv_start);

    // Diagonal / own b
    new_b_coeffs[inv_row * inversion_count + gid] = -one<ValueType>();

    for (IndexType i = start; i >= inv_start; --i) {
        // Reduce the coefficeint at inv_row,i
        // Carrying all transitive deps into earlier slots

        const auto c = new_deps[gid * n + i];

        if (c != zero<ValueType>()) {
            // DEBUG METRIC
            nonzero_count++;

            const auto selected_row_start = rowptrs[i];
            const auto selected_row_end = rowptrs[i + 1] - 1;
            const auto d = vals[rowptrs[i + 1] - 1];
            const auto f = c / d;

            for (IndexType j = selected_row_start; j < selected_row_end; ++j) {
                const auto trans_dep = colidxs[j];
                const auto trans_c = vals[j];
                new_deps[gid * n + trans_dep] -= f * trans_c;

                // DEBUG METRIC
                reduction_ops++;

                // DEBUG
                // if(gid == 1){
                //     printf("Reducing invrow %d colidx %d into %d\n",
                //     (int32)gid, (int32)i, (int32)trans_dep);
                // }

                // TODO
                // Optimize by remembering the min ever reduced to, early
                // exiting if that is reached
            }

            // DEBUG
            // if(gid == 1){
            //     printf("Reducing invrow %d wrote bcoff %lf to %d * %d +
            //     %d\n", (int32)gid, c, (int32)gid, (int32)n, (int32)i);
            // }

            new_b_coeffs[i * inversion_count + gid] = f;
            new_deps[gid * n + i] = zero<ValueType>();
        }
    }

    // printf("Row %d had to do %d nonzero reductions\n", (int32)inv_row,
    // (int32)nonzero_count); printf("Row %d had a total of %d reduction
    // operations\n", (int32)inv_row, (int32)reduction_ops);
}


template <typename ValueType, typename IndexType>
__global__ void copy_partial_inversion_product_kernel(
    const IndexType* const rowptrs, const ValueType* const vals,
    const IndexType* const inversion_ids, const ValueType* const tmp_x,
    ValueType* const x, const IndexType inversion_count)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();

    if (gid >= inversion_count) {
        return;
    }

    // DEBUG
    // if (inversion_ids == NULL){
    //     printf("inversion_ids is null\n");
    //     return;
    // }
    // if (tmp_x == NULL){
    //     printf("tmpx is null\n");
    //     return;
    // }
    // if (x == NULL){
    //     printf("x is null\n");
    //     return;
    // }

    const auto row = inversion_ids[gid];
    const auto d = vals[rowptrs[row + 1] - 1];
    x[row] = -tmp_x[gid] / d;

    // DEBUG
    // printf("Scatterwrote %lf to x at %d\n", x[row], (int32)row);
}


template <typename ValueType, typename IndexType>
void partial_inversion_product(std::shared_ptr<const CudaExecutor> exec,
                               const matrix::Csr<ValueType, IndexType>* matrix,
                               const matrix::Dense<ValueType>* b,
                               matrix::Dense<ValueType>* x,
                               const ValueType* const new_b_coeffs,
                               const IndexType* const inv_ids,
                               const IndexType inv_count, IndexType n)
{
    // const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    // if (gid >= n * inversion_count) {
    //     return;
    // }

    // const auto inversion_row = gid / n;
    // const auto colidx = gid % n;

    // // TODO: Accumulate local product in shared memory

    // atomic_add(x + inversion_row, new_deps[n * inversion_row + colidx] *
    // x[colidx]);

    array<ValueType> tmp_x(exec, inv_count);
    tmp_x.fill(zero<ValueType>());

    auto handle = exec->get_cublas_handle();
    cublas::pointer_mode_guard pm_guard(handle);
    auto alpha = one<ValueType>();
    auto beta = zero<ValueType>();
    cublas::gemv(handle, cublasOperation_t::CUBLAS_OP_N, (int)inv_count, (int)n,
                 &alpha, new_b_coeffs, (int)inv_count, b->get_const_values(), 1,
                 &beta, tmp_x.get_data(), 1);

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(inv_count, default_block_size), 1,
                         1);  // will usually be (1, 1, 1)
    copy_partial_inversion_product_kernel<<<grid_size, block_size>>>(
        matrix->get_const_row_ptrs(), as_cuda_type(matrix->get_const_values()),
        inv_ids, as_cuda_type(tmp_x.get_const_data()),
        as_cuda_type(x->get_values()), inv_count);
}


template <typename ValueType, typename IndexType>
__global__ void dense_dep_gen_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const IndexType* const inversion_ids,
    const IndexType* const inversion_idxs, ValueType* const new_deps,
    const IndexType inversion_count, IndexType n, const IndexType nnz,
    IndexType threadcount)
{
    const auto tgid = blockIdx.x * blockDim.x + threadIdx.x;

    for (auto gid = tgid;; gid += threadcount) {
        IndexType right = 0;
        IndexType left = 0;
        IndexType offset = -1;
        IndexType inv_row = 0;
        IndexType inv_id = 0;
        for (; inv_id < inversion_count; ++inv_id) {
            inv_row = inversion_ids[inv_id];
            const auto c = rowptrs[inv_row + 1] - rowptrs[inv_row];
            right += c;

            if (left <= gid && gid < right) {
                offset = gid - left;
                break;
            } else {
                left += c;
            }
        }

        if (offset == -1) {
            return;
        }

        // DEBUG
        // if (rowptrs[inv_row] + offset >= nnz){
        //     printf("M1: Illegal memory access about to happen (%d)\n",
        //     (int32)rowptrs[inv_id] + offset);
        // }

        const auto colidx = colidxs[rowptrs[inv_row] + offset];
        const auto val = vals[rowptrs[inv_row] + offset];

        //
        new_deps[n * inv_id + colidx] = val;

        // DEBUG
        // printf("Writing to row %d ()inv_id: %d) at idx %d val: %lf (offset:
        // %d (%d-%d), tgid: %d gid: %d)\n", (int32)inv_row, (int32)inv_id,
        // (int32)colidx, val, (int32)offset, (int32)left, (int32)right,
        // (int32)tgid, (int32)gid);
    }
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void pdi_solve_kernel(const IndexType* const rowptrs,
                                 const IndexType* const colidxs,
                                 const ValueType* const vals,
                                 const ValueType* const b, size_type b_stride,
                                 ValueType* const x, size_type x_stride,
                                 const size_type n, const size_type nrhs,
                                 bool unit_diag, bool* nan_produced)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;

    const auto full_gid = thread::get_thread_id_flat<IndexType>();
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }


    const auto self_shmem_id = full_gid / default_block_size;
    const auto self_shid = full_gid % default_block_size;

    ValueType* x_s = x_s_array;

    // In that case pdi already did that row
    if (!is_nan(x[row])) {
        x_s[self_shid] = x[row];
        return;
    }
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = zero<ValueType>();
    auto i = row_begin;
    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load_relaxed(x_p);
        }

        sum += x * vals[i];
    }

    // The first entry past the triangular part will be the diagonal
    const auto diag = unit_diag ? one<ValueType>() : vals[i];
    const auto r = (b[row * b_stride + rhs] - sum) / diag;

    store_relaxed_shared(x_s + self_shid, r);
    x[row * x_stride + rhs] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store_relaxed_shared(x_s + self_shid, zero<ValueType>());
        x[row * x_stride + rhs] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
struct SptrsvebrpdiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;
    bool unit_diag;

    // n spots for each inversion_id, already filled
    // (later optimze to dynamic slab/arena alloc)
    array<ValueType> new_deps;

    // the accompanying b coeffs, initialized to
    // all zeros. Diagonal handled implicitly.
    array<ValueType> new_b_coeffs;

    IndexType inv_count;

    array<IndexType> inversion_ids;
    array<IndexType> inv_boundaries;

    SptrsvebrpdiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* matrix,
                            size_type, bool is_upper, bool unit_diag,
                            IndexType inv_count)
        : is_upper{is_upper},
          unit_diag{unit_diag},
          inv_count{inv_count},
          new_deps{exec, inv_count * matrix->get_size()[0]},
          new_b_coeffs{exec, inv_count * matrix->get_size()[0]},
          inversion_ids{exec->get_master(), inv_count},
          inv_boundaries{exec->get_master(), 2 * inv_count}
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nnz = matrix->get_num_stored_elements();


        for (IndexType i = 0; i < inv_count; ++i) {
            const auto v = std::min(std::max((n * i) / inv_count, (IndexType)1),
                                    (IndexType)n - 1);
            inversion_ids.get_data()[i] = v;
        }
        inversion_ids.set_executor(exec);


        for (auto i = 0; i < inv_count; ++i) {
            inv_boundaries.get_data()[2 * i] = 0;
            inv_boundaries.get_data()[2 * i + 1] = n;
        }
        inv_boundaries.set_executor(exec);


        new_deps.fill(zero<ValueType>());
        new_b_coeffs.fill(zero<ValueType>());

        const dim3 block_size_1(default_block_size, 1, 1);
        // const dim3 grid_size_1(ceildiv((matrix->get_num_stored_elements() *
        // inv_count) / n, default_block_size), 1, 1);
        const dim3 grid_size_1(ceildiv(inv_count, default_block_size), 1, 1);

        dense_dep_gen_kernel<<<grid_size_1, block_size_1>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            inversion_ids.get_const_data(), inv_boundaries.get_const_data(),
            as_cuda_type(new_deps.get_data()), inv_count, n, nnz,
            (IndexType)(default_block_size *
                        ceildiv(
                            (matrix->get_num_stored_elements() * inv_count) / n,
                            default_block_size)));


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(inv_count, default_block_size), 1,
                               1);  // will usually be (1, 1, 1)

        partial_inversion_kernel<<<grid_size_2, block_size_2>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            inversion_ids.get_const_data(), inv_boundaries.get_const_data(),
            as_cuda_type(new_deps.get_data()),
            as_cuda_type(new_b_coeffs.get_data()), inv_count, n, nnz);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const IndexType n = matrix->get_size()[0];
        const IndexType nrhs = b->get_size()[1];

        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, {false});

        // TODO: FIXME: This cant be done this way,
        // x must be all nan, have the  pre-accumulated b sums in a sperate
        // vector, add that to the internal sum at the start of each row

        partial_inversion_product(exec, matrix, b, x,
                                  new_b_coeffs.get_const_data(),
                                  inversion_ids.get_const_data(), inv_count, n);

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(n, default_block_size), nrhs, 1);

        pdi_solve_kernel<false><<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(x->get_values()), x->get_stride(), n, nrhs, unit_diag,
            nan_produced.get_data());

#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


// In COO format
//
//
// Load nnz_map[gid] to find what nnz needs to be handled by this gid
//
// Each operation has one (row-muls) or many (diags) locks to wait for
// Perform the operation
// Afer each operation, note that operation as 'done' in global mem, at
// lock[nnz]
//
// row-muls need to wait for the respective col, diags for their entire row
// alternative: make each row-mul wait for the preceding one from that row
//
// alternative to atomic_add after each row-mul: assign up to about 8 row-muls
// of the same row to the same row be careful to have these at a schedule
// position that matches with the worst individual position


template <typename ValueType, typename IndexType>
struct BlockedSolveStruct : solver::SolveStruct {
    struct pos_size_depth {
        std::pair<IndexType, IndexType> pos;
        std::pair<IndexType, IndexType> size;
        IndexType depth;

        pos_size_depth left_child(IndexType max_depth) const
        {
            if (depth == max_depth - 1) {  // Check if triangle
                return pos_size_depth{
                    std::make_pair(pos.first - size.second, pos.second),
                    std::make_pair(size.second, size.second), max_depth};
            } else {
                return pos_size_depth{
                    std::make_pair(pos.first - size.first / 2, pos.second),
                    std::make_pair(size.first / 2,
                                   size.second - size.first / 2),
                    depth + 1};
            }
        }

        pos_size_depth right_child(IndexType max_depth) const
        {
            if (depth == max_depth - 1) {
                return pos_size_depth{
                    std::make_pair(pos.first, pos.second + size.second),
                    std::make_pair(size.first, size.first), max_depth};
            } else {
                return pos_size_depth{
                    std::make_pair(pos.first + size.first / 2,
                                   pos.second + size.second),
                    std::make_pair(
                        ceildiv(size.first, 2),
                        (pos.first + size.first - (pos.second + size.second)) /
                            2),
                    depth + 1};
            }
        }
    };

    std::vector<std::shared_ptr<solver::SolveStruct>> solvers;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> blocks;
    std::vector<pos_size_depth> block_coords;

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        auto mb = matrix::Dense<ValueType>::create(exec);
        mb->copy_from(b);

        const auto block_count = blocks.size();
        for (IndexType i = 0; i < block_count; ++i) {
            if (i % 2 == 0) {
                const auto bv =
                    mb->create_submatrix(span{block_coords[i].pos.second,
                                              block_coords[i].pos.second +
                                                  block_coords[i].size.second},
                                         span{0, 1});
                auto xv =
                    x->create_submatrix(span{block_coords[i].pos.first,
                                             block_coords[i].pos.first +
                                                 block_coords[i].size.first},
                                        span{0, 1});

                solvers[i / 2]->solve(exec, blocks[i].get(), bv.get(), xv.get(),
                                      bv.get(), xv.get());
            } else {
                const auto xv =
                    x->create_submatrix(span{block_coords[i].pos.second,
                                             block_coords[i].pos.second +
                                                 block_coords[i].size.second},
                                        span{0, 1});
                auto bv =
                    mb->create_submatrix(span{block_coords[i].pos.first,
                                              block_coords[i].pos.first +
                                                  block_coords[i].size.first},
                                         span{0, 1});
                auto neg_one =
                    gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
                auto one =
                    gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
                blocks[i]->apply(neg_one.get(), xv.get(), one.get(), bv.get());
            }
        }
    }


    BlockedSolveStruct(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const gko::size_type num_rhs, bool is_upper, bool unit_diag,
        std::shared_ptr<
            std::vector<std::shared_ptr<gko::solver::trisolve_strategy>>>
            solver_ids)
    {
        const auto host_exec = exec->get_master();
        const auto n = matrix->get_size()[0];
        const auto sptrsv_count = solver_ids->size();
        const auto block_count = 2 * sptrsv_count - 1;
        const auto depth = get_significant_bit(sptrsv_count);

        // Generate the block sizes and positions.
        array<pos_size_depth> blocks(host_exec, block_count);
        pos_size_depth* blocksp = blocks.get_data();
        blocksp[0] = pos_size_depth{std::make_pair(n / 2, 0),
                                    std::make_pair(ceildiv(n, 2), n / 2), 0};
        IndexType write = 1;
        for (IndexType read = 0; write < block_count; ++read) {
            const auto cur = blocksp[read];
            blocksp[write++] = cur.left_child(depth);
            blocksp[write++] = cur.right_child(depth);
        }

        // Generate a permutation to execution order
        array<IndexType> perm(host_exec, block_count);
        IndexType* permp = perm.get_data();
        for (IndexType i = 0; i <= depth; ++i) {
            const auto step = 2 << i;
            const auto start = (1 << i) - 1;
            const auto add = sptrsv_count / (1 << i) - 1;

            for (IndexType j = start; j < block_count; j += step) {
                permp[j] = (j - start) / step + add;
            }
        }

        // Apply the perm
        // For upper_trs, we also need to reflect the cuts
        for (IndexType i = 0; i < block_count; ++i) {
            auto block = blocksp[permp[i]];

            if (is_upper) {
                std::swap(block.pos.first, block.pos.second);
                std::swap(block.size.first, block.size.second);
            }

            block_coords.push_back(block);
            this->blocks.push_back(std::move(matrix->create_submatrix(
                span{block.pos.first, block.pos.first + block.size.first},
                span{block.pos.second, block.pos.second + block.size.second})));
            this->blocks[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }

        if (is_upper) {
            for (auto i = 0; i < block_count / 2; ++i) {
                std::swap(block_coords[i], block_coords[block_count - i - 1]);
                std::swap(this->blocks[i], this->blocks[block_count - i - 1]);
            }
        }

        // Finally create the appropriate solvers
        for (IndexType i = 0; i < sptrsv_count; ++i) {
            this->solvers.push_back(std::make_shared<solver::SolveStruct>());
            solver::SolveStruct::generate(
                exec, this->blocks[2 * i].get(), this->solvers[i], num_rhs,
                solver_ids.get()->at(i).get(), is_upper, unit_diag);
        }
    }
};


// EXPERIMENTAL BELOW


__device__ unsigned get_smid(void)
{
    unsigned ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsvebcrnuts_time_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const vals, const ValueType* const b,
    const gko::size_type b_stride, ValueType* const x,
    const gko::size_type x_stride, const gko::size_type n,
    const gko::size_type nrhs, unsigned long long int* const sm_baselines,
    int32* begin_sms, int32* end_sms, int64* begin_times,
    int64* poll_start_times, int64* end_times, int64* loop_times,
    int64* loop_spin_counts)
{
    __shared__ uninitialized_array<ValueType, default_block_size> x_s_array;

    const auto unthinned_full_gid = (blockIdx.x * blockDim.x + threadIdx.x);

    const auto full_gid = unthinned_full_gid / 8;

    if (unthinned_full_gid % 8 != 0) {
        return;
    }

    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // initialize per-SM baseline
    if (threadIdx.x == 0)
        atomicCAS(sm_baselines + get_smid(), (unsigned long long int)0,
                  (unsigned long long int)clock());

    // storing timers
    begin_sms[row] = get_smid();
    begin_times[row] = clock();

    const auto self_shmem_id = gid / default_block_size;
    const auto self_shid = gid % default_block_size;

    ValueType* x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();


    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;

    auto sum = b[row * b_stride + rhs];
    auto i = row_begin;

    poll_start_times[row] = clock();

    for (; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }
        auto x_p = &x[dependency * x_stride + rhs];

        const auto dependency_gid = is_upper ? (n - 1 - dependency) * nrhs + rhs
                                             : dependency * nrhs + rhs;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency_gid % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        int64 spins = 0;

        ValueType x = *x_p;
        while (is_nan(x)) {
            // membar_seqcst();
            x = load_relaxed(x_p);
            // membar_seqcst();
            // x = load_relaxed(x_p);
            ++spins;
        }

        sum -= x * vals[i];

        loop_times[i] = clock();
        loop_spin_counts[i] = spins;
    }

    // membar_seqcst();
    // store_release_shared(x_s + self_shid, sum);
    // store_release(x + row, sum);
    // membar_seqcst();
    store_relaxed_shared(x_s + self_shid, sum);
    store_relaxed(x + row, sum);
    // store_relaxed_shared(x_s + self_shid, sum);
    // store_relaxed(x + row, sum);


    // INVESTIGATION
    // First poke at memory location.
    // store_release(x + row, load_acquire(x + row));

    __threadfence();  // Time when that write is propagated to memory

    // storing timers
    end_times[row] = clock();
    end_sms[row] = get_smid();
}


template <bool is_upper, typename IndexType>
__global__ void sptrsvebcrnuts_order_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const gko::int64 alpha, const gko::int64 beta, const gko::int64 gamma,
    IndexType* const n_betas,  // Initialized to zeros
    const gko::size_type n, const gko::int64* const start_poll_times,
    const gko::int64* const end_times, volatile gko::int64* const est_new_times)
{
    const auto gid = (blockIdx.x * blockDim.x + threadIdx.x);
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    const auto self_shmem_id = gid / default_block_size;
    const auto self_shid = gid % default_block_size;

    const auto start = start_poll_times[row];

    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] - 1 : rowptrs[row + 1];
    const int row_step = is_upper ? -1 : 1;
    const auto row_length = (row_end - row_begin) / row_step;

    // First loop, compute n_betas
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        const auto dependency_gid =
            is_upper ? (n - 1 - dependency) : dependency;
        const bool shmem_possible =
            (dependency_gid / default_block_size) == self_shmem_id;
        const bool is_beta_suitable = shmem_possible;

        if (is_beta_suitable) {
            for (auto j = row_begin; j != i; j += row_step) {
                n_betas[j] += 1;
            }
            n_betas[i] += 1;
        }
    }

    // Second loop, compute estimated new timings with perfect reordering
    // If the new ordering does not get a better estimate than the old one,
    // don't change it, and carry over the runtime
    gko::int64 max_runtime = 0;
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        auto est_new_dep_time_p = est_new_times + dependency;

        gko::int64 est_new_dep_time = *est_new_dep_time_p;
        while (est_new_dep_time == -1) {
            est_new_dep_time = load(est_new_dep_time_p, 0);
        }

        max_runtime =
            max(max_runtime, (est_new_dep_time - start) + n_betas[i] * beta +
                                 (row_length - n_betas[i] - i) * alpha);
    }

    est_new_times[row] = start + max_runtime + gamma;
}


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsvebcrnuts_reorder_front_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    IndexType* const rcolidxs, ValueType* const rvals,
    IndexType* const n_betas,  // Initialized to zeros
    const gko::size_type n, const gko::size_type front_size,
    const gko::size_type front_id, const gko::int64* const start_times,
    const gko::int64* const end_times)
{
    const auto gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= front_size) {
        return;
    }

    const IndexType row = is_upper ? n - 1 - (gid + front_id * front_size)
                                   : gid + front_id * front_size;

    if (row >= n || row < 0) {
        return;
    }

    // Compute not predicted runtimes, but predicted deltas due to one
    // rearranging by new best order sequentially (sptrsv-fashion) within each
    // block, each block after the previous one is done in between, after each
    // front, start times are remeasured (maybe other re-measuring intervals
    // make sense?) (maybe re-measuring and recomputing of arrangments for one
    // front block makes sense, if worsening of perf occurs?)
}


// ts for "unitdiag (inner) timing-scheduled"
template <typename ValueType, typename IndexType>
struct SptrsvebcrnutsSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_m;


    SptrsvebcrnutsSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                              const matrix::Csr<ValueType, IndexType>* matrix,
                              size_type, bool is_upper)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}
    {
        scaled_m = matrix::Csr<ValueType, IndexType>::create(exec);
        scaled_m->copy_from(matrix);
        diag->inverse_apply(matrix, scaled_m.get());


        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();

        auto fakex = gko::matrix::Dense<ValueType>::create(exec, dim<2>(n, 1));
        fakex->fill(nan<ValueType>());
        auto fakeb = gko::matrix::Dense<ValueType>::create(exec, dim<2>(n, 1));
        fakex->fill(one<ValueType>());

        // setup
        array<unsigned long long int> sm_baselines(
            exec, 1000);  // I'm too lazy to query the number of SMs
        array<int> element_begin_sms(exec, n);
        array<int> element_end_sms(exec, n);
        array<int64> element_begin(exec, n);
        array<int64> element_poll_start(exec, n);
        array<int64> element_end(exec, n);
        array<int64> element_loop(exec, nz);
        array<int64> element_spins(exec, nz);

        cudaMemset(sm_baselines.get_data(), 0,
                   1000 * sizeof(unsigned long long int));

        const auto block_size = default_block_size;
        const auto block_count = (8 * n + block_size - 1) / block_size;

        sptrsvebcrnuts_time_kernel<false><<<block_count, block_size>>>(
            scaled_m->get_const_row_ptrs(), scaled_m->get_const_col_idxs(),
            as_cuda_type(scaled_m->get_const_values()),
            as_cuda_type(fakeb->get_const_values()), fakeb->get_stride(),
            as_cuda_type(fakex->get_values()), fakex->get_stride(), n, nrhs,
            sm_baselines.get_data(), element_begin_sms.get_data(),
            element_end_sms.get_data(), element_begin.get_data(),
            element_poll_start.get_data(), element_end.get_data(),
            element_loop.get_data(), element_spins.get_data());


        // output timings
        sm_baselines.set_executor(exec->get_master());
        element_begin_sms.set_executor(exec->get_master());
        element_end_sms.set_executor(exec->get_master());
        element_begin.set_executor(exec->get_master());
        element_end.set_executor(exec->get_master());
        element_loop.set_executor(exec->get_master());
        element_spins.set_executor(exec->get_master());
        Array<int64> finish_times(exec->get_master(), n);

        for (size_type i = 0; i < n; i++) {
            auto begin_rel = element_begin.get_data()[i];
            auto begin_anch =
                sm_baselines.get_data()[element_begin_sms.get_data()[i]];
            auto begin =
                begin_anch <= begin_rel
                    ? begin_rel - begin_anch
                    : (std::numeric_limits<decltype(begin_anch)>::max() -
                       begin_anch) +
                          begin_rel;
            auto end_rel = element_end.get_data()[i];
            auto end_anch =
                sm_baselines.get_data()[element_end_sms.get_data()[i]];
            auto end = end_anch <= end_rel
                           ? end_rel - end_anch
                           : (std::numeric_limits<decltype(end_anch)>::max() -
                              end_anch) +
                                 end_rel;

            finish_times.get_data()[i] = end;
        }


        // JUST SOME HACKED TOGETHER STUFF
        std::basic_ofstream<char> output;
        output.open("pkustk11.input.timings");
        const auto rowptrs = scaled_m->get_const_row_ptrs();
        array<IndexType> rowptrsh(exec->get_master(), n + 1);
        exec->get_master()->copy_from(gko::as<const gko::Executor>(exec.get()),
                                      n + 1, rowptrs, rowptrsh.get_data());
        sm_baselines.set_executor(exec->get_master());
        element_begin_sms.set_executor(exec->get_master());
        element_end_sms.set_executor(exec->get_master());
        element_begin.set_executor(exec->get_master());
        element_end.set_executor(exec->get_master());
        for (size_type i = 0; i < n; i++) {
            auto begin_rel = element_begin.get_data()[i];
            auto begin_anch =
                sm_baselines.get_data()[element_begin_sms.get_data()[i]];
            auto begin = begin_rel - (gko::int64)begin_anch;
            auto end_rel = element_end.get_data()[i];
            auto end_anch =
                sm_baselines.get_data()[element_end_sms.get_data()[i]];
            auto end = end_rel - (gko::int64)end_anch;

            output << i << ',' << begin << ',' << end;
            for (auto j = rowptrsh.get_const_data()[i];
                 j < rowptrsh.get_const_data()[i + 1] - 1; ++j) {
                output << ','
                       << element_loop.get_const_data()[j] -
                              (gko::int64)begin_anch;
                output << ',' << element_spins.get_const_data()[j];
            }
            output << '\n';
        }

        output.close();
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        // Pre-Volta GPUs may deadlock due to missing independent thread
        // scheduling.
        const auto is_fallback_required = exec->get_major_version() < 7;

        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        const auto new_b = matrix::Dense<ValueType>::create(exec);
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());

        // Initialize x to all NaNs.
        dense::fill(exec, x, nan<ValueType>());

        array<bool> nan_produced(exec, 1);
        array<IndexType> atomic_counter(exec, 1);
        sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                     atomic_counter.get_data());

        const dim3 block_size(
            is_fallback_required ? fallback_block_size : default_block_size, 1,
            1);
        const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

        if (is_fallback_required) {
            if (is_upper) {
                sptrsv_naive_legacy_kernel<true><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    true, nan_produced.get_data(), atomic_counter.get_data());
            } else {
                sptrsv_naive_legacy_kernel<false><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    true, nan_produced.get_data(), atomic_counter.get_data());
            }
        } else {
            if (is_upper) {
                sptrsv_naive_caching_ud_kernel<true><<<grid_size, block_size>>>(
                    scaled_m->get_const_row_ptrs(),
                    scaled_m->get_const_col_idxs(),
                    as_cuda_type(scaled_m->get_const_values()),
                    as_cuda_type(new_b->get_const_values()), b->get_stride(),
                    as_cuda_type(x->get_values()), x->get_stride(), n, nrhs,
                    nan_produced.get_data(), atomic_counter.get_data());
            } else {
                sptrsv_naive_caching_ud_kernel<false>
                    <<<grid_size, block_size>>>(
                        scaled_m->get_const_row_ptrs(),
                        scaled_m->get_const_col_idxs(),
                        as_cuda_type(scaled_m->get_const_values()),
                        as_cuda_type(new_b->get_const_values()),
                        b->get_stride(), as_cuda_type(x->get_values()),
                        x->get_stride(), n, nrhs, nan_produced.get_data(),
                        atomic_counter.get_data());
            }
        }
#if GKO_VERBOSE_LEVEL >= 1
        if (exec->copy_val_to_host(nan_produced.get_const_data())) {
            std::cerr << "Error: triangular solve produced NaN, either not all "
                         "diagonal "
                         "elements are nonzero, or the system is very "
                         "ill-conditioned. "
                         "The NaN will be replaced with a zero.\n";
        }
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
};


// EXPERIMENTAL ABOVE


template <bool is_upper, typename ValueType, typename IndexType>
__global__ void sptrsvppi_create_ppi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const IndexType* const colptrs, const IndexType* const rowidxs,
    IndexType* const factor_sizes, const IndexType level_height,
    IndexType* const factor_assignments,
    IndexType* const
        work_buffers,  // block_size elements of workspace for each block
    const IndexType* const purely_diagonal_elements,
    int32* const blocks_done,  // inited to zeros
    ValueType dummy)
{}


template <typename ValueType, typename IndexType>
void sptrsvppi_create_ppi_cpukernel(
    std::shared_ptr<const CudaExecutor> exec, const IndexType* const rowptrs,
    const IndexType* const colidxs, const IndexType* const colptrs,
    const IndexType* const rowidxs,
    IndexType* const factor_sizes,  // init to 0
    IndexType* const factor_assignments,
    const IndexType* const purely_diagonal_elements,
    const IndexType purely_diagonal_count,
    std::unordered_set<int>* const rem_preds,
    IndexType* const counts,  // inited to indegrees
    const IndexType n, const IndexType nnz, IndexType* const m, ValueType dummy)
{
    auto e = std::vector<IndexType>();
    auto d = std::vector<IndexType>();

    e.insert(e.end(), purely_diagonal_elements,
             purely_diagonal_elements + purely_diagonal_count);

    auto factor = 0;

    auto i = 0;
    while (i < n) {
        while (!e.empty()) {
            const auto v = e.back();
            e.pop_back();

            const auto succ_start = colptrs[v] + 1;
            const auto succ_end = colptrs[v + 1];

            auto each_succ_is_succ_of_all_preds = true;


            for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
                const auto succ = rowidxs[succ_i];

                for (const auto& pred : *(rem_preds + v)) {
                    if (factor_assignments[pred] != factor) {
                        continue;
                    }

                    auto is_succ_of_this_pred = false;

                    for (auto pred_succ_i = colptrs[pred] + 1;
                         pred_succ_i < colptrs[pred + 1]; ++pred_succ_i) {
                        if (rowidxs[pred_succ_i] == succ) {
                            is_succ_of_this_pred = true;
                            // printf("grep1 At least one true no-fill add!\n");
                            break;
                        }
                    }

                    if (!is_succ_of_this_pred) {
                        each_succ_is_succ_of_all_preds = false;
                        goto after;
                    }
                }
            }
        after:

            if (each_succ_is_succ_of_all_preds) {
                i += 1;
                factor_assignments[v] = factor;
                factor_sizes[factor] += 1;

                for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
                    const auto succ = rowidxs[succ_i];

                    for (const auto& v_pred : *(rem_preds + v)) {
                        (rem_preds + succ)->erase(v_pred);
                    }

                    counts[succ] -= 1;

                    if (counts[succ] == 0) {
                        e.push_back(succ);
                    }
                }
            } else {
                d.push_back(v);
            }
        }

        factor += 1;
        *m = factor + 1;

        e.swap(d);
        d.clear();  // should already be empty
    }
}


template <typename IndexType, bool is_upper>
__global__ void sptrsvdrpi_first_two_levels_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const int32* const already_factorized,  // 1 where already factorized
    IndexType* const levels, const IndexType n)
{
    // __shared__ uninitialized_array<IndexType, default_block_size>
    // level_s_array;
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    if (already_factorized[row]) {
        levels[row] = -one<IndexType>();
        return;
    }

    // const auto self_shmem_id = full_gid / default_block_size;
    // const auto self_shid = full_gid % default_block_size;

    // IndexType* level_s = level_s_array;
    // level_s[self_shid] = -1;

    // __syncthreads();

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_end = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    IndexType level = -one<IndexType>();
    for (auto i = row_begin; i != row_end; i += row_step) {
        const auto dependency = colidxs[i];
        if (is_upper ? dependency <= row : dependency >= row) {
            break;
        }

        if (already_factorized[dependency]) {
            continue;
        }

        auto l_p = &levels[dependency];

        // const auto dependency_gid = is_upper ? n - 1 - dependency :
        // dependency; const bool shmem_possible =
        //     (dependency_gid / default_block_size) == self_shmem_id;
        // if (shmem_possible) {
        //     const auto dependency_shid = dependency_gid % default_block_size;
        //     l_p = &level_s[dependency_shid];
        // }

        IndexType l = *l_p;
        while (l == -one<IndexType>()) {
            l = load_relaxed(l_p);
        }

        level = max(l, level);

        if (level > 1) {
            break;
        }
    }

    // store(level_s, self_shid, level + 1);
    levels[row] = level + 1;
}


template <typename IndexType>
__global__ void sptrsvdrpi_set_factorized_status_kernel(
    const IndexType* const factor_set,
    int32* const already_factorized,  // set to 1 where already factorized
    const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto fgid = factor_set[gid];

    if (fgid == -one<IndexType>()) {
        return;
    }

    already_factorized[fgid] = 1;
}


// TODO:
// SPeed up this kernel by precomputing the predecessor-predecessor sets of each
// row that should be L^2
template <typename IndexType, bool is_upper>
__global__ void sptrsvdrpi_factor_partition_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const int32* const row_hashes,
    IndexType* const factor_assignments,  // init to -1
    IndexType* const factor_sizes, IndexType* const m, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto row = gid;

    auto factor = -one<IndexType>();

    const auto pred_start = rowptrs[row];
    const auto pred_end = rowptrs[row + 1] - 1;
    for (auto pred_i = pred_start; pred_i < pred_end; ++pred_i) {
        const auto pred = colidxs[pred_i];

        // Spin while this predecessor has not been decided upon
        auto factor_assignment_load = factor_assignments[pred];
        while (factor_assignment_load == -1) {
            // Acquire becuase this syncs with the later pred-pred load/store
            factor_assignment_load = load_acquire(factor_assignments + pred);
        }

        factor = max(factor, factor_assignment_load);
    }

    // Early exit for independent rows
    if (pred_start >= pred_end) {
        store_release(factor_assignments + row, 0);
        atomic_add(factor_sizes, one<IndexType>());

        if (load_relaxed(m) < 1) {
            atomic_max(m, one<IndexType>());
        }

        return;
    }


    for (auto pred_i = pred_start; pred_i < pred_end; ++pred_i) {
        const auto pred = colidxs[pred_i];

        if (factor_assignments[pred] != factor) {
            continue;
        }

        const auto pred_pred_start = rowptrs[pred];
        const auto pred_pred_end = rowptrs[pred + 1] - 1;
        // auto relevant_pred_pred_count = 0;
        // for(auto pred_pred_i = pred_pred_start; pred_pred_i < pred_pred_end;
        // ++pred_pred_i){
        //     const auto pred_pred = colidxs[pred_pred_i];
        //     relevant_pred_pred_count += factor_assignments[pred_pred] ==
        //     factor;
        // }

        // if(relevant_pred_pred_count > pred_end - pred_start){
        //     // Fail - include into next factor
        //     store_release(factor_assignments + row, factor + 1);
        //     atomic_add(factor_sizes + factor + 1, one<IndexType>());

        //     if (load_relaxed(m) < factor + 2){
        //         atomic_max(m, factor + 2);
        //     }

        //     return;
        // }

        for (auto pred_pred_i = pred_pred_start; pred_pred_i < pred_pred_end;
             ++pred_pred_i) {
            const auto pred_pred = colidxs[pred_pred_i];

            // DEBUG
            // if(load_acquire(factor_assignments + pred_pred) == -1){
            //     printf("Error: Not included pred pred\n");
            // }

            if (factor_assignments[pred_pred] != factor) {
                continue;
            }

            // Check if this pred pred is also a pred of this row
            // TODO: Speed up this check by having minimal hashed info for each
            // row available Like, 32 bits, and a bit that's set means "this row
            // has a pred whose index divded by 32 leaves this remainder" Then,
            // if this check is already negative, immediate fail

            // const auto has_necessary_hash = row_hashes[row] & (0b1 <<
            // (pred_pred & 0b11111));

            // if(!has_necessary_hash){
            //     // Fail - include into next factor
            //     store_release(factor_assignments + row, factor + 1);
            //     atomic_add(factor_sizes + factor + 1, one<IndexType>());

            //     if (load_relaxed(m) < factor + 2){
            //         atomic_max(m, factor + 2);
            //     }

            //     return;
            // }

            auto pred_pred_also_pred = false;
            for (auto pred_j = pred_start; pred_j < pred_i; ++pred_j) {
                const auto pred_b = colidxs[pred_j];

                if (pred_b == pred_pred) {
                    // it is, we can still hope to include this row into the
                    // current factor
                    pred_pred_also_pred = true;
                    break;
                }
            }


            if (!pred_pred_also_pred) {
                // Fail - include into next factor
                store_release(factor_assignments + row, factor + 1);
                atomic_add(factor_sizes + factor + 1, one<IndexType>());

                if (load_relaxed(m) < factor + 2) {
                    atomic_max(m, factor + 2);
                }

                return;
            }
        }
    }

    // Include into this factor
    store_release(factor_assignments + row, factor);
    atomic_add(factor_sizes + factor, one<IndexType>());

    if (load_relaxed(m) < factor + 1) {
        atomic_max(m, factor + 1);
    }
}


template <typename IndexType, bool is_upper>
__global__ void sptrsvdrpi_factor_partition_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    IndexType* const
        waits,  // at pos i sum of row lengths of each element present in row i
                // (minus all diagonals) + number of direct elements in that row
    IndexType* const factor_assignments,  // init to 0
    IndexType* const factor_sizes, IndexType* const m, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;

    // DEBUG
    // auto i = 0;
    while (load_acquire(waits + col) > 0) {
        // Spin
        // i += 1;
        // if(col < 5 ){
        //     // printf("Col %d keeeps on spinning, with wait count %d\n",
        //     (int32)col, (int32)load_acquire(waits + col)); printf("Col %d has
        //     following predecessors: %d\n", (int32)col,
        //     (int32)load_acquire(waits + col));

        //     for(auto i = rowptrs[col]; i  < rowptrs[col + 1] - 1; ++i){
        //         const auto pred = colidxs[i];
        //         printf("%d\n", (int32)pred);
        //         for(auto j = rowptrs[pred]; j  < rowptrs[pred + 1] - 1; ++j){
        //             const auto ppred = colidxs[j];
        //             printf("%d ", (int32)ppred);
        //         }
        //         printf("\n");
        //     }
        // }
    }

    const auto factor = load_relaxed(factor_assignments +
                                     col);  // is load_relaxed necessary here?
    atomic_add(factor_sizes + factor, one<IndexType>());
    if (load_relaxed(m) < factor + 1) {
        atomic_max(m, factor + 1);
    }

    // if(col < 10){
    //         printf("Col %d (factor %d)has %d following predecessors\n",
    //         (int32)col,  (int32)load_acquire(factor_assignments + col),
    //         (int32)(rowptrs[col + 1] - rowptrs[col] - 1));

    // for(auto i = rowptrs[col]; i  < rowptrs[col + 1] - 1; ++i){
    //     const auto pred = colidxs[i];
    //     printf("%d (%d)\n", (int32)pred,
    //     (int32)load_acquire(factor_assignments + pred)); for(auto j =
    //     rowptrs[pred]; j  < rowptrs[pred + 1] - 1; ++j){
    //         const auto ppred = colidxs[j];
    //         printf("%d (%d)", (int32)ppred,
    //         (int32)load_acquire(factor_assignments + ppred));
    //     }
    //     printf("\n");
    // }
    // }


    const auto succ_start = colptrs[col] + 1;
    const auto succ_end = colptrs[col + 1];
    for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
        const auto succ = rowidxs[succ_i];

        // Non global load is intended
        if (factor_assignments[succ] < factor) {
            atomic_max(factor_assignments + succ, factor);
            __threadfence();
        }
        atomic_add(waits + succ, -one<IndexType>());

        // if(succ == 133){
        //     printf("Case 1; Col %d just subtracted 1 from 133\n",
        //     (int32)col);
        // }
    }


    for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
        const auto succ = rowidxs[succ_i];

        const auto succ_succ_start = colptrs[succ] + 1;
        const auto succ_succ_end = colptrs[succ + 1];

        // In that case, there's no hope for the succ-succs to be in factor
        if (load_acquire(factor_assignments + succ) > factor) {
            for (auto succ_succ_i = succ_succ_start;
                 succ_succ_i < succ_succ_end; ++succ_succ_i) {
                const auto succ_succ = rowidxs[succ_succ_i];

                atomic_add(waits + succ_succ, -one<IndexType>());

                // if(succ_succ == 133){
                //     printf("Case 2; Col %d just subtracted 1 from 133\n",
                //     (int32)col);
                // }
            }
            continue;
        }

        for (auto succ_succ_i = succ_succ_start; succ_succ_i < succ_succ_end;
             ++succ_succ_i) {
            const auto succ_succ = rowidxs[succ_succ_i];

            if (factor_assignments[succ_succ] > factor) {
                atomic_add(waits + succ_succ, -one<IndexType>());
                // if(succ_succ == 133){
                //         printf("Case 5; Col %d just subtracted 1 from 133\n",
                //         (int32)col);
                // }
                continue;
            }

            auto found_as_succ = false;

            for (auto succ_j = succ_i + 1; succ_j < succ_end; ++succ_j) {
                const auto succ_b = rowidxs[succ_j];

                if (succ_b > succ_succ) {
                    break;
                }

                if (succ_b == succ_succ) {
                    found_as_succ = true;
                    atomic_add(waits + succ_succ, -one<IndexType>());
                    break;
                }
            }

            // const auto pred_start = rowptrs[succ_succ];
            // const auto pred_end = rowptrs[succ_succ + 1] - 1;
            // for(auto pred_i = pred_start; pred_i < pred_end; ++pred_i){
            //     const auto pred = colidxs[pred_i];

            //     if(pred > col){
            //         break;
            //     }

            //     if(pred == col){
            //         found_self = true;
            //         atomic_add(waits + succ_succ, -one<IndexType>());
            //         // if(succ_succ == 5){
            //         //     printf("Case 3; Col %d just subtracted 1 from
            //         133\n", (int32)col);
            //         // }
            //         break;
            //     }
            // }

            if (!found_as_succ) {
                // if(succ_succ == 5){
                //     printf("Error; Col %d did not find itself as pred of
                //     5\n", (int32)col);
                // }
                atomic_max(factor_assignments + succ_succ, factor + 1);
                __threadfence();
                atomic_add(waits + succ_succ, -one<IndexType>());
                // if(succ_succ == 133){
                //         printf("Case 4; Col %d just subtracted 1 from 133\n",
                //         (int32)col);
                // }
            }
        }
    }
}


template <typename IndexType, bool is_upper>
__global__ void sptrsvdrmpi_factor_partition_csc_kernel(
    const IndexType* const* const rowptrs_batch,
    const IndexType* const* const colptrs_batch,
    const IndexType* const* const rowidxs_batch,
    const IndexType* const* const colidxs_batch,
    IndexType* const csc_waits_batch,
    IndexType* const factor_sizes_batch,  // init to 0
    IndexType* const factor_assignments_batch, const IndexType* const ns,
    const IndexType* const partition_starts, const IndexType multi_p_count,
    IndexType* const ms)
{
    const auto full_gid = thread::get_thread_id_flat();

    const auto warp_id = full_gid / 32;
    const auto partition = warp_id % multi_p_count;
    const auto gid = (full_gid % 32) + 32 * (full_gid / (multi_p_count * 32));

    const auto n = ns[partition];

    if (gid >= n) {
        return;
    }

    const auto partition_start = partition_starts[partition];

    const auto colptrs = colptrs_batch[partition];
    const auto rowidxs = rowidxs_batch[partition];
    auto factor_sizes = factor_sizes_batch + partition_start;
    auto factor_assignments = factor_assignments_batch + partition_start;
    auto waits = csc_waits_batch + partition_start;
    auto m = ms + partition;


    const auto col = gid;

    auto i = 0;
    while (load_acquire(waits + col) > 0) {
        // Spin
        i += 1;
        if (col < 5) {
            // printf("Col %d partition %d keeeps on spinning, with wait count
            // %d\n", (int32)col, (int32)partition, (int32)load_acquire(waits +
            // col)); printf("Col %d has following predecessors: %d\n",
            // (int32)col, (int32)load_acquire(waits + col));

            // for(auto i = rowptrs[col]; i  < rowptrs[col + 1] - 1; ++i){
            //     const auto pred = colidxs[i];
            //     printf("%d\n", (int32)pred);
            //     for(auto j = rowptrs[pred]; j  < rowptrs[pred + 1] - 1; ++j){
            //         const auto ppred = colidxs[j];
            //         printf("%d ", (int32)ppred);
            //     }
            //     printf("\n");
            // }
        }
    }

    const auto factor = load_relaxed(factor_assignments +
                                     col);  // is load_relaxed necessary here?
    atomic_add(factor_sizes + factor, one<IndexType>());

    // if(partition == 0 && factor == 0){
    //     printf("Col %d partition %d decided on factor 0\n", (int32)col,
    //     (int32)partition);
    // }

    // if(old > 200){
    //     printf("Col %d partition %d added over 200 to factor %d\n",
    //     (int32)col, (int32)partition, (int32)factor);
    // }

    // if(factor == 2){
    //     printf("Col %d partition %d decided on factor 2\n", (int32)col,
    //     (int32)partition);
    // }

    if (load_relaxed(m) < factor + 1) {
        // if(factor + 1 == 3){
        //     printf("Col %d partition %d decided to max m to 3\n", (int32)col,
        //     (int32)partition);
        // }
        atomic_max(m, factor + 1);
    }


    const auto succ_start = colptrs[col] + 1;
    const auto succ_end = colptrs[col + 1];
    for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
        const auto succ = rowidxs[succ_i];

        // Non global load is intended
        if (factor_assignments[succ] < factor) {
            atomic_max(factor_assignments + succ, factor);
            __threadfence();

            // if(factor < 0 || factor > 10){
            //     printf("Col %d partition %d wrote factor %d to col %d\n",
            //     (int32)col, (int32)partition, (int32)factor, (int32)succ);
            // }
        }
        atomic_add(waits + succ, -one<IndexType>());
    }


    for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
        const auto succ = rowidxs[succ_i];

        const auto succ_succ_start = colptrs[succ] + 1;
        const auto succ_succ_end = colptrs[succ + 1];

        // In that case, there's no hope for the succ-succs to be in factor
        if (load_acquire(factor_assignments + succ) > factor) {
            for (auto succ_succ_i = succ_succ_start;
                 succ_succ_i < succ_succ_end; ++succ_succ_i) {
                const auto succ_succ = rowidxs[succ_succ_i];

                atomic_add(waits + succ_succ, -one<IndexType>());
            }
            continue;
        }

        for (auto succ_succ_i = succ_succ_start; succ_succ_i < succ_succ_end;
             ++succ_succ_i) {
            const auto succ_succ = rowidxs[succ_succ_i];

            if (factor_assignments[succ_succ] > factor) {
                atomic_add(waits + succ_succ, -one<IndexType>());
                continue;
            }

            auto found_as_succ = false;

            for (auto succ_j = succ_i + 1; succ_j < succ_end; ++succ_j) {
                const auto succ_b = rowidxs[succ_j];

                if (succ_b > succ_succ) {
                    break;
                }

                if (succ_b == succ_succ) {
                    found_as_succ = true;
                    atomic_add(waits + succ_succ, -one<IndexType>());
                    break;
                }
            }

            if (!found_as_succ) {
                atomic_max(factor_assignments + succ_succ, factor + 1);
                __threadfence();
                atomic_add(waits + succ_succ, -one<IndexType>());

                // if(factor < 0 || factor > 9){
                //     printf("Col %d partition %d wrote factor %d to col %d\n",
                //     (int32)col, (int32)partition, (int32)factor,
                //     (int32)succ_succ);
                // }
            }
        }
    }
}


template <typename IndexType, bool is_upper>
__global__ void sptrsvdrpi_rowhash_kernel(const IndexType* const rowptrs,
                                          const IndexType* const colidxs,
                                          int32* const row_hashes,
                                          const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto row = gid;

    const auto pred_start = rowptrs[row];
    const auto pred_end = rowptrs[row + 1] - 1;
    for (auto pred_i = pred_start; pred_i < pred_end; ++pred_i) {
        const auto pred = colidxs[pred_i];
        row_hashes[row] |= (0b1 << (pred & 0b11111));
    }
}

template <typename IndexType, bool is_upper>
__global__ void sptrsvdrpi_compute_cscwaits_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    IndexType* const csc_waits, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto row = gid;

    const auto row_start = rowptrs[row];
    const auto row_end = rowptrs[row + 1] - 1;

    auto waits = row_end - row_start;
    for (auto i = row_start; i < row_end; ++i) {
        const auto pred = colidxs[i];
        waits += rowptrs[pred + 1] - rowptrs[pred] - 1;
    }

    csc_waits[row] = waits;
}


template <typename ValueType, typename IndexType>
void sptrsvdrpi_create_ppi(std::shared_ptr<const CudaExecutor> exec,
                           const IndexType* const rowptrs,
                           const IndexType* const colidxs,
                           const IndexType* const colptrs,
                           const IndexType* const rowidxs,
                           IndexType* const factor_sizes,  // init to 0
                           IndexType* const factor_assignments,
                           const IndexType n, IndexType* const m,
                           ValueType dummy)
{
    // array<IndexType> satellites(exec, n);
    // thrust::sequence(thrust::device, satellites.get_data(), 0);

    array<IndexType> csc_waits(exec, n);

    array<IndexType> m_d(exec, 1);
    m_d.fill(zero<IndexType>());

    array<int32> row_hashes(exec, n);
    cudaMemset(row_hashes.get_data(), 0, n * sizeof(int32));

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);

    cudaMemset(factor_assignments, 0, n * sizeof(IndexType));

    sptrsvdrpi_compute_cscwaits_kernel<IndexType, false>
        <<<grid_size, block_size>>>(rowptrs, colidxs, csc_waits.get_data(), n);

    sptrsvdrpi_rowhash_kernel<IndexType, false>
        <<<grid_size, block_size>>>(rowptrs, colidxs, row_hashes.get_data(), n);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // sptrsvdrpi_factor_partition_kernel<IndexType, false><<<grid_size,
    // block_size>>>(
    //     rowptrs, colidxs, row_hashes.get_const_data(), factor_assignments,
    //     factor_sizes, m_d.get_data(), n
    // );

    sptrsvdrpi_factor_partition_csc_kernel<IndexType, false>
        <<<grid_size, block_size>>>(colptrs, rowidxs, csc_waits.get_data(),
                                    factor_assignments, factor_sizes,
                                    m_d.get_data(), n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Time to generate:  %3.1f ms \n", time);


    *m = exec->copy_val_to_host(m_d.get_const_data()) + 1;

    // auto factor = 1;
    // auto i = 0;

    // while(i < n){

    //     // sptrsvdrpi_first_two_levels_kernel<IndexType, false><<<grid_size,
    //     block_size>>>(
    //     //     rowptrs,
    //     //     colidxs,
    //     //     already_factorized.get_const_data(),
    //     //     levels.get_data(), n);

    //     thrust::copy_if(thrust::device, satellites.get_data(),
    //     satellites.get_data() + n, factor_set.get_data(),
    //         [completion_status_p](IndexType sat) { return
    //         completion_status[sat] == 1; });

    //     sptrsvdrpi_set_factorized_status_kernel<<<grid_size,
    //     block_size>>>(factor_set.get_data(), already_factorized.get_data(),
    //     n);


    //     sptrsvdrpi_complete_factor_kernel<IndexType, false><<<grid_size,
    //     block_size>>>(
    //         rowptrs, colidxs, completion_status.get_data(),
    //         factor_assignments, factor, n
    //     );


    //     cudaMemset(factor_set.get_data(), 0, n * sizeof(IndexType));
    // }
}

template <typename IndexType>
__global__ void sptrsvrdmpi_factor_asssignments_gather_kernel(
    const IndexType* const factor_assignments_batch,
    IndexType* const factor_assignments, const IndexType* const ns,
    const IndexType* const ms, const IndexType multi_p_count, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    auto partition = 0;
    auto n_counter = 0;
    auto m_sum = 0;
    for (; partition < multi_p_count; ++partition) {
        const auto next = n_counter + ns[partition];

        if (next > gid && n_counter <= gid) {
            break;
        }
        n_counter = next;
        m_sum += ms[partition];
    }

    factor_assignments[gid] = factor_assignments_batch[gid] + m_sum;
}


template <typename IndexType, bool is_upper>
__global__ void sptrsvdrmpi_compute_cscwaits_kernel(
    const IndexType* const* const rowptrs_batch,
    const IndexType* const* const colidxs_batch,
    IndexType* const csc_waits_batch, const IndexType* const partition_starts,
    const IndexType multi_p_count, const IndexType* const ns)
{
    const auto full_gid = thread::get_thread_id_flat();

    const auto warp_id = full_gid / 32;
    const auto partition = warp_id % multi_p_count;
    const auto gid = (full_gid % 32) + 32 * (full_gid / (multi_p_count * 32));

    const auto n = ns[partition];

    if (gid >= n) {
        return;
    }

    const auto partition_start = partition_starts[partition];

    const auto rowptrs = rowptrs_batch[partition];
    const auto colidxs = colidxs_batch[partition];
    auto csc_waits = csc_waits_batch + partition_start;

    const auto row = gid;

    const auto row_start = rowptrs[row];
    const auto row_end = rowptrs[row + 1] - 1;

    auto waits = row_end - row_start;
    for (auto i = row_start; i < row_end; ++i) {
        const auto pred = colidxs[i];
        waits += rowptrs[pred + 1] - rowptrs[pred] - 1;
    }

    // if(gid == 0){
    //     printf("Indeed gid 0 juut got written %d waits\n", (int32)waits);
    // }

    csc_waits[row] = waits;
}


template <typename IndexType>
__global__ void sptrsvrdmpi_assemble_perm_kernel(
    const IndexType* const levelsperm, const IndexType* const second_perm,
    IndexType* const final_perm, const IndexType n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto first = levelsperm[gid];
    const auto second = second_perm[first];

    final_perm[gid] = second;
}


template <typename ValueType, typename IndexType>
void sptrsvdrmpi_create_ppi(
    std::shared_ptr<const CudaExecutor> exec, const IndexType* const rowptrs,
    const IndexType* const colidxs, const IndexType* const colptrs,
    const IndexType* const rowidxs, const IndexType* const* const rowptrs_batch,
    const IndexType* const* const colptrs_batch,
    const IndexType* const* const rowidxs_batch,
    const IndexType* const* const colidxs_batch,
    IndexType* const factor_sizes_batch,  // init to 0
    IndexType* const factor_assignments_batch, const IndexType n,
    const IndexType* const ns, const IndexType* const partition_starts,
    const IndexType multi_p_count, IndexType* const ms, ValueType dummy)
{
    // array<IndexType> satellites(exec, n);
    // thrust::sequence(thrust::device, satellites.get_data(), 0);

    auto blocks_needed = 0;

    array<IndexType> csc_waits_batch(exec, n);
    for (auto i = 0; i < multi_p_count; ++i) {
        blocks_needed += ceildiv(ns[i], default_block_size);
    }
    cudaMemset(csc_waits_batch.get_data(), 0, n * sizeof(IndexType));

    array<IndexType> ns_d(exec, multi_p_count);
    cudaMemcpy(ns_d.get_data(), ns, multi_p_count * sizeof(IndexType),
               cudaMemcpyHostToDevice);


    array<IndexType> ms_d(exec, multi_p_count);
    cudaMemset(ms_d.get_data(), 0, multi_p_count * sizeof(IndexType));

    // array<int32> row_hashes(exec, n);
    // cudaMemset(row_hashes.get_data(), 0, n * sizeof(int32));
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(blocks_needed, 1, 1);
    sptrsvdrmpi_compute_cscwaits_kernel<IndexType, false>
        <<<grid_size, block_size>>>(
            rowptrs_batch, colidxs_batch, csc_waits_batch.get_data(),
            partition_starts, multi_p_count, ns_d.get_const_data());


    // sptrsvdrpi_rowhash_kernel<IndexType, false><<<grid_size,
    // block_size>>>(rowptrs, colidxs, row_hashes.get_data(), n);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // sptrsvdrpi_factor_partition_kernel<IndexType, false><<<grid_size,
    // block_size>>>(
    //     rowptrs, colidxs, row_hashes.get_const_data(), factor_assignments,
    //     factor_sizes, m_d.get_data(), n
    // );

    sptrsvdrmpi_factor_partition_csc_kernel<IndexType, false>
        <<<grid_size, block_size>>>(
            rowptrs_batch, colptrs_batch, rowidxs_batch, colidxs_batch,
            csc_waits_batch.get_data(), factor_sizes_batch,
            factor_assignments_batch, ns_d.get_data(), partition_starts,
            multi_p_count, ms_d.get_data());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Time to generate:  %3.1f ms \n", time);


    cudaMemcpy(ms, ms_d.get_data(), multi_p_count * sizeof(IndexType),
               cudaMemcpyDeviceToHost);
}


template <typename ValueType, typename IndexType>
void sptrsvdrpi_create_ppi_cpukernel(
    std::shared_ptr<const CudaExecutor> exec, const IndexType* const rowptrs,
    const IndexType* const colidxs, const IndexType* const colptrs,
    const IndexType* const rowidxs,
    IndexType* const factor_sizes,  // init to 0
    IndexType* const factor_assignments,
    const IndexType* const purely_diagonal_elements,
    const IndexType purely_diagonal_count,
    const std::unordered_set<int>* const preds,
    IndexType* const counts,  // inited to indegrees
    const IndexType n, const IndexType nnz, IndexType* const m, ValueType dummy)
{
    auto e = std::vector<IndexType>();
    auto d = std::vector<IndexType>();

    for (auto diag_i = 0; diag_i < purely_diagonal_count; ++diag_i) {
        e.push_back(purely_diagonal_elements[diag_i]);
    }

    auto factor = 0;

    auto i = 0;
    while (i < n) {
        while (!e.empty()) {
            // pop the smallest node
            const auto v = e.back();
            e.pop_back();

            // check if all this nodes predecessors in this factor only have
            // predecessors in this factor which are also predecessors of this
            // node if yes, add this node to this factor
            auto can_be_included = true;

            const auto pred_start = rowptrs[v];
            const auto pred_end = rowptrs[v + 1] - 1;

            for (auto pred_i = pred_start; pred_i < pred_end; ++pred_i) {
                const auto pred = colidxs[pred_i];

                if (factor_assignments[pred] != factor) {
                    continue;
                }

                const auto pred_pred_start = rowptrs[pred];
                const auto pred_pred_end = rowptrs[pred + 1] - 1;

                for (auto pred_pred_i = pred_pred_start;
                     pred_pred_i < pred_pred_end; ++pred_pred_i) {
                    const auto pred_pred = colidxs[pred_pred_i];

                    if (factor_assignments[pred_pred] != factor) {
                        continue;
                    }

                    if (preds[v].find(pred_pred) == preds[v].end()) {
                        can_be_included = false;
                        goto after;
                    }
                }
            }

        after:

            // if this node has been added to this factor, insert all it's
            // successors which have no predeceassors left to be numbered into e
            if (can_be_included) {
                const auto succ_start = colptrs[v] + 1;
                const auto succ_end = colptrs[v + 1];

                for (auto succ_i = succ_start; succ_i < succ_end; ++succ_i) {
                    const auto succ = rowidxs[succ_i];

                    counts[succ] -= 1;

                    if (counts[succ] == 0) {
                        e.push_back(succ);
                    }
                }

                factor_assignments[v] = factor;
                factor_sizes[factor] += 1;
                i += 1;
            } else {
                // if no, try adding this node to the next factor
                d.push_back(v);
            }
        }

        for (auto d_i = 0; d_i < d.size(); ++d_i) {
            e.push_back(d[d_i]);
        }
        std::vector<IndexType>().swap(d);

        factor += 1;
    }

    *m = factor + 1;
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_negate_ppi_kernel(const IndexType* const colptrs,
                                            const IndexType* const rowidxs,
                                            ValueType* const cvalues,
                                            const gko::size_type nnz)
{
    const auto nnz_gid = thread::get_thread_id_flat();

    if (nnz_gid >= nnz) {
        return;
    }

    cvalues[nnz_gid] = -cvalues[nnz_gid];
}

template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_negate_rdpi_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    ValueType* const cvalues, const IndexType* const factor_offsets,
    const IndexType* const factor_assignments, const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;
    const auto factor = factor_assignments[col];
    const auto factor_start = factor_offsets[factor];
    const auto factor_end = factor_offsets[factor + 1];

    for (auto i = colptrs[col] + 1; i < colptrs[col + 1]; ++i) {
        const auto row = rowidxs[i];

        if (row >= factor_end) {
            break;
        }

        cvalues[i] = -cvalues[i];
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_invert_ppi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    ValueType* const rvalues, const IndexType* const colptrs,
    const IndexType* const rowidxs,
    ValueType* const cvalues,  // all values under the diagonal already negated
    const IndexType* const factor_offsets,
    const IndexType* const factor_assignments, const gko::size_type n,
    const gko::size_type m,  // factor count
    const gko::size_type nnz)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n - 1) {
        return;
    }


    const auto col = gid;
    const auto factor = factor_assignments[col];
    const auto factor_start = factor_offsets[factor];
    const auto factor_end = factor < m ? factor_offsets[factor + 1] : n;

    for (auto i = colptrs[col] + 1; i < colptrs[col + 1]; ++i) {
        // compute the inv element at col, idx i

        // This is a sparse vector-vector dot A'[col+1:i,col] * A[i,col+1:i]
        // where A' is the in-place generated inv
        // handle the first entry (1*val => -val) by hand
        const auto row = rowidxs[i];
        const auto dot_end = std::min(
            row,
            (IndexType)factor_end);  // check if std::min(i, factor_end) works
        auto rowidx_current = colptrs[col] + 1;
        auto colidx_current = rowptrs[row];
        auto row_current = rowidxs[rowidx_current];
        auto col_current = colidxs[colidx_current];
        ValueType sum = zero<ValueType>();

        // if(row >= n){
        //     printf("Now performing illegal memory access 5\n");
        // }

        // if(col == 87801){
        //     printf("Col 87801 inital conf row %d: %d %d %d %d %d\n",
        //     (int32)row, (int32)dot_end, (int32)rowidx_current,
        //     (int32)colidx_current, (int32)row_current, (int32)col_current);
        // }

        while (row_current < dot_end && col_current < dot_end &&
               colidx_current < rowptrs[row + 1]) {
            if (row_current < col_current) {
                ++rowidx_current;

                // if(rowidx_current >= nnz){
                //     printf("Now performing illegal memory access 1\n");
                // }

                // if(col == 87801){
                //         printf("Col 87801 incremented rowidx_current, to %d
                //         (%d)\n", (int32)row_current, (int32)rowidx_current);
                // }


                row_current = rowidxs[rowidx_current];
            } else if (col_current < row_current) {
                ++colidx_current;

                // if(colidx_current >= nnz){
                //     printf("Now performing illegal memory access 2\n");
                // }

                // if(col == 87801){
                //         printf("Col 87801 incremented colidx_current, to %d
                //         (%d)\n", (int32)col_current, (int32)colidx_current);
                // }

                col_current = colidxs[colidx_current];
            } else {
                sum += cvalues[rowidx_current] * rvalues[colidx_current];
                ++rowidx_current;

                // if(col == 87801){
                //         printf("Col 87801 added to sum at %d == %d\n",
                //         (int32)col_current, (int32)row_current);
                // }

                // if(rowidx_current >= nnz){
                //     printf("Now performing illegal memory access 3\n");
                // }

                row_current = rowidxs[rowidx_current];
                ++colidx_current;

                // if(colidx_current >= nnz){
                //     printf("Now performing illegal memory access 4\n");
                // }

                col_current = colidxs[colidx_current];
            }
        }

        // if (thrust::abs(sum) >= 1e-8){
        //     printf("Adding something (fact %d - %d) (%d, %d, %d)\n",
        //     (int32)factor_start, (int32)factor_end, (int32)col, (int32)i,
        //     (int32)row);
        // }
        cvalues[i] += -sum;  // add to the -val already there
    }
}


template <typename IndexType>
__global__ void sptrsvppi_rowfactorsstarts_rspi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const IndexType* const factor_offsets, IndexType* const row_factor_starts,
    const gko::size_type n,
    const gko::size_type m  // factor count
)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto row = gid;
    IndexType factor;

    for (auto m_i = 0; m_i < m; ++m_i) {
        if (factor_offsets[m_i + 1] > row && factor_offsets[m_i] <= row) {
            factor = m_i;
            break;
        }
    }

    const auto row_start = rowptrs[row];
    const auto row_end = rowptrs[row + 1];  // Include diag for simplicity

    for (auto i = row_start; i < row_end; ++i) {
        const auto col = colidxs[i];

        if (col >= factor_offsets[factor]) {
            row_factor_starts[row] = i;
            break;
        }
    }
}


template <typename IndexType>
__global__ void sptrsvppi_rowfactorsstarts_rscpi_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const IndexType* const factor_offsets,
    IndexType* const col_factor_ends,
    int32 *const row_counts,
    const gko::size_type n,
    const gko::size_type m  // factor count
)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;
    IndexType factor;

    for (auto m_i = 0; m_i < m; ++m_i) {
        if (factor_offsets[m_i + 1] > col && factor_offsets[m_i] <= col) {
            factor = m_i;
            break;
        }
    }

    const auto col_start = colptrs[col];
    const auto col_end = colptrs[col + 1];  
    const auto row_start = rowptrs[col];
    const auto row_end = rowptrs[col + 1] - 1; 
     IndexType col_factor_i = col_end;
    // Actually, "col" and "row" is the same thing

    for (auto i = col_start; i < col_end; ++i) {
        const auto row = rowidxs[i];

        if (row >= factor_offsets[factor + 1]) {
            col_factor_i = i;
            break;
        }
    }
    col_factor_ends[col] = col_factor_i;


    // if(col == 51868){
    //             printf("Col/Row 51868 (factor: %d from %d to %d) (rowptr from %d to %d (diag, excl) (colptr %d : %d : %d))\n",
    //             (int32)factor, (int32)factor_offsets[factor], (int32)factor_offsets[factor + 1],
    //             (int32)row_start, (int32)row_end,
    //             (int32)col_start, (int32)col_factor_i, (int32)col_end);
    // }

    int32 row_count_0 = 0;
    int32 row_count_1 = row_end - row_start;

    for (auto i = row_end; i >= row_start; --i) {
        const auto colidx = colidxs[i];

        // if(col == 2129){
        //     printf("Row 2129 nnz: %d\n", (int32)colidx);
        // }

        if (colidx < factor_offsets[factor]) {
                      
            row_count_0 = i - row_start + 1;
            row_count_1 = row_end - i - 1;
            break;
        }
    }
    
    // if(col == 2129){
    //             printf("Col/Row 2129 (factor: %d from %d to %d) (rowptr from %d to %d (diag, excl)) computes c0 %d and c1 %d\n",
    //             (int32)factor, (int32)factor_offsets[factor], (int32)factor_offsets[factor + 1],
    //             (int32)row_start, (int32)row_end, 
    //             (int32)row_count_0, (int32)row_count_1);
    // }

    row_counts[2 * col] = row_count_0;
    row_counts[2 * col + 1] = row_count_1;
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_solve_rspi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const values, const IndexType* const row_factor_starts,
    const ValueType* const b,
    ValueType* const x_one,  // intermediate output
    ValueType* const x_two,  // final output
    const gko::size_type n)
{
    // __shared__ uninitialized_array<ValueType, default_block_size / 2> inner_sums_arr;
    // auto inner_sums_s = &inner_sums_arr[0];
    const auto gid = thread::get_thread_id_flat();
    

    if (gid >= 2 * n) {
        return;
    }

    // const auto row = gid;
    const auto row = gid / 2;
    const auto tid = gid % 2;


    // store_relaxed_shared(inner_sums_s + (row % 256), nan<ValueType>());
    // __syncthreads();

    
    const auto row_factor_i = min(row_factor_starts[row], rowptrs[row + 1] - 1);
    const auto row_start = tid == 0 ? rowptrs[row] : row_factor_i;
    const auto row_end = tid == 0 ? row_factor_i : rowptrs[row + 1] - 1;

    auto x = tid == 0 ? x_two : x_one;
    ValueType sum = tid == 0 ? -b[row] : zero<ValueType>();

    for(auto i = row_start; i < row_end; ++i){

        const auto col = colidxs[i];
        const auto v = values[i];
        
        auto x_load = x[col];
        while(is_nan(x_load)){
            x_load = load_relaxed(x + col);
        }

        sum += x_load * v;
    }

    if(tid == 0){
        sum = -sum;
        store_relaxed(x_one + row, sum);
        // store_relaxed_shared(inner_sums_s + (row % 256), sum);
    }
    else{
        // auto x_one_load = inner_sums_s[row % 256];
        auto x_one_load = x_one[row];
        while(is_nan(x_one_load)){
            // x_one_load = load_relaxed_shared(inner_sums_s + (row % 256));
            x_one_load = load_relaxed(x_one + row);
        }
        store_relaxed(x_two + row, sum + x_one_load);
    }

    // // const auto offset = (((gid % 32) / 2) * 2);
    // // const uint32 mask = 0b11 << offset;
    // // const gko::remove_complex<ValueType> send = thrust::complex<gko::remove_complex<ValueType>>(sum).real();

    // // __syncwarp(mask);
    
    // const auto shfl_sum_a = __shfl_sync(mask, send, offset);

    // if(tid == 1){
    //     store_relaxed(x_two + row, sum + shfl_sum_a);
    // }

    // const auto row_start = rowptrs[row];
    // const auto row_end = rowptrs[row + 1] - 1;
    // const auto row_factor_i = min(row_factor_starts[row], rowptrs[row + 1] - 1);

    // ValueType sum_1 = b[row];
    // ValueType sum_2 = 0;

    // auto i_2 = row_start;
    // auto i_1 = row_factor_i;

    // if(i_2 == row_factor_i){
    //     store_relaxed(x_one + row, sum_1);
    //     ++i_2;
    // }

    // int time_1 = 0;
    // int time_2 = 0;
    // while (i_2 < row_factor_i || i_1 < row_end) {

    //     if(i_2 < row_factor_i){
    //         const auto col = colidxs[i_2];

            
    //         auto x_two_load = load_relaxed(x_two + col);
    //         if (!is_nan(x_two_load)) {
    //             sum_1 -= x_two_load * values[i_2];
    //             // sum_2 -= x_two_load * values[i_2];

    //             ++i_2;
    //             time_1 = 0;
    //         }else{
    //             // __nanosleep(time_1);

    //             // if(time_1 < 1000){
    //             //     time_1 += 2;
    //             // }
    //         }
    //     }

    //     if(i_1 < row_end){
    //         const auto col = colidxs[i_1];

            
    //         auto x_one_load = load_relaxed(x_one + col);
    //         if (!is_nan(x_one_load)) {
    //             sum_2 += x_one_load * values[i_1];

    //             ++i_1;
    //             time_2 = 0;
    //         }
    //         else{
    //             // __nanosleep(time_2);

    //             // if(time_2 < 1000){
    //             //     time_2 += 2;
    //             // }
    //         }
    //     }
    //     if(i_2 == row_factor_i){
    //         store_relaxed(x_one + row, sum_1);

    //         ++i_2;
    //     }

    //     // if(time < 4000){
    //     //     time += 2;
    //     // }
    // }
    // // store_relaxed(x_two + row, sum_2);
    // store_relaxed(x_two + row, sum_1 + sum_2);



    // __syncwarp(); // Good for perf?

    // store_relaxed(x_one + row, sum);


    // for (auto i = row_factor_i; i < row_end; ++i) {
    //     const auto col = colidxs[i];

    //     const auto v = values[i];

    //     auto x_one_load = x_one[col];
    //     while (is_nan(x_one_load)) {
    //         x_one_load = load_relaxed(x_one + col);
    //     }

    //     sum += x_one_load * v;

    //     // if(row == 50){
    //     //     printf("Non-same-factor subtracting %.10lf = %.10lf * %.10lf to
    //     //     row 50\n", x_load * values[i], x_load, values[i]); printf("Sum is
    //     //     now %.10lf for row 50\n", sum);
    //     // }
    // }

    // store_relaxed(x_two + row, sum);
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_solve_rscpi_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const values, 
    const IndexType *const factor_assignments, // n values, each assigning a row/col (same thing) to a factor
    const IndexType* const col_factor_ends, // n values, col factor split points
    int32 *const row_counts, // 2n values, row count tid 0 and 1 each
    const ValueType* const b,
    ValueType* const x_one,  // intermediate output
    ValueType* const x_two,  // final output
    const gko::size_type n)
{
    // __shared__ uninitialized_array<ValueType, default_block_size / 2> inner_sums_arr;
    // auto inner_sums_s = &inner_sums_arr[0];
    const auto gid = thread::get_thread_id_flat();
    

    if (gid >= 2 * n) {
        return;
    }

    // const auto row = gid;
    const auto col = gid / 2;
    const auto tid = gid % 2;
    const auto factor = factor_assignments[col];

    
    const auto col_factor_i = col_factor_ends[col];
    const auto col_start = tid == 0 ? colptrs[col] + 1 : col_factor_i;
    const auto col_end = tid == 0 ? col_factor_i : colptrs[col + 1];

    
    const auto read_x = tid == 0 ? x_one : x_two;
    const auto write_x = tid == 0 ? x_two : x_one;

    
    const auto b_val = b[col];
    const auto row_counts_idx_0 = 2 * col;
    const auto row_counts_idx_1 = 2 * col + 1;



    while (load_relaxed(row_counts + row_counts_idx_0) > 0) { } // both spin on outer count

    if(tid == 1){
        while (load_relaxed(row_counts + row_counts_idx_1) > 0) { } // tid 1 also spin on inner count
    }

    const auto val = tid == 0 ? 
      (-load_relaxed(x_one + col) + b_val) 
    : (-load_relaxed(x_one + col) + b_val + load_relaxed(x_two + col));

    for(auto i = col_start; i < col_end; ++i){
            const auto row = rowidxs[i];
            const auto coeff = values[i];

            // if(row == 37914){
            //     printf("Col %d tid %d (%d : &%d : %d) writing to row %d\n", (int32)col, (int32)tid, 
            //     (int32)col_start, (int32)col_factor_i, (int32)col_end,
            //     (int32)row);
            // }

            atomic_add(write_x + row, val * coeff);
            // __threadfence();
            // __threadfence();
            // const auto is_inner = factor_assignments[row] == factor;
            atomic_add(row_counts + 2 * row + (1 - tid), -1);
     }

    if(tid == 1){
        store_relaxed(x_two + col, val);
    }
    
    // store_relaxed(write_x + col, tid == 0 ? val : -val + load_relaxed(x_one + col)); 
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_invert_rdpi_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    ValueType* const rvalues, const IndexType* const colptrs,
    const IndexType* const rowidxs,
    ValueType* const cvalues,  // all values under the diagonal already negated
    const IndexType* const factor_offsets,
    const IndexType* const factor_assignments, const gko::size_type n,
    const gko::size_type m,  // factor count
    const gko::size_type nnz)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n - 1) {
        return;
    }


    const auto col = gid;
    const auto factor = factor_assignments[col];
    const auto factor_start = factor_offsets[factor];
    const auto factor_end = factor < m ? factor_offsets[factor + 1] : n;

    for (auto i = colptrs[col] + 1; i < colptrs[col + 1]; ++i) {
        // compute the inv element at col, idx i

        // This is a sparse vector-vector dot A'[col+1:i,col] * A[i,col+1:i]
        // where A' is the in-place generated inv
        // handle the first entry (1*val => -val) by hand
        const auto row = rowidxs[i];

        if (row >= factor_end) {
            break;
        }

        const auto dot_end = std::min(
            row,
            (IndexType)factor_end);  // check if std::min(i, factor_end) works
        auto rowidx_current = colptrs[col] + 1;
        auto colidx_current = rowptrs[row];
        auto row_current = rowidxs[rowidx_current];
        auto col_current = colidxs[colidx_current];
        ValueType sum = zero<ValueType>();

        // if(row >= n){
        //     printf("Now performing illegal memory access 5\n");
        // }

        // if(col == 87801){
        //     printf("Col 87801 inital conf row %d: %d %d %d %d %d\n",
        //     (int32)row, (int32)dot_end, (int32)rowidx_current,
        //     (int32)colidx_current, (int32)row_current, (int32)col_current);
        // }

        while (row_current < dot_end && col_current < dot_end &&
               colidx_current < rowptrs[row + 1]) {
            if (row_current < col_current) {
                ++rowidx_current;

                // if(rowidx_current >= nnz){
                //     printf("Now performing illegal memory access 1\n");
                // }

                // if(col == 87801){
                //         printf("Col 87801 incremented rowidx_current, to %d
                //         (%d)\n", (int32)row_current, (int32)rowidx_current);
                // }


                row_current = rowidxs[rowidx_current];
            } else if (col_current < row_current) {
                ++colidx_current;

                // if(colidx_current >= nnz){
                //     printf("Now performing illegal memory access 2\n");
                // }

                // if(col == 87801){
                //         printf("Col 87801 incremented colidx_current, to %d
                //         (%d)\n", (int32)col_current, (int32)colidx_current);
                // }

                col_current = colidxs[colidx_current];
            } else {
                sum += cvalues[rowidx_current] * rvalues[colidx_current];
                ++rowidx_current;

                // if(col == 87801){
                //         printf("Col 87801 added to sum at %d == %d\n",
                //         (int32)col_current, (int32)row_current);
                // }

                // if(rowidx_current >= nnz){
                //     printf("Now performing illegal memory access 3\n");
                // }

                row_current = rowidxs[rowidx_current];
                ++colidx_current;

                // if(colidx_current >= nnz){
                //     printf("Now performing illegal memory access 4\n");
                // }

                col_current = colidxs[colidx_current];
            }
        }

        // if (thrust::abs(sum) >= 1e-8){
        //     printf("Adding something (fact %d - %d) (%d, %d, %d)\n",
        //     (int32)factor_start, (int32)factor_end, (int32)col, (int32)i,
        //     (int32)row);
        // }
        cvalues[i] += -sum;  // add to the -val already there
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_diag_correct_kernel(const IndexType* const colptrs,
                                              const IndexType* const rowidxs,
                                              ValueType* const cvalues,
                                              const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;

    const auto col_diag = colptrs[col];

    // if(rowidxs[col_diag] != col){
    //     printf("ERROR: Non diagonal element oned\n");
    // }

    cvalues[col_diag] = one<ValueType>();
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_solve_factor_csr_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const ValueType* const values, const IndexType factor_start,
    const IndexType factor_end, const ValueType* const b, ValueType* const x,
    const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n - factor_start) {
        return;
    }

    const auto row = factor_start + gid;

    const auto row_start = rowptrs[row];
    const auto row_end = rowptrs[row + 1];

    for (auto i = row_start; i < row_end - 1; ++i) {
        const auto colidx = colidxs[i];

        if (colidx < factor_start || colidx >= factor_end) {
            continue;
        }

        // if(row == 3034){
        //     printf("Adding to 1 (3034) in fact (%d - %d): %lf = %lf * %lf
        //     \n", (int32)factor_start, (int32)factor_end, values[i] *
        //     b[colidx], values[i], b[colidx]);
        // }

        x[row] += values[i] * b[colidx];

        // if(row == 3034){
        //     printf("Gid 1 amounts now to %lf\n", x[row]);
        // }
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_solve_factor_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const cvalues, const IndexType factor_start,
    const IndexType factor_end, const ValueType* const b, ValueType* const x,
    const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= factor_end - factor_start) {
        return;
    }

    const auto col = factor_start + gid;

    const auto col_start = colptrs[col] + 1;
    const auto col_end = colptrs[col + 1];

    const auto b_col = b[col];

    for (auto i = col_start; i < col_end; ++i) {
        const auto rowidx = rowidxs[i];
        atomic_add(x + rowidx, cvalues[i] * b_col);

        // if(rowidx == 27){
        //     printf("Simple kernel adding to 27 in fact (%d - %d): %.9lf =
        //     %.9lf * %.9lf \n", (int32)factor_start, (int32)factor_end,
        //     cvalues[i] * b_col, cvalues[i], b_col);
        // }
        // if(rowidx == 27){
        //     printf("Simple kernel gid 27 amounts now to %.9lf\n", x[rowidx]);
        // }
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvrdpi_solve_factor_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const cvalues, const IndexType factor_start,
    const IndexType factor_end, const ValueType* const b, ValueType* const x,
    const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= factor_end - factor_start) {
        return;
    }

    const auto col = factor_start + gid;

    const auto col_start = colptrs[col] + 1;
    const auto col_end = colptrs[col + 1];

    const auto b_col = b[col];

    for (auto i = col_start; i < col_end; ++i) {
        const auto rowidx = rowidxs[i];

        if (rowidx >= factor_end) {
            break;
        }

        atomic_add(x + rowidx, cvalues[i] * b_col);

        //     if(rowidx == 85087){
        //         printf("Simple kernel adding to 85087 in fact (%d - %d): %lf
        //         = %lf * %lf \n", (int32)factor_start, (int32)factor_end,
        //         cvalues[i] * b_col, cvalues[i], b_col);
        //     }
        //     if(rowidx == 85087){
        //         printf("Simple kernel gid 85087 amounts now to %lf\n",
        //         x[rowidx]);
        // }
    }
}

template <typename ValueType, typename IndexType>
__global__ void sptrsvrdpi_spmv_rect_csc_kernel(const IndexType* const colptrs,
                                                const IndexType* const rowidxs,
                                                const ValueType* const cvalues,
                                                const ValueType* const input,
                                                ValueType* const output,
                                                const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;

    const auto col_start = colptrs[col];
    const auto col_end = colptrs[col + 1];

    const auto b_col = input[col];

    for (auto i = col_start; i < col_end; ++i) {
        const auto rowidx = rowidxs[i];

        // if(rowidx == 0){
        //     printf("rect spmv adding to 83 %lf = %lf * %lf \n", -cvalues[i] *
        //     b_col, -cvalues[i], b_col);
        // }
        // if(rowidx == 0){
        //     printf("rect spmv gid 83 amounts now to %lf\n", output[rowidx]);
        // }

        atomic_add(output + rowidx, -cvalues[i] * b_col);
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvrdpi_spmv_triangle_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const cvalues, const ValueType* const input,
    ValueType* const output, const gko::size_type n)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    const auto col = gid;

    atomic_add(output + col, input[col]);

    const auto col_start = colptrs[col] + 1;
    const auto col_end = colptrs[col + 1];

    const auto b_col = input[col];

    for (auto i = col_start; i < col_end; ++i) {
        const auto rowidx = rowidxs[i];

        // if(rowidx == 0){
        //     printf("rect spmv adding to 83 %lf = %lf * %lf \n", -cvalues[i] *
        //     b_col, -cvalues[i], b_col);
        // }
        // if(rowidx == 0){
        //     printf("rect spmv gid 83 amounts now to %lf\n", output[rowidx]);
        // }

        atomic_add(output + rowidx, cvalues[i] * b_col);
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_solve_factor_batched_csc_kernel(
    const IndexType* const colptrs, const IndexType* const rowidxs,
    const ValueType* const cvalues, const IndexType* const factor_offsets,
    const IndexType factor_batch_start, const IndexType factor_batch_end,
    const ValueType* const input, ValueType* const output,
    int32* const factors_done,  // single counter init to 0
    const gko::size_type n, IndexType m)
{
    const auto thread_id = threadIdx.x;
    const auto block_id = blockIdx.x;

    const auto factor = factor_batch_start + block_id;
    const auto factor_start = factor_offsets[factor];
    auto factor_end = factor < m ? factor_offsets[factor + 1] : n;

    // if(thread_id == 0){
    //     printf("Adv kernel block %d operating for factor %d from %d to %d\n",
    //     (int32)block_id, (int32)factor, (int32)factor_start,
    //     (int32)factor_end);
    // }

    // TODO: Check this
    if (factor_end == 0) {
        factor_end = n;
    }

    if (thread_id >= factor_end - factor_start) {
        return;
    }

    const auto col = factor_start + thread_id;

    const auto col_start = colptrs[col] + 1;
    const auto col_end = colptrs[col + 1];

    // const auto b_col = input[col];

    // Wait for previous block to finish
    if (thread_id == 0) {
        while (load_acquire(factors_done) != block_id) {
            __nanosleep(128);
        }
    }

    __syncthreads();
    // Previous block is now finished and values are visible

    const auto load_output_col = load_relaxed(output + col);

    // this block might below overwrite the values we were just loading,
    // therefore sync
    __syncthreads();

    const auto full_b_col = load_output_col;

    for (auto i = col_start; i < col_end; ++i) {
        const auto rowidx = rowidxs[i];
        atomic_add(output + rowidx, cvalues[i] * full_b_col);

        //     if(rowidx == 85087){
        //         printf("Adding to 85087 from col %d in fact (%d - %d): %lf =
        //         %lf * %lf \n", (int32)col, (int32)factor_start,
        //         (int32)factor_end, cvalues[i] * full_b_col, cvalues[i],
        //         full_b_col);
        //     }
        //     if(rowidx == 85087){
        //         printf("Gid 85087 amounts now to %lf\n", output[rowidx]);
        // }
    }


    // threadfence for visibility
    // (should also be implied by the sync below, though)
    __threadfence();

    __syncthreads();
    if (thread_id == 0) {
        store_release(factors_done, block_id + 1);
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_debug_kernel(const ValueType* const xa,
                                       const ValueType* const xb,
                                       const gko::size_type n, IndexType dummy)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    if (true) {
        if (thrust::abs(xa[gid] - xb[gid]) >= 1e-10) {
            printf("Gid %d disagrees, abs is %.10lf (%.10lf, %.10lf)\n",
                   (int32)gid, thrust::abs(xa[gid] - xb[gid]), xa[gid],
                   xb[gid]);
        } else {
            // printf("Gid %d agrees at %lf \n", (int32)gid, xa[gid]);
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsvppi_debug_bb_kernel(const ValueType* const b,
                                          const ValueType* const new_b,
                                          const gko::size_type n,
                                          IndexType dummy)
{
    const auto gid = thread::get_thread_id_flat();

    if (gid >= n) {
        return;
    }

    if (gid < 96) {
        if (thrust::abs(b[gid] - new_b[gid]) <= 1e-8) {
            printf("Gid %d AGREES FOR B,NEW_B, abs is %lf (%lf, %lf)\n",
                   (int32)gid, thrust::abs(b[gid] - new_b[gid]), b[gid],
                   new_b[gid]);
        }
    }
}


// ANOTHER EXPERIMENTAL BELOW

constexpr int ppi_batch_bocksize = 128;

// ppi for partitioned product inverse
template <typename ValueType, typename IndexType>
struct SptrsvppiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> unit_inv_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_transp;


    std::unique_ptr<array<IndexType>> factor_offsets;
    std::unique_ptr<array<IndexType>> factor_offsets_d;
    std::unique_ptr<array<IndexType>> factor_assignments;
    std::unique_ptr<matrix::Permutation<IndexType>> factor_perm;

    IndexType m;


    SptrsvppiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* matrix,
                         size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}
    {
        scaled_p_m = matrix::Csr<ValueType, IndexType>::create(exec);
        scaled_p_m->copy_from(matrix);

        if (!unit_diag) {
            diag->inverse_apply(matrix, scaled_p_m.get());
        }

        scaled_transp =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());

        const auto n = matrix->get_size()[0];
        const auto nnz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        generate_factormin_perm(exec, matrix, scaled_transp.get());
        scaled_transp.reset();


        scaled_p_m = scaled_p_m->permute(factor_perm);

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("pkust11.input.ppi.mtx");
        // gko::write(output, scaled_p_m);
        // output.close();


        unit_inv_p_m =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const dim3 block_size_0(default_block_size, 1, 1);
        const dim3 grid_size_0(ceildiv(nnz, block_size_0.x), 1, 1);
        sptrsvppi_negate_ppi_kernel<<<grid_size_0, block_size_0>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()), nnz);


        const dim3 block_size_1(default_block_size, 1, 1);
        const dim3 grid_size_1(ceildiv(n, block_size_1.x), 1, 1);
        sptrsvppi_diag_correct_kernel<<<grid_size_1, block_size_1>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()), n);


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(n, block_size_2.x), 1, 1);
        sptrsvppi_invert_ppi_kernel<<<grid_size_2, block_size_2>>>(
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            as_cuda_type(scaled_p_m->get_values()),
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n, factor_offsets->get_size(),
            nnz);

        // FIXME here for csr/csc swap
        // unit_inv_p_m = gko::as<matrix::Csr<ValueType,
        // IndexType>>(unit_inv_p_m->transpose());
    }

    void generate_factormin_perm(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const matrix::Csr<ValueType, IndexType>* t_matrix)
    {
        // TODO
        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();

        // This should be minimized
        // m = (IndexType)n + 1;


        // // TEMP
        // array<IndexType> levels(exec, n);
        // cudaMemset(levels.get_data(), 0xFF, n * sizeof(IndexType));

        // array<IndexType> height_d(exec, 1);
        // cudaMemset(height_d.get_data(), 0, sizeof(IndexType));

        // array<IndexType> atomic_counter(exec, 1);
        // cudaMemset(atomic_counter.get_data(), 0, sizeof(IndexType));

        // const auto block_size = default_block_size;
        // const auto block_count = (n + block_size - 1) / block_size;


        // level_generation_kernel<IndexType, false>
        //     <<<block_count, block_size>>>(
        //         matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
        //         levels.get_data(), height_d.get_data(), n,
        //         atomic_counter.get_data());


        // const auto height = exec->copy_val_to_host(height_d.get_const_data())
        // + 1;

        // array<IndexType> level_counts(exec, height);
        // cudaMemset(level_counts.get_data(), 0, height * sizeof(IndexType));

        // array<IndexType> lperm(exec, n);

        // sptrsv_level_counts_kernel<<<block_count, block_size>>>(
        //     levels.get_const_data(), level_counts.get_data(),
        //     lperm.get_data(), (IndexType)n);

        // TEMP ABVOVE

        // m = height;

        array<IndexType> purely_diagonal_elements(exec->get_master(), n);
        array<IndexType> purely_diagonal_count(exec->get_master(), 1);
        purely_diagonal_count.fill(zero<IndexType>());

        auto matrix_cpu =
            matrix::Csr<ValueType, IndexType>::create(exec->get_master());
        matrix_cpu->copy_from(matrix);
        auto t_matrix_cpu =
            matrix::Csr<ValueType, IndexType>::create(exec->get_master());
        t_matrix_cpu->copy_from(t_matrix);

        std::vector<std::unordered_set<int>> rem_preds(n);

        for (auto row = 0; row < n; ++row) {
            const auto row_start = matrix_cpu->get_const_row_ptrs()[row];
            const auto row_end = matrix_cpu->get_const_row_ptrs()[row + 1] - 1;
            for (auto i = row_start; i < row_end; ++i) {
                const auto colidx = matrix_cpu->get_const_col_idxs()[i];
                rem_preds[row].insert(colidx);
            }

            if (row_start >= row_end) {
                purely_diagonal_elements
                    .get_data()[purely_diagonal_count.get_const_data()[0]] =
                    row;
                *purely_diagonal_count.get_data() += 1;
            }
        }

        array<IndexType> counts(exec->get_master(), n);
        for (auto counts_i = 0; counts_i < n; ++counts_i) {
            counts.get_data()[counts_i] = rem_preds[counts_i].size();
        }


        array<IndexType> factor_sizes(exec->get_master(), n);
        array<IndexType> factor_assignments_(exec->get_master(), n);
        factor_sizes.fill(zero<IndexType>());
        factor_assignments_.fill(-one<IndexType>());

        sptrsvppi_create_ppi_cpukernel(
            exec, matrix_cpu->get_const_row_ptrs(),
            matrix_cpu->get_const_col_idxs(),
            t_matrix_cpu->get_const_row_ptrs(),
            t_matrix_cpu->get_const_col_idxs(), factor_sizes.get_data(),
            factor_assignments_.get_data(),
            purely_diagonal_elements.get_const_data(),
            purely_diagonal_count.get_const_data()[0], &rem_preds[0],
            counts.get_data(), (IndexType)n, (IndexType)nz, &m,
            one<ValueType>());

        const auto dbg_tmp_453 = 0;

        factor_sizes.set_executor(exec);
        factor_assignments_.set_executor(exec);


        components::prefix_sum_nonnegative(exec, factor_sizes.get_data(), m);


        array<IndexType> perm(exec, n);
        thrust::sequence(thrust::device, perm.get_data(), perm.get_data() + n);
        // thrust::stable_sort_by_key(thrust::device, levels.get_data(),
        // levels.get_data() + n, level_perm.get_data());
        thrust::stable_sort_by_key(
            thrust::device, factor_assignments_.get_data(),
            factor_assignments_.get_data() + n, perm.get_data());


        factor_perm = matrix::Permutation<IndexType>::create(exec, perm);
        factor_offsets =
            std::make_unique<array<IndexType>>(std::move(factor_sizes));
        factor_assignments =
            std::make_unique<array<IndexType>>(std::move(factor_assignments_));

        // factor_offsets_d = std::make_unique<array<IndexType>>(exec);
        // factor_offsets_d->resize_and_reset(m + 1);
        // cudaMemcpy(factor_offsets_d->get_data(),
        // factor_assignments->get_const_data(), (m + 1) * sizeof(IndexType),
        // cudaMemcpyDeviceToDevice);
        // factor_offsets->set_executor(exec->get_master());
        // factor_offsets_d->set_executor(exec);

        // INFO
        // printf("PPi m is %d\n", (int32)m);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        auto new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());
        new_b = new_b->permute(factor_perm, matrix::permute_mode::rows);

        // const dim3 block_size_2(default_block_size, 1,1);
        // const dim3 grid_size_2(ceildiv(n * nrhs, block_size_2.x), 1, 1);
        // // DEBUG
        // sptrsvppi_debug_bb_kernel<<<grid_size_2,block_size_2>>>(
        //     as_cuda_type(b->get_const_values()),
        //     as_cuda_type(new_b->get_const_values()), n, one<IndexType>());


        factor_offsets->set_executor(exec->get_master());

        cudaMemcpy(x->get_values(), new_b->get_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);


        // IndexType count_solves_batched = 0;
        // IndexType i = 0;
        // while (i < m){
        //     const auto factor_start = factor_offsets->get_const_data()[i];
        //     const auto factor_end = i < m ?
        //     factor_offsets->get_const_data()[i + 1] : (IndexType)n;

        //     const auto is_batch_eligible = factor_end - factor_start <= 512;
        //     const auto is_batch_end = !is_batch_eligible || (i == m - 1);

        //     if(is_batch_eligible && !is_batch_end){
        //         count_solves_batched += 1;
        //         i += 1;
        //         continue;
        //     }

        //     else if(is_batch_eligible && is_batch_end){

        //         const auto batch_start = i - count_solves_batched;
        //         const auto batch_end = batch_start + count_solves_batched;

        //         array<int32> factors_done(exec, 1);
        //         factors_done.fill(zero<int32>());

        //         factor_offsets->set_executor(exec);

        //         const dim3 block_size_0(default_block_size, 1, 1);
        //         const dim3 grid_size_0(batch_end - batch_start + 1, 1, 1);
        //         sptrsvppi_solve_factor_batched_csc_kernel<<<grid_size_0,
        //         block_size_0>>>(
        //             unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_const_values()),
        //                 factor_offsets->get_const_data(),
        //                 batch_start,
        //                 batch_end,
        //                 as_cuda_type(x->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 factors_done.get_data(), (gko::size_type)n,
        //                 (IndexType)m);

        //         factor_offsets->set_executor(exec->get_master());

        //     }

        //     else if(!is_batch_eligible && is_batch_end){

        //         if(count_solves_batched > 0){

        //             const auto batch_start = i - count_solves_batched;
        //             const auto batch_end = batch_start + count_solves_batched
        //             - 1;

        //             array<int32> factors_done(exec, 1);
        //             factors_done.fill(zero<int32>());

        //             factor_offsets->set_executor(exec);

        //             const dim3 block_size_0(default_block_size, 1, 1);
        //             const dim3 grid_size_0(batch_end - batch_start + 1, 1,
        //             1);
        //             sptrsvppi_solve_factor_batched_csc_kernel<<<grid_size_0,
        //             block_size_0>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_const_values()),
        //                 factor_offsets->get_const_data(),
        //                 batch_start,
        //                 batch_end,
        //                 as_cuda_type(x->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 factors_done.get_data(), (gko::size_type)n,
        //                 (IndexType)m);

        //             factor_offsets->set_executor(exec->get_master());


        //             cudaMemcpy(new_b->get_values(), x->get_const_values(), n
        //             * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //             cudaMemcpyDeviceToDevice);

        //         }


        //         const dim3 block_size(default_block_size, 1, 1);
        //         const dim3 grid_size(ceildiv(factor_end - factor_start,
        //         block_size.x), 1, 1); // FIXME here for csr/csc swap
        //         sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_values()),
        //                 factor_start,
        //                 factor_end,
        //                 as_cuda_type(new_b->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 n);

        //                 // const auto one_val_2 =
        //                 exec->copy_val_to_host(x->get_values() + 2);
        //                 // printf("VAL at in x 2 at point 2: %lf\n",
        //                 one_val_2);

        //         const auto next_factor_start = i < m ?
        //         factor_offsets->get_const_data()[i + 1] : (IndexType)n; const
        //         auto next_factor_end = i < m - 1 ?
        //         factor_offsets->get_const_data()[i + 2] : (IndexType)n;

        //         if(next_factor_end - next_factor_start <= 512){
        //             cudaMemcpy(new_b->get_values() + next_factor_start,
        //             x->get_const_values() + next_factor_start, (n -
        //             next_factor_start) *
        //             sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //             cudaMemcpyDeviceToDevice);

        //         }else{
        //             cudaMemcpy(new_b->get_values() + next_factor_start,
        //             x->get_const_values() + next_factor_start,
        //             (next_factor_end - next_factor_start) *
        //             sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //             cudaMemcpyDeviceToDevice);
        //         }

        //     }
        //     else {

        //         const dim3 block_size(default_block_size, 1, 1);
        //         const dim3 grid_size(ceildiv(factor_end - factor_start,
        //         block_size.x), 1, 1); // FIXME here for csr/csc swap
        //         sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_values()),
        //                 factor_start,
        //                 factor_end,
        //                 as_cuda_type(new_b->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 n);

        //                 // const auto one_val_2 =
        //                 exec->copy_val_to_host(x->get_values() + 2);
        //                 // printf("VAL at in x 2 at point 2: %lf\n",
        //                 one_val_2);

        //         const auto next_factor_start = i < m ?
        //         factor_offsets->get_const_data()[i + 1] : (IndexType)n; const
        //         auto next_factor_end = i < m - 1 ?
        //         factor_offsets->get_const_data()[i + 2] : (IndexType)n;

        //         if(next_factor_end - next_factor_start <= 512){
        //             cudaMemcpy(new_b->get_values() + next_factor_start,
        //             x->get_const_values() + next_factor_start, (n -
        //             next_factor_start) *
        //             sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //             cudaMemcpyDeviceToDevice);

        //         }else{
        //             cudaMemcpy(new_b->get_values() + next_factor_start,
        //             x->get_const_values() + next_factor_start,
        //             (next_factor_end - next_factor_start) *
        //             sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //             cudaMemcpyDeviceToDevice);
        //         }
        //     }

        //     count_solves_batched = 0;
        //     i += 1;
        // }

        // const auto final_x = x->permute(factor_perm,
        // matrix::permute_mode::inverse_rows); cudaMemcpy(x->get_values(),
        // final_x->get_const_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);

        auto batch_instructions = std::vector<int32>();
        for (auto i = 0; i < m - 1; ++i) {
            const auto factor_start = factor_offsets->get_const_data()[i];
            const auto factor_end = factor_offsets->get_const_data()[i + 1];

            const auto is_batch_eligible =
                false;  // factor_end - factor_start <= ppi_batch_bocksize;

            if (is_batch_eligible) {
                if (batch_instructions.empty() ||
                    batch_instructions.back() == 0) {
                    batch_instructions.push_back(1);
                } else {
                    batch_instructions.back() += 1;
                }
            } else {
                if (!batch_instructions.empty() &&
                    batch_instructions.back() == 1) {
                    batch_instructions.back() -=
                        1;  // singular batches are re-transformed to plain
                            // spmvs
                }
                batch_instructions.push_back(0);
            }
        }


        const auto reduced_m = batch_instructions.size();
        IndexType start_factor_i = 0;
        for (auto i = 0; i < reduced_m; ++i) {
            const auto batch_instr = batch_instructions[i];

            const IndexType end_factor_i = batch_instr == 0
                                               ? start_factor_i + 1
                                               : start_factor_i + batch_instr;

            const auto batch_start =
                factor_offsets->get_const_data()[start_factor_i];
            const auto batch_end =
                end_factor_i < m
                    ? factor_offsets->get_const_data()[end_factor_i]
                    : (IndexType)n;


            if (batch_instr == 0) {
                const dim3 block_size(default_block_size, 1, 1);
                const dim3 grid_size(
                    ceildiv(batch_end - batch_start, block_size.x), 1,
                    1);  // FIXME here for csr/csc swap
                sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
                    unit_inv_p_m->get_const_row_ptrs(),
                    unit_inv_p_m->get_const_col_idxs(),
                    as_cuda_type(unit_inv_p_m->get_values()), batch_start,
                    batch_end, as_cuda_type(new_b->get_const_values()),
                    as_cuda_type(x->get_values()), n);

                // const auto one_val_2 = exec->copy_val_to_host(x->get_values()
                // + 2); printf("VAL at in x 2 at point 2: %lf\n", one_val_2);


                if (batch_end < n) {
                    const auto next_start_factor_i = end_factor_i;
                    const auto next_end_factor_i =
                        batch_instructions[i + 1] == 0
                            ? next_start_factor_i + 1
                            : next_start_factor_i + batch_instructions[i + 1];
                    const auto next_batch_start =
                        factor_offsets->get_const_data()[next_start_factor_i];
                    const auto next_batch_end =
                        next_end_factor_i < m
                            ? factor_offsets
                                  ->get_const_data()[next_end_factor_i]
                            : (IndexType)n;
                    if (next_end_factor_i - next_start_factor_i == 1) {
                        cudaMemcpy(new_b->get_values() + next_batch_start,
                                   x->get_const_values() + next_batch_start,
                                   (next_batch_end - next_batch_start) *
                                       sizeof(decltype(as_cuda_type(
                                           zero<ValueType>()))),
                                   cudaMemcpyDeviceToDevice);
                    }
                }

            } else {
                array<int32> factors_done(exec, 1);
                factors_done.fill(zero<int32>());
                factor_offsets->set_executor(exec);

                const dim3 block_size_0(ppi_batch_bocksize, 1, 1);
                const dim3 grid_size_0(end_factor_i - start_factor_i, 1, 1);
                sptrsvppi_solve_factor_batched_csc_kernel<<<grid_size_0,
                                                            block_size_0>>>(
                    unit_inv_p_m->get_const_row_ptrs(),
                    unit_inv_p_m->get_const_col_idxs(),
                    as_cuda_type(unit_inv_p_m->get_const_values()),
                    factor_offsets->get_const_data(), start_factor_i,
                    end_factor_i, as_cuda_type(x->get_const_values()),
                    as_cuda_type(x->get_values()), factors_done.get_data(),
                    (gko::size_type)n, (IndexType)m);

                factor_offsets->set_executor(exec->get_master());

                if (batch_end < n) {
                    const auto next_start_factor_i = end_factor_i;
                    const auto next_end_factor_i =
                        batch_instructions[i + 1] == 0
                            ? next_start_factor_i + 1
                            : next_start_factor_i + batch_instructions[i + 1];
                    const auto next_batch_start =
                        factor_offsets->get_const_data()[next_start_factor_i];
                    const auto next_batch_end =
                        next_end_factor_i < m
                            ? factor_offsets
                                  ->get_const_data()[next_end_factor_i]
                            : (IndexType)n;

                    if (next_end_factor_i - next_start_factor_i == 1) {
                        cudaMemcpy(new_b->get_values() + next_batch_start,
                                   x->get_const_values() + next_batch_start,
                                   (next_batch_end - next_batch_start) *
                                       sizeof(decltype(as_cuda_type(
                                           zero<ValueType>()))),
                                   cudaMemcpyDeviceToDevice);
                    }
                }
            }


            start_factor_i = end_factor_i;
        }

        const auto final_x =
            x->permute(factor_perm, matrix::permute_mode::inverse_rows);
        cudaMemcpy(x->get_values(), final_x->get_const_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);

        // IDEA
        // IDEA
        // IDEA
        // This product decomposition is also possible for row matrices
        //
        //             | 1 0 0 |   | 1 0 0 |   | 1 0 0 |
        // Meaning A = | d 1 0 | = | d 1 0 | * | 0 1 0 |
        //             | g h 1 |   | 0 0 1 |   | g h 1 |
        //
        // This immdeaitely begs the question: can we do a factorization of ther
        // form


        // WORKING BACKUP BELOW

        // for(auto i = 0; i < m; ++i){

        //         const auto factor_start =
        //         factor_offsets->get_const_data()[i]; const auto factor_end =
        //         i < m ? factor_offsets->get_const_data()[i + 1] :
        //         (IndexType)n;

        //         const dim3 block_size(default_block_size, 1, 1);
        //         const dim3 grid_size(ceildiv(factor_end - factor_start,
        //         block_size.x), 1, 1); // FIXME here for csr/csc swap
        //         sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_values()),
        //                 factor_start,
        //                 factor_end,
        //                 as_cuda_type(new_b->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 n);

        //                 // const auto one_val_2 =
        //                 exec->copy_val_to_host(x->get_values() + 2);
        //                 // printf("VAL at in x 2 at point 2: %lf\n",
        //                 one_val_2);

        //         const auto next_factor_start = i < m ?
        //         factor_offsets->get_const_data()[i + 1] : (IndexType)n; const
        //         auto next_factor_end = i < m - 1 ?
        //         factor_offsets->get_const_data()[i + 2] : (IndexType)n;

        //         cudaMemcpy(new_b->get_values() + next_factor_start,
        //         x->get_const_values() + next_factor_start, (next_factor_end -
        //         next_factor_start) *
        //         sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //         cudaMemcpyDeviceToDevice);
        // }


        // const auto final_x = x->permute(factor_perm,
        // matrix::permute_mode::inverse_rows); cudaMemcpy(x->get_values(),
        // final_x->get_const_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);


        // WORKING BACVKUP ABOVE


        // INFO
        // printf("%d/%d solves were small\n", (int32)count_small_solves,
        // (int32)m);


        // DEBUG BELOW

        //     // Initialize x to all NaNs.
        //     const auto clone_x = matrix::Dense<ValueType>::create(exec);
        //     clone_x->copy_from(b);
        //     dense::fill(exec, clone_x.get(), nan<ValueType>());

        //     array<bool> nan_produced(exec, 1);
        //     array<IndexType> atomic_counter(exec, 1);
        //     sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
        //                                  atomic_counter.get_data());

        //    const dim3 block_size_3(default_block_size, 1, 1);
        //             const dim3 grid_size_3(ceildiv(n, block_size_3.x), 1, 1);
        //     sptrsv_naive_caching_kernel<false><<<grid_size_3,block_size_3>>>(
        //         matrix->get_const_row_ptrs(),
        //         matrix->get_const_col_idxs(),
        //         as_cuda_type(matrix->get_const_values()),
        //         as_cuda_type(b->get_const_values()), b->get_stride(),
        //         as_cuda_type(clone_x->get_values()), clone_x->get_stride(),
        //         n, nrhs, false, nan_produced.get_data(),
        //         atomic_counter.get_data());

        //     const auto clone_final_x = clone_x->permute(factor_perm,
        //     matrix::permute_mode::rows); const auto computed_final_x =
        //     x->permute(factor_perm, matrix::permute_mode::rows);

        //     const auto dbg_tmp_0 = 0;
        //     sptrsvppi_debug_kernel<<<grid_size_3,block_size_3>>>(
        //         as_cuda_type(clone_final_x->get_const_values()),
        //         as_cuda_type(computed_final_x->get_const_values()), n,
        //         one<IndexType>());


        // DEBUG ABOVE
    }
};


// rdpi for recursive delayed product inverse
template <typename ValueType, typename IndexType>
struct SptrsvrdpiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> unit_inv_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_transp;


    std::unique_ptr<array<IndexType>> factor_offsets;
    std::unique_ptr<array<IndexType>> factor_assignments;
    std::unique_ptr<matrix::Permutation<IndexType>> factor_perm;

    std::unique_ptr<matrix::Dense<ValueType>> m_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_neg_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_zero;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> rblocks;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> cblocks;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> rtriangles;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> ctriangles;

    // TODO: Implement an algorithm where not factor_offsets, but these offsets
    // define the triangle borders Then you can do bigger triangular solves
    // including rectanuglar parts do this in csr format, probably
    std::vector<std::vector<IndexType>> rtriangles_offsets;

    IndexType m;


    SptrsvrdpiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}
    {
        const auto n = matrix->get_size()[0];

        scaled_p_m =
            matrix::Csr<ValueType, IndexType>::create(exec, gko::dim<2>(n, n));
        // scaled_p_m->copy_from(matrix);

        if (!unit_diag) {
            diag->inverse_apply(matrix, scaled_p_m.get());
        }

        // std::basic_ofstream<char> output;
        // output.open("stencil9pt.mtx");
        // gko::write(output, gko::clone(exec, matrix));
        // output.close();

        scaled_transp =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const auto nnz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        generate_factormin_perm(exec, matrix, scaled_transp.get());


        scaled_p_m = scaled_p_m->permute(factor_perm);

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("stencil9pt.mtx.rcm.rdpi");
        // gko::write(output, scaled_p_m);
        // output.close();


        unit_inv_p_m =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const dim3 block_size_0(default_block_size, 1, 1);
        const dim3 grid_size_0(ceildiv(n, block_size_0.x), 1, 1);
        sptrsvppi_negate_rdpi_kernel<<<grid_size_0, block_size_0>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n);


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(n, block_size_2.x), 1, 1);
        sptrsvppi_invert_rdpi_kernel<<<grid_size_2, block_size_2>>>(
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            as_cuda_type(scaled_p_m->get_values()),
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n, factor_offsets->get_size(),
            nnz);

        factor_offsets->set_executor(exec->get_master());

        // INFO
        // for(auto i = 0; i < m; ++i){
        //     printf(" %d ", (int32)factor_offsets->get_const_data()[i]);
        // }
        // printf("\n");

        // rtriangles are now in csr
        unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
            unit_inv_p_m->transpose());


        rtriangles = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 1);
        ctriangles = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 1);

        for (auto i = 0; i < m - 1; ++i) {
            // printf("creating triangle from %d to %d\n",
            // (int32)factor_offsets->get_const_data()[i],
            // (int32)factor_offsets->get_const_data()[i + 1]);

            rtriangles[i] = unit_inv_p_m->create_submatrix(
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]},
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]});
            rtriangles[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }

        // ctriangles are now in csc
        unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
            unit_inv_p_m->transpose());

        for (auto i = 0; i < m - 1; ++i) {
            // printf("creating triangle from %d to %d\n",
            // (int32)factor_offsets->get_const_data()[i],
            // (int32)factor_offsets->get_const_data()[i + 1]);

            ctriangles[i] = unit_inv_p_m->create_submatrix(
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]},
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]});
            ctriangles[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }


        rblocks = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 2);
        cblocks = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 2);

        auto block_coords =
            std::vector<std::pair<std::pair<IndexType, IndexType>,
                                  std::pair<IndexType, IndexType>>>();
        block_coords.push_back(
            std::make_pair(std::make_pair(0, n), std::make_pair(0, n)));

        do {
            const auto popped = block_coords.back();
            block_coords.pop_back();

            const IndexType from_row = popped.first.first;
            const IndexType to_row = popped.first.second;
            const IndexType from_col = popped.second.first;
            const IndexType to_col = popped.second.second;

            const auto row_dist = to_row - from_row;
            const auto col_dist = to_col - from_col;

            auto eligible_offsets = std::vector<IndexType>();
            for (auto offset_i = 0; offset_i < m; ++offset_i) {
                const auto offset = factor_offsets->get_const_data()[offset_i];
                if (offset >= from_col && offset <= to_col) {
                    eligible_offsets.push_back(offset);
                }
            }

            if (eligible_offsets.size() <= 2) {
                continue;
            }

            auto closest_offset = std::min_element(
                eligible_offsets.begin(), eligible_offsets.end(),
                [&](IndexType off_a, IndexType off_b) {
                    return std::abs((float)from_col + (float)col_dist / 2 -
                                    (float)off_a) <
                           std::abs((float)from_col + (float)col_dist / 2 -
                                    (float)off_b);
                });
            const auto closest_offset_i =
                std::distance(factor_offsets->get_const_data(),
                              std::find(factor_offsets->get_const_data(),
                                        factor_offsets->get_const_data() + m,
                                        *closest_offset));

            // printf("Added rect %d %d %d %d at pos %d\n",
            // (int32)*closest_offset, (int32)to_row, (int32)from_col,
            // (int32)*closest_offset, (int32)closest_offset_i);

            block_coords.push_back(
                std::make_pair(std::make_pair(from_row, *closest_offset),
                               std::make_pair(from_col, *closest_offset)));
            block_coords.push_back(
                std::make_pair(std::make_pair(*closest_offset, to_row),
                               std::make_pair(*closest_offset, to_col)));

            // when using scaled_p_m here, it's csr
            // if you want csc, use unit_inv_p_m

            if (to_row - *closest_offset >=
                (5 * (*closest_offset - from_col)) / 10) {
                cblocks[closest_offset_i - 1] = unit_inv_p_m->create_submatrix(
                    span{from_col, *closest_offset},
                    span{*closest_offset, to_row});
                cblocks[closest_offset_i - 1]->set_strategy(
                    std::make_shared<typename matrix::Csr<
                        ValueType, IndexType>::automatical>(exec));
            } else {
                rblocks[closest_offset_i - 1] = scaled_p_m->create_submatrix(
                    span{*closest_offset, to_row},
                    span{from_col, *closest_offset});
                rblocks[closest_offset_i - 1]->set_strategy(
                    std::make_shared<typename matrix::Csr<
                        ValueType, IndexType>::automatical>(exec));
            }

        } while (!block_coords.empty());


        // FREE
        scaled_p_m.reset();
        unit_inv_p_m.reset();

        m_neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
        m_one = gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
        m_zero = gko::initialize<gko::matrix::Dense<ValueType>>({0}, exec);
    }

    void generate_factormin_perm(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const matrix::Csr<ValueType, IndexType>* t_matrix)
    {
        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        // array<IndexType> levels(exec, n);
        // cudaMemset(levels.get_data(), 0xFF, n * sizeof(IndexType));

        // array<IndexType> height_d(exec, 1);
        // cudaMemset(height_d.get_data(), 0, sizeof(IndexType));

        // array<IndexType> atomic_counter(exec, 1);
        // cudaMemset(atomic_counter.get_data(), 0, sizeof(IndexType));

        // const auto block_size = default_block_size;
        // const auto block_count = (n + block_size - 1) / block_size;


        // _sptrsvrdpi_level_generation_kernel<IndexType, false>
        //     <<<block_count, block_size>>>(
        //         matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
        //         levels.get_data(), height_d.get_data(), n,
        //         atomic_counter.get_data());

        // levels.set_executor(exec->get_master());


        // array<IndexType> purely_diagonal_elements(exec->get_master(), n);
        // array<IndexType> purely_diagonal_count(exec->get_master(), 1);
        // purely_diagonal_count.fill(zero<IndexType>());

        // auto matrix_cpu = matrix::Csr<ValueType,
        // IndexType>::create(exec->get_master());
        // matrix_cpu->copy_from(matrix);
        // auto t_matrix_cpu = matrix::Csr<ValueType,
        // IndexType>::create(exec->get_master());
        // t_matrix_cpu->copy_from(t_matrix);

        // std::vector<std::unordered_set<int>> rem_preds(n);

        // for(auto row = 0; row < n; ++row){
        //     const auto row_start = matrix_cpu->get_const_row_ptrs()[row];
        //     const auto row_end = matrix_cpu->get_const_row_ptrs()[row + 1] -
        //     1; for(auto i = row_start; i < row_end; ++i){
        //         const auto colidx = matrix_cpu->get_const_col_idxs()[i];
        //         rem_preds[row].insert(colidx);
        //     }

        //     if (row_start >= row_end){
        //         purely_diagonal_elements.get_data()[purely_diagonal_count.get_const_data()[0]]
        //         = row; *purely_diagonal_count.get_data() += 1;
        //     }
        // }


        // array<IndexType> counts(exec->get_master(), n);
        // for(auto counts_i = 0; counts_i < n; ++counts_i){
        //     counts.get_data()[counts_i] = rem_preds[counts_i].size();
        // }


        array<IndexType> factor_sizes(exec, n);
        array<IndexType> factor_assignments_(exec, n);
        cudaMemset(factor_sizes.get_data(), 0, n * sizeof(IndexType));
        cudaMemset(factor_assignments_.get_data(), 0xFF, n * sizeof(IndexType));
        // factor_sizes.fill(zero<IndexType>());
        // factor_assignments_.fill(-one<IndexType>());

        sptrsvdrpi_create_ppi(
            exec, matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            t_matrix->get_const_row_ptrs(), t_matrix->get_const_col_idxs(),
            factor_sizes.get_data(), factor_assignments_.get_data(),
            // purely_diagonal_elements.get_const_data(),
            // purely_diagonal_count.get_const_data()[0],
            // &rem_preds[0],
            // counts.get_data(),
            (IndexType)n,
            // (IndexType)nz,
            &m, one<ValueType>());

        const auto dbg_tmp_453 = 0;

        // factor_sizes.set_executor(exec);
        // factor_assignments_.set_executor(exec);


        components::prefix_sum_nonnegative(exec, factor_sizes.get_data(), m);


        array<IndexType> perm(exec, n);
        thrust::sequence(thrust::device, perm.get_data(), perm.get_data() + n);
        // thrust::stable_sort_by_key(thrust::device, levels.get_data(),
        // levels.get_data() + n, level_perm.get_data());
        thrust::stable_sort_by_key(
            thrust::device, factor_assignments_.get_data(),
            factor_assignments_.get_data() + n, perm.get_data());


        factor_perm = matrix::Permutation<IndexType>::create(exec, perm);
        factor_offsets =
            std::make_unique<array<IndexType>>(std::move(factor_sizes));
        factor_assignments =
            std::make_unique<array<IndexType>>(std::move(factor_assignments_));

        // factor_offsets_d = std::make_unique<array<IndexType>>(exec);
        // factor_offsets_d->resize_and_reset(m + 1);
        // cudaMemcpy(factor_offsets_d->get_data(),
        // factor_assignments->get_const_data(), (m + 1) * sizeof(IndexType),
        // cudaMemcpyDeviceToDevice);
        // factor_offsets->set_executor(exec->get_master());
        // factor_offsets_d->set_executor(exec);

        // INFO
        printf("PPi m is %d\n", (int32)m);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        auto new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());
        new_b = new_b->permute(factor_perm, matrix::permute_mode::rows);

        // const dim3 block_size_2(default_block_size, 1,1);
        // const dim3 grid_size_2(ceildiv(n * nrhs, block_size_2.x), 1, 1);
        // // DEBUG
        // sptrsvppi_debug_bb_kernel<<<grid_size_2,block_size_2>>>(
        //     as_cuda_type(b->get_const_values()),
        //     as_cuda_type(new_b->get_const_values()), n, one<IndexType>());


        // WIP BELOW

        factor_offsets->set_executor(exec->get_master());

        // cudaMemcpy(x->get_values(), new_b->get_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);
        cudaMemset(x->get_values(), 0,
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))));


        // auto batch_instructions = std::vector<int32>();
        // for(auto i = 0; i < m - 1; ++i){
        //     const auto factor_start = factor_offsets->get_const_data()[i];
        //     const auto factor_end = factor_offsets->get_const_data()[i + 1];

        //     const auto is_batch_eligible = false; // factor_end -
        //     factor_start <= ppi_batch_bocksize;

        //     if(is_batch_eligible){
        //         if(batch_instructions.empty() || batch_instructions.back() ==
        //         0){
        //             batch_instructions.push_back(1);
        //         }else{
        //             batch_instructions.back() += 1;
        //         }
        //     }else{
        //         if(!batch_instructions.empty() && batch_instructions.back()
        //         == 1){
        //             batch_instructions.back() -= 1; // singular batches are
        //             re-transformed to plain spmvs
        //         }
        //         batch_instructions.push_back(0);
        //     }
        // }

        // batched execution is disabled for now, though it is not commented out below


        const auto reduced_m = m - 1;
        IndexType start_factor_i = 0;
        for (auto i = 0; i < reduced_m; ++i) {
            const IndexType end_factor_i = start_factor_i + 1;

            const auto batch_start =
                factor_offsets->get_const_data()[start_factor_i];
            const auto batch_end =
                end_factor_i < m
                    ? factor_offsets->get_const_data()[end_factor_i]
                    : (IndexType)n;


            {
                const auto xv = x->create_submatrix(
                    span{batch_start, batch_end}, span{0, 1});
                auto bv = new_b->create_submatrix(span{batch_start, batch_end},
                                                  span{0, 1});

                const auto is_csc_triangle = true;
                if (is_csc_triangle) {
                    // cudaMemcpy(x->get_values() + batch_start,
                    // new_b->get_const_values() + batch_start, (batch_end -
                    // batch_start) * sizeof(ValueType),
                    // cudaMemcpyDeviceToDevice);
                    const dim3 block_size_8(default_block_size, 1, 1);
                    const dim3 grid_size_8(
                        ceildiv(batch_end - batch_start, default_block_size), 1,
                        1);
                    sptrsvrdpi_spmv_triangle_csc_kernel<<<grid_size_8,
                                                          block_size_8>>>(
                        ctriangles[i]->get_const_row_ptrs(),
                        ctriangles[i]->get_const_col_idxs(),
                        as_cuda_type(ctriangles[i]->get_const_values()),
                        as_cuda_type(bv->get_const_values()),
                        as_cuda_type(xv->get_values()),
                        batch_end - batch_start);
                } else {
                    rtriangles[i]->apply(m_one.get(), bv.get(), m_zero.get(),
                                         xv.get());
                }
            }


            // const auto one_val_2 = exec->copy_val_to_host(x->get_values() +
            // 2); printf("VAL at in x 2 at point 2: %lf\n", one_val_2);


            if (batch_end < n) {
                const auto next_start_factor_i = end_factor_i;
                const auto next_end_factor_i = next_start_factor_i + 1;
                const auto next_batch_start =
                    factor_offsets->get_const_data()[next_start_factor_i];
                const auto next_batch_end =
                    next_end_factor_i < m
                        ? factor_offsets->get_const_data()[next_end_factor_i]
                        : (IndexType)n;

                // const auto xv = x->create_submatrix(span{batch_start,
                // batch_end}, span{0, 1}); auto bv =
                // new_b->create_submatrix(span{batch_end, n}, span{0, 1});

                const auto is_csc_block = rblocks[i].get() == NULL;

                const auto xl = is_csc_block ? cblocks[i]->get_size()[0]
                                             : rblocks[i]->get_size()[1];
                const auto bl = is_csc_block ? cblocks[i]->get_size()[1]
                                             : rblocks[i]->get_size()[0];

                const auto xv = x->create_submatrix(
                    span{batch_end - xl, batch_end}, span{0, 1});
                auto bv = new_b->create_submatrix(
                    span{batch_end, batch_end + bl}, span{0, 1});

                // printf("bv[27] at 1: %.9lf\n",
                // exec->copy_val_to_host(new_b->get_const_values() +27));
                // printf("xv[27] at 1: %.9lf\n",
                // exec->copy_val_to_host(x->get_const_values() +27));

                if (is_csc_block) {
                    const dim3 block_size_7(default_block_size, 1, 1);
                    const dim3 grid_size_7(ceildiv(xl, default_block_size), 1,
                                           1);
                    sptrsvrdpi_spmv_rect_csc_kernel<<<grid_size_7,
                                                      block_size_7>>>(
                        cblocks[i]->get_const_row_ptrs(),
                        cblocks[i]->get_const_col_idxs(),
                        as_cuda_type(cblocks[i]->get_const_values()),
                        as_cuda_type(xv->get_const_values()),
                        as_cuda_type(bv->get_values()), xl);
                } else {
                    rblocks[i]->apply(m_neg_one.get(), xv.get(), m_one.get(),
                                      bv.get());
                }

                // cudaMemcpy(x->get_values() + next_batch_start,
                // new_b->get_const_values() + next_batch_start, (next_batch_end
                // - next_batch_start) * sizeof(ValueType),
                // cudaMemcpyDeviceToDevice);


                // printf("bv[27] at 2: %.9lf\n",
                // exec->copy_val_to_host(new_b->get_const_values() +27));
                // printf("xv[27] at 2: %.9lf\n",
                // exec->copy_val_to_host(x->get_const_values() +27));

                // if(next_end_factor_i - next_start_factor_i == 1){
                //     cudaMemcpy(new_b->get_values() + next_batch_start,
                //     x->get_const_values() + next_batch_start, (next_batch_end
                //     - next_batch_start) *
                //     sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                //     cudaMemcpyDeviceToDevice);
                // }
            }

            start_factor_i = end_factor_i;
        }

        const auto final_x =
            x->permute(factor_perm, matrix::permute_mode::inverse_rows);
        cudaMemcpy(x->get_values(), final_x->get_const_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);


        // WIP ABOVE

        // IDEA
        // IDEA
        // IDEA
        // This product decomposition is also possible for row matrices
        //
        //             | 1 0 0 |   | 1 0 0 |   | 1 0 0 |
        // Meaning A = | d 1 0 | = | d 1 0 | * | 0 1 0 |
        //             | g h 1 |   | 0 0 1 |   | g h 1 |
        //
        // This immdeaitely begs the question: can we do a factorization of ther
        // form


        // WORKING BACKUP BELOW

        // for(auto i = 0; i < m; ++i){

        //         const auto factor_start =
        //         factor_offsets->get_const_data()[i]; const auto factor_end =
        //         i < m ? factor_offsets->get_const_data()[i + 1] :
        //         (IndexType)n;

        //         const dim3 block_size(default_block_size, 1, 1);
        //         const dim3 grid_size(ceildiv(factor_end - factor_start,
        //         block_size.x), 1, 1); // FIXME here for csr/csc swap
        //         sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_values()),
        //                 factor_start,
        //                 factor_end,
        //                 as_cuda_type(new_b->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 n);

        //                 // const auto one_val_2 =
        //                 exec->copy_val_to_host(x->get_values() + 2);
        //                 // printf("VAL at in x 2 at point 2: %lf\n",
        //                 one_val_2);

        //         const auto next_factor_start = i < m ?
        //         factor_offsets->get_const_data()[i + 1] : (IndexType)n; const
        //         auto next_factor_end = i < m - 1 ?
        //         factor_offsets->get_const_data()[i + 2] : (IndexType)n;

        //         cudaMemcpy(new_b->get_values() + next_factor_start,
        //         x->get_const_values() + next_factor_start, (next_factor_end -
        //         next_factor_start) *
        //         sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //         cudaMemcpyDeviceToDevice);
        // }


        // const auto final_x = x->permute(factor_perm,
        // matrix::permute_mode::inverse_rows); cudaMemcpy(x->get_values(),
        // final_x->get_const_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);


        // WORKING BACVKUP ABOVE


        // INFO
        // printf("%d/%d solves were small\n", (int32)count_small_solves,
        // (int32)m);


        // DEBUG BELOW

        //     // Initialize x to all NaNs.
        //     const auto clone_x = matrix::Dense<ValueType>::create(exec);
        //     clone_x->copy_from(b);
        //     dense::fill(exec, clone_x.get(), nan<ValueType>());

        //     array<bool> nan_produced(exec, 1);
        //     array<IndexType> atomic_counter(exec, 1);
        //     sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
        //                                  atomic_counter.get_data());

        //    const dim3 block_size_3(default_block_size, 1, 1);
        //             const dim3 grid_size_3(ceildiv(n, block_size_3.x), 1, 1);
        //     sptrsv_naive_caching_kernel<false><<<grid_size_3,block_size_3>>>(
        //         matrix->get_const_row_ptrs(),
        //         matrix->get_const_col_idxs(),
        //         as_cuda_type(matrix->get_const_values()),
        //         as_cuda_type(b->get_const_values()), b->get_stride(),
        //         as_cuda_type(clone_x->get_values()), clone_x->get_stride(),
        //         n, nrhs, false, nan_produced.get_data(),
        //         atomic_counter.get_data());

        //     const auto clone_final_x = clone_x->permute(factor_perm,
        //     matrix::permute_mode::rows); const auto computed_final_x =
        //     x->permute(factor_perm, matrix::permute_mode::rows);

        //     const auto dbg_tmp_0 = 0;
        //     sptrsvppi_debug_kernel<<<grid_size_3,block_size_3>>>(
        //         as_cuda_type(clone_final_x->get_const_values()),
        //         as_cuda_type(computed_final_x->get_const_values()), n,
        //         one<IndexType>());


        // DEBUG ABOVE
    }
};


// rdmpi for recursive delayed multi-partition inverse
template <typename ValueType, typename IndexType>
struct SptrsvrdmpiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> unit_inv_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_transp;


    std::unique_ptr<array<IndexType>> factor_offsets;
    std::unique_ptr<array<IndexType>> factor_assignments;
    std::unique_ptr<matrix::Permutation<IndexType>> factor_perm;

    std::unique_ptr<matrix::Dense<ValueType>> m_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_neg_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_zero;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> rblocks;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> cblocks;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> rtriangles;
    std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>> ctriangles;

    // TODO: Implement an algorithm where not factor_offsets, but these offsets
    // define the triangle borders Then you can do bigger triangular solves
    // including rectanuglar parts do this in csr format, probably
    std::vector<std::vector<IndexType>> rtriangles_offsets;

    // How many partitions to partition into?
    IndexType multi_p_count;

    array<IndexType> ms;
    IndexType m;


    SptrsvrdmpiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                           const matrix::Csr<ValueType, IndexType>* matrix,
                           size_type, bool is_upper, bool unit_diag, int32 rdmpi_m)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}, multi_p_count{rdmpi_m}
    {
        const auto n = matrix->get_size()[0];


        scaled_p_m =
            matrix::Csr<ValueType, IndexType>::create(exec, gko::dim<2>(n, n));
        // scaled_p_m->copy_from(matrix);

        if (!unit_diag) {
            diag->inverse_apply(matrix, scaled_p_m.get());
        }


        const auto nnz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();

        ms = array<IndexType>(exec->get_master(), multi_p_count);


        generate_factormin_perm(exec, matrix);

        scaled_p_m = scaled_p_m->permute(factor_perm);

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("stencil9pt.mtx.amd.rdm2pi");
        // gko::write(output, scaled_p_m);
        // output.close();


        unit_inv_p_m =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const dim3 block_size_0(default_block_size, 1, 1);
        const dim3 grid_size_0(ceildiv(n, block_size_0.x), 1, 1);
        sptrsvppi_negate_rdpi_kernel<<<grid_size_0, block_size_0>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n);


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(n, block_size_2.x), 1, 1);
        sptrsvppi_invert_rdpi_kernel<<<grid_size_2, block_size_2>>>(
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            as_cuda_type(scaled_p_m->get_values()),
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n, factor_offsets->get_size(),
            nnz);

        factor_offsets->set_executor(exec->get_master());

        // for (auto i = 0; i < m; ++i) {
        //     printf(" %d ", (int32)factor_offsets->get_const_data()[i]);
        // }
        // printf("\n");

        // rtriangles are now in csr
        unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
            unit_inv_p_m->transpose());


        rtriangles = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 1);
        ctriangles = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 1);

        for (auto i = 0; i < m - 1; ++i) {
            // printf("creating triangle from %d to %d\n",
                //    (int32)factor_offsets->get_const_data()[i],
                //    (int32)factor_offsets->get_const_data()[i + 1]);

            rtriangles[i] = unit_inv_p_m->create_submatrix(
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]},
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]});
            rtriangles[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }

        // ctriangles are now in csc
        unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
            unit_inv_p_m->transpose());

        for (auto i = 0; i < m - 1; ++i) {
            // printf("creating triangle from %d to %d\n",
                //    (int32)factor_offsets->get_const_data()[i],
                //    (int32)factor_offsets->get_const_data()[i + 1]);

            ctriangles[i] = unit_inv_p_m->create_submatrix(
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]},
                span{factor_offsets->get_const_data()[i],
                     factor_offsets->get_const_data()[i + 1]});
            ctriangles[i]->set_strategy(
                std::make_shared<
                    typename matrix::Csr<ValueType, IndexType>::automatical>(
                    exec));
        }


        rblocks = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 2);
        cblocks = std::vector<
            std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>>>(m - 2);

        auto block_coords =
            std::vector<std::pair<std::pair<IndexType, IndexType>,
                                  std::pair<IndexType, IndexType>>>();
        block_coords.push_back(
            std::make_pair(std::make_pair(0, n), std::make_pair(0, n)));

        do {
            const auto popped = block_coords.back();
            block_coords.pop_back();

            const IndexType from_row = popped.first.first;
            const IndexType to_row = popped.first.second;
            const IndexType from_col = popped.second.first;
            const IndexType to_col = popped.second.second;

            const auto row_dist = to_row - from_row;
            const auto col_dist = to_col - from_col;

            auto eligible_offsets = std::vector<IndexType>();
            for (auto offset_i = 0; offset_i < m; ++offset_i) {
                const auto offset = factor_offsets->get_const_data()[offset_i];
                if (offset >= from_col && offset <= to_col) {
                    eligible_offsets.push_back(offset);
                }
            }

            if (eligible_offsets.size() <= 2) {
                continue;
            }

            auto closest_offset = std::min_element(
                eligible_offsets.begin(), eligible_offsets.end(),
                [&](IndexType off_a, IndexType off_b) {
                    return std::abs((float)from_col + (float)col_dist / 2 -
                                    (float)off_a) <
                           std::abs((float)from_col + (float)col_dist / 2 -
                                    (float)off_b);
                });
            const auto closest_offset_i =
                std::distance(factor_offsets->get_const_data(),
                              std::find(factor_offsets->get_const_data(),
                                        factor_offsets->get_const_data() + m,
                                        *closest_offset));

            // printf("Added rect %d %d %d %d at pos %d\n", (int32)*closest_offset,
            //        (int32)to_row, (int32)from_col, (int32)*closest_offset,
            //        (int32)closest_offset_i);

            block_coords.push_back(
                std::make_pair(std::make_pair(from_row, *closest_offset),
                               std::make_pair(from_col, *closest_offset)));
            block_coords.push_back(
                std::make_pair(std::make_pair(*closest_offset, to_row),
                               std::make_pair(*closest_offset, to_col)));

            // when using scaled_p_m here, it's csr
            // if you want csc, use unit_inv_p_m

            const auto csc_better = to_row - *closest_offset >=
                                    (5 * (*closest_offset - from_col)) / 10;

            if (csc_better) {
                cblocks[closest_offset_i - 1] = unit_inv_p_m->create_submatrix(
                    span{from_col, *closest_offset},
                    span{*closest_offset, to_row});
                cblocks[closest_offset_i - 1]->set_strategy(
                    std::make_shared<typename matrix::Csr<
                        ValueType, IndexType>::automatical>(exec));
            } else {
                rblocks[closest_offset_i - 1] = scaled_p_m->create_submatrix(
                    span{*closest_offset, to_row},
                    span{from_col, *closest_offset});
                rblocks[closest_offset_i - 1]->set_strategy(
                    std::make_shared<typename matrix::Csr<
                        ValueType, IndexType>::automatical>(exec));
            }

        } while (!block_coords.empty());


        // FREE
        scaled_p_m.reset();
        unit_inv_p_m.reset();

        m_neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
        m_one = gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
        m_zero = gko::initialize<gko::matrix::Dense<ValueType>>({0}, exec);
    }

    void generate_factormin_perm(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix)
    {
        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        array<IndexType> levels(exec, n);
        cudaMemset(levels.get_data(), 0xFF, n * sizeof(IndexType));

        array<IndexType> height_d(exec, 1);
        cudaMemset(height_d.get_data(), 0, sizeof(IndexType));

        array<IndexType> atomic_counter(exec, 1);
        cudaMemset(atomic_counter.get_data(), 0, sizeof(IndexType));

        const auto block_size = default_block_size;
        const auto block_count = (n + block_size - 1) / block_size;


        _sptrsvrdpi_level_generation_kernel<IndexType, false>
            <<<block_count, block_size>>>(
                matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                levels.get_data(), height_d.get_data(), n,
                atomic_counter.get_data());

        // INFO
        // printf("Level height is %d\n",
        //        (int32)exec->copy_val_to_host(height_d.get_const_data()));

        array<IndexType> levelsperm(exec, n);
        thrust::sequence(thrust::device, levelsperm.get_data(),
                         levelsperm.get_data() + n);
        thrust::stable_sort_by_key(thrust::device, levels.get_data(),
                                   levels.get_data() + n,
                                   levelsperm.get_data());

        // Multipartition starts with an initial permutation
        array<IndexType> initial_perm(exec, n);
        // thrust::sequence(thrust::device, initial_perm.get_data(),
        // initial_perm.get_data() + n);
        cudaMemcpy(initial_perm.get_data(), levelsperm.get_const_data(),
                   n * sizeof(IndexType), cudaMemcpyDeviceToDevice);
        auto iperm = matrix::Permutation<IndexType>::create(exec, initial_perm);


        auto pmatrix = matrix->permute(iperm);
        auto t_pmatrix =
            gko::as<matrix::Csr<ValueType, IndexType>>(pmatrix->transpose());


        gko::array<IndexType> submatrix_starts(exec->get_master(),
                                               multi_p_count);
        auto ns = gko::array<IndexType>(exec->get_master(), multi_p_count);

        auto submatrices_csr =
            std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>>(
                multi_p_count);
        auto submatrices_csc =
            std::vector<std::shared_ptr<matrix::Csr<ValueType, IndexType>>>(
                multi_p_count);
        for (auto i = 0; i < multi_p_count; ++i) {
            const auto a = (size_type)(((float)i / (float)multi_p_count) * n);
            const auto b =
                (size_type)((((float)i + 1) / (float)multi_p_count) * n);
            submatrices_csr[i] =
                pmatrix->create_submatrix(span{a, b}, span{a, b});
            submatrices_csc[i] =
                t_pmatrix->create_submatrix(span{a, b}, span{a, b});
            submatrix_starts.get_data()[i] = a;
            ns.get_data()[i] = (IndexType)(b - a);
        }
        submatrix_starts.set_executor(exec);

        array<const IndexType*> rowptrs_batch(exec->get_master(),
                                              multi_p_count);
        array<const IndexType*> colptrs_batch(exec->get_master(),
                                              multi_p_count);
        array<const IndexType*> rowidxs_batch(exec->get_master(),
                                              multi_p_count);
        array<const IndexType*> colidxs_batch(exec->get_master(),
                                              multi_p_count);
        for (auto i = 0; i < multi_p_count; ++i) {
            rowptrs_batch.get_data()[i] =
                submatrices_csr[i]->get_const_row_ptrs();
            colptrs_batch.get_data()[i] =
                submatrices_csc[i]->get_const_row_ptrs();
            rowidxs_batch.get_data()[i] =
                submatrices_csc[i]->get_const_col_idxs();
            colidxs_batch.get_data()[i] =
                submatrices_csr[i]->get_const_col_idxs();
        }
        rowptrs_batch.set_executor(exec);
        colptrs_batch.set_executor(exec);
        rowidxs_batch.set_executor(exec);
        colidxs_batch.set_executor(exec);


        gko::array<IndexType> factor_sizes_batch(exec, n);
        gko::array<IndexType> factor_assignments_batch(exec, n);
        cudaMemset(factor_assignments_batch.get_data(), 0,
                   n * sizeof(IndexType));
        cudaMemset(factor_sizes_batch.get_data(), 0, n * sizeof(IndexType));

        sptrsvdrmpi_create_ppi(
            exec, pmatrix->get_const_row_ptrs(), pmatrix->get_const_col_idxs(),
            t_pmatrix->get_const_row_ptrs(), t_pmatrix->get_const_col_idxs(),
            rowptrs_batch.get_const_data(), colptrs_batch.get_const_data(),
            rowidxs_batch.get_const_data(), colidxs_batch.get_const_data(),
            factor_sizes_batch.get_data(), factor_assignments_batch.get_data(),
            // purely_diagonal_elements.get_const_data(),
            // purely_diagonal_count.get_const_data()[0],
            // &rem_preds[0],
            // counts.get_data(),
            (IndexType)n, ns.get_const_data(),
            // (IndexType)nz,
            submatrix_starts.get_const_data(), multi_p_count, ms.get_data(),
            one<ValueType>());

        const auto dbg_tmp_453 = 0;

        auto ms_sum = 0;
        for (auto i = 0; i < multi_p_count; ++i) {
            ms_sum += ms.get_const_data()[i];
        }
        // factor_assignments_.set_executor(exec);

        submatrix_starts.set_executor(exec->get_master());

        array<IndexType> factor_sizes(exec, ms_sum + 1);
        array<IndexType> factor_assignments_(exec, n);
        auto off = 0;
        for (auto i = 0; i < multi_p_count; ++i) {
            cudaMemcpy(factor_sizes.get_data() + off,
                       factor_sizes_batch.get_data() +
                           submatrix_starts.get_const_data()[i],
                       ms.get_const_data()[i] * sizeof(IndexType),
                       cudaMemcpyDeviceToDevice);
            off += ms.get_const_data()[i];
        }
        cudaMemset(factor_sizes.get_data() + ms_sum, 0, sizeof(IndexType));


        ns.set_executor(exec);
        ms.set_executor(exec);
        const dim3 block_size_54(default_block_size, 1, 1);
        const dim3 grid_size_54(ceildiv(n, block_size_54.x), 1, 1);

        sptrsvrdmpi_factor_asssignments_gather_kernel<<<grid_size_54,
                                                        block_size_54>>>(
            factor_assignments_batch.get_const_data(),
            factor_assignments_.get_data(), ns.get_const_data(),
            ms.get_const_data(), multi_p_count, (IndexType)n);
        ns.set_executor(exec->get_master());
        ms.set_executor(exec->get_master());

        // thrust::exclusive_scan(thrust::device, factor_sizes.get_data(),
        // factor_sizes.get_data() + ms_sum + 1, factor_sizes.get_data());
        components::prefix_sum_nonnegative(exec, factor_sizes.get_data(),
                                           ms_sum + 1);
        m = ms_sum + 1;


        array<IndexType> second_perm(exec, n);
        array<IndexType> final_perm(exec, n);
        thrust::sequence(thrust::device, second_perm.get_data(),
                         second_perm.get_data() + n);
        // // thrust::stable_sort_by_key(thrust::device, levels.get_data(),
        // levels.get_data() + n, level_perm.get_data());
        thrust::stable_sort_by_key(
            thrust::device, factor_assignments_.get_data(),
            factor_assignments_.get_data() + n, second_perm.get_data());

        // for(auto i = 0; i < multi_p_count; ++i){
        //     thrust::sequence(thrust::device, second_perm.get_data() +
        //     submatrix_starts.get_const_data()[i],
        //         second_perm.get_data() + submatrix_starts.get_const_data()[i]
        //         + ns.get_data()[i], submatrix_starts.get_const_data()[i]);
        //     thrust::stable_sort_by_key(thrust::device,
        //         factor_assignments_.get_data() +
        //         submatrix_starts.get_const_data()[i],
        //         factor_assignments_.get_data() +
        //         submatrix_starts.get_const_data()[i] +
        //         ns.get_const_data()[i], second_perm.get_data() +
        //         submatrix_starts.get_const_data()[i]);
        // }

        thrust::gather(thrust::device, second_perm.get_data(),
                       second_perm.get_data() + n, levelsperm.get_data(),
                       final_perm.get_data());
        // sptrsvrdmpi_assemble_perm_kernel<<<grid_size_54, block_size_54>>>(
        //     levelsperm.get_const_data(),
        //     second_perm.get_const_data(),
        //     final_perm.get_data(),
        //     (IndexType)n
        // );

        factor_perm = matrix::Permutation<IndexType>::create(exec, final_perm);
        factor_offsets =
            std::make_unique<array<IndexType>>(std::move(factor_sizes));
        factor_assignments =
            std::make_unique<array<IndexType>>(std::move(factor_assignments_));

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("bcsstk11.secondperm.mtx");
        // gko::write(output, pmatrix->permute(factor_perm));
        // output.close();


        // factor_offsets_d = std::make_unique<array<IndexType>>(exec);
        // factor_offsets_d->resize_and_reset(m + 1);
        // cudaMemcpy(factor_offsets_d->get_data(),
        // factor_assignments->get_const_data(), (m + 1) * sizeof(IndexType),
        // cudaMemcpyDeviceToDevice);
        // factor_offsets->set_executor(exec->get_master());
        // factor_offsets_d->set_executor(exec);

        // INFO
        printf("PPi m is %d\n", (int32)m);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        auto new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());
        new_b = new_b->permute(factor_perm, matrix::permute_mode::rows);

        // const dim3 block_size_2(default_block_size, 1,1);
        // const dim3 grid_size_2(ceildiv(n * nrhs, block_size_2.x), 1, 1);
        // // DEBUG
        // sptrsvppi_debug_bb_kernel<<<grid_size_2,block_size_2>>>(
        //     as_cuda_type(b->get_const_values()),
        //     as_cuda_type(new_b->get_const_values()), n, one<IndexType>());


        // WIP BELOW

        factor_offsets->set_executor(exec->get_master());

        // cudaMemcpy(x->get_values(), new_b->get_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);
        cudaMemset(x->get_values(), 0,
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))));


        auto batch_instructions = std::vector<int32>();
        for (auto i = 0; i < m - 1; ++i) {
            const auto factor_start = factor_offsets->get_const_data()[i];
            const auto factor_end = factor_offsets->get_const_data()[i + 1];

            const auto is_batch_eligible =
                false;  // factor_end - factor_start <= ppi_batch_bocksize;

            if (is_batch_eligible) {
                if (batch_instructions.empty() ||
                    batch_instructions.back() == 0) {
                    batch_instructions.push_back(1);
                } else {
                    batch_instructions.back() += 1;
                }
            } else {
                if (!batch_instructions.empty() &&
                    batch_instructions.back() == 1) {
                    batch_instructions.back() -=
                        1;  // singular batches are re-transformed to plain
                            // spmvs
                }
                batch_instructions.push_back(0);
            }
        }


        const auto reduced_m = batch_instructions.size();
        IndexType start_factor_i = 0;
        for (auto i = 0; i < reduced_m; ++i) {
            const auto batch_instr = batch_instructions[i];

            const IndexType end_factor_i = batch_instr == 0
                                               ? start_factor_i + 1
                                               : start_factor_i + batch_instr;

            const auto batch_start =
                factor_offsets->get_const_data()[start_factor_i];
            const auto batch_end =
                end_factor_i < m
                    ? factor_offsets->get_const_data()[end_factor_i]
                    : (IndexType)n;


            if (batch_instr == 0) {
                // const dim3 block_size(default_block_size, 1, 1);
                // const dim3 grid_size(ceildiv(batch_end - batch_start,
                // block_size.x), 1, 1); // FIXME here for csr/csc swap
                // sptrsvrdpi_solve_factor_csc_kernel<<<grid_size,
                // block_size>>>(
                //         unit_inv_p_m->get_const_row_ptrs(),
                //         unit_inv_p_m->get_const_col_idxs(),
                //         as_cuda_type(unit_inv_p_m->get_values()),
                //         batch_start,
                //         batch_end,
                //         as_cuda_type(new_b->get_const_values()),
                //         as_cuda_type(x->get_values()),
                //         n);

                const auto xv = x->create_submatrix(
                    span{batch_start, batch_end}, span{0, 1});
                auto bv = new_b->create_submatrix(span{batch_start, batch_end},
                                                  span{0, 1});

                const auto is_csc_triangle = true;
                if (is_csc_triangle) {
                    // cudaMemcpy(x->get_values() + batch_start,
                    // new_b->get_const_values() + batch_start, (batch_end -
                    // batch_start) * sizeof(ValueType),
                    // cudaMemcpyDeviceToDevice);
                    const dim3 block_size_8(default_block_size, 1, 1);
                    const dim3 grid_size_8(
                        ceildiv(batch_end - batch_start, default_block_size), 1,
                        1);
                    sptrsvrdpi_spmv_triangle_csc_kernel<<<grid_size_8,
                                                          block_size_8>>>(
                        ctriangles[i]->get_const_row_ptrs(),
                        ctriangles[i]->get_const_col_idxs(),
                        as_cuda_type(ctriangles[i]->get_const_values()),
                        as_cuda_type(bv->get_const_values()),
                        as_cuda_type(xv->get_values()),
                        batch_end - batch_start);
                } else {
                    rtriangles[i]->apply(m_one.get(), bv.get(), m_zero.get(),
                                         xv.get());
                }
                // const auto one_val_2 = exec->copy_val_to_host(x->get_values()
                // + 2); printf("VAL at in x 2 at point 2: %lf\n", one_val_2);

            } else {
                array<int32> factors_done(exec, 1);
                factors_done.fill(zero<int32>());
                factor_offsets->set_executor(exec);

                const dim3 block_size_0(ppi_batch_bocksize, 1, 1);
                const dim3 grid_size_0(end_factor_i - start_factor_i, 1, 1);
                sptrsvppi_solve_factor_batched_csc_kernel<<<grid_size_0,
                                                            block_size_0>>>(
                    unit_inv_p_m->get_const_row_ptrs(),
                    unit_inv_p_m->get_const_col_idxs(),
                    as_cuda_type(unit_inv_p_m->get_const_values()),
                    factor_offsets->get_const_data(), start_factor_i,
                    end_factor_i, as_cuda_type(x->get_const_values()),
                    as_cuda_type(x->get_values()), factors_done.get_data(),
                    (gko::size_type)n, (IndexType)m);

                factor_offsets->set_executor(exec->get_master());
            }


            if (batch_end < n) {
                const auto next_start_factor_i = end_factor_i;
                const auto next_end_factor_i =
                    batch_instructions[i + 1] == 0
                        ? next_start_factor_i + 1
                        : next_start_factor_i + batch_instructions[i + 1];
                const auto next_batch_start =
                    factor_offsets->get_const_data()[next_start_factor_i];
                const auto next_batch_end =
                    next_end_factor_i < m
                        ? factor_offsets->get_const_data()[next_end_factor_i]
                        : (IndexType)n;

                // const auto xv = x->create_submatrix(span{batch_start,
                // batch_end}, span{0, 1}); auto bv =
                // new_b->create_submatrix(span{batch_end, n}, span{0, 1});

                const auto is_csc_block = rblocks[i].get() == NULL;

                const auto xl = is_csc_block ? cblocks[i]->get_size()[0]
                                             : rblocks[i]->get_size()[1];
                const auto bl = is_csc_block ? cblocks[i]->get_size()[1]
                                             : rblocks[i]->get_size()[0];

                const auto xv = x->create_submatrix(
                    span{batch_end - xl, batch_end}, span{0, 1});
                auto bv = new_b->create_submatrix(
                    span{batch_end, batch_end + bl}, span{0, 1});

                // printf("bv[27] at 1: %.9lf\n",
                // exec->copy_val_to_host(new_b->get_const_values() +27));
                // printf("xv[27] at 1: %.9lf\n",
                // exec->copy_val_to_host(x->get_const_values() +27));

                if (is_csc_block) {
                    const dim3 block_size_7(default_block_size, 1, 1);
                    const dim3 grid_size_7(ceildiv(xl, default_block_size), 1,
                                           1);
                    sptrsvrdpi_spmv_rect_csc_kernel<<<grid_size_7,
                                                      block_size_7>>>(
                        cblocks[i]->get_const_row_ptrs(),
                        cblocks[i]->get_const_col_idxs(),
                        as_cuda_type(cblocks[i]->get_const_values()),
                        as_cuda_type(xv->get_const_values()),
                        as_cuda_type(bv->get_values()), xl);
                } else {
                    rblocks[i]->apply(m_neg_one.get(), xv.get(), m_one.get(),
                                      bv.get());
                }

                // cudaMemcpy(x->get_values() + next_batch_start,
                // new_b->get_const_values() + next_batch_start, (next_batch_end
                // - next_batch_start) * sizeof(ValueType),
                // cudaMemcpyDeviceToDevice);


                // printf("bv[27] at 2: %.9lf\n",
                // exec->copy_val_to_host(new_b->get_const_values() +27));
                // printf("xv[27] at 2: %.9lf\n",
                // exec->copy_val_to_host(x->get_const_values() +27));

                // if(next_end_factor_i - next_start_factor_i == 1){
                //     cudaMemcpy(new_b->get_values() + next_batch_start,
                //     x->get_const_values() + next_batch_start, (next_batch_end
                //     - next_batch_start) *
                //     sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                //     cudaMemcpyDeviceToDevice);
                // }
            }

            start_factor_i = end_factor_i;
        }

        const auto final_x =
            x->permute(factor_perm, matrix::permute_mode::inverse_rows);
        cudaMemcpy(x->get_values(), final_x->get_const_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);


        // WIP ABOVE

        // IDEA
        // IDEA
        // IDEA
        // This product decomposition is also possible for row matrices
        //
        //             | 1 0 0 |   | 1 0 0 |   | 1 0 0 |
        // Meaning A = | d 1 0 | = | d 1 0 | * | 0 1 0 |
        //             | g h 1 |   | 0 0 1 |   | g h 1 |
        //
        // This immdeaitely begs the question: can we do a factorization of ther
        // form


        // WORKING BACKUP BELOW

        // for(auto i = 0; i < m; ++i){

        //         const auto factor_start =
        //         factor_offsets->get_const_data()[i]; const auto factor_end =
        //         i < m ? factor_offsets->get_const_data()[i + 1] :
        //         (IndexType)n;

        //         const dim3 block_size(default_block_size, 1, 1);
        //         const dim3 grid_size(ceildiv(factor_end - factor_start,
        //         block_size.x), 1, 1); // FIXME here for csr/csc swap
        //         sptrsvppi_solve_factor_csc_kernel<<<grid_size, block_size>>>(
        //                 unit_inv_p_m->get_const_row_ptrs(),
        //                 unit_inv_p_m->get_const_col_idxs(),
        //                 as_cuda_type(unit_inv_p_m->get_values()),
        //                 factor_start,
        //                 factor_end,
        //                 as_cuda_type(new_b->get_const_values()),
        //                 as_cuda_type(x->get_values()),
        //                 n);

        //                 // const auto one_val_2 =
        //                 exec->copy_val_to_host(x->get_values() + 2);
        //                 // printf("VAL at in x 2 at point 2: %lf\n",
        //                 one_val_2);

        //         const auto next_factor_start = i < m ?
        //         factor_offsets->get_const_data()[i + 1] : (IndexType)n; const
        //         auto next_factor_end = i < m - 1 ?
        //         factor_offsets->get_const_data()[i + 2] : (IndexType)n;

        //         cudaMemcpy(new_b->get_values() + next_factor_start,
        //         x->get_const_values() + next_factor_start, (next_factor_end -
        //         next_factor_start) *
        //         sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        //         cudaMemcpyDeviceToDevice);
        // }


        // const auto final_x = x->permute(factor_perm,
        // matrix::permute_mode::inverse_rows); cudaMemcpy(x->get_values(),
        // final_x->get_const_values(), n *
        // sizeof(decltype(as_cuda_type(zero<ValueType>()))),
        // cudaMemcpyDeviceToDevice);


        // WORKING BACVKUP ABOVE


        // INFO
        // printf("%d/%d solves were small\n", (int32)count_small_solves,
        // (int32)m);


        // DEBUG BELOW

        //     // Initialize x to all NaNs.
        //     const auto clone_x = matrix::Dense<ValueType>::create(exec);
        //     clone_x->copy_from(b);
        //     dense::fill(exec, clone_x.get(), nan<ValueType>());

        //     array<bool> nan_produced(exec, 1);
        //     array<IndexType> atomic_counter(exec, 1);
        //     sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
        //                                  atomic_counter.get_data());

        //    const dim3 block_size_3(default_block_size, 1, 1);
        //             const dim3 grid_size_3(ceildiv(n, block_size_3.x), 1, 1);
        //     sptrsv_naive_caching_kernel<false><<<grid_size_3,block_size_3>>>(
        //         matrix->get_const_row_ptrs(),
        //         matrix->get_const_col_idxs(),
        //         as_cuda_type(matrix->get_const_values()),
        //         as_cuda_type(b->get_const_values()), b->get_stride(),
        //         as_cuda_type(clone_x->get_values()), clone_x->get_stride(),
        //         n, nrhs, false, nan_produced.get_data(),
        //         atomic_counter.get_data());

        //     const auto clone_final_x = clone_x->permute(factor_perm,
        //     matrix::permute_mode::rows); const auto computed_final_x =
        //     x->permute(factor_perm, matrix::permute_mode::rows);

        //     const auto dbg_tmp_0 = 0;
        //     sptrsvppi_debug_kernel<<<grid_size_3,block_size_3>>>(
        //         as_cuda_type(clone_final_x->get_const_values()),
        //         as_cuda_type(computed_final_x->get_const_values()), n,
        //         one<IndexType>());


        // DEBUG ABOVE
    }
};


// rspi for recursive single product inverse
template <typename ValueType, typename IndexType>
struct SptrsvrspiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> unit_inv_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_transp;


    std::unique_ptr<array<IndexType>> factor_offsets;
    std::unique_ptr<array<IndexType>> factor_assignments;
    std::unique_ptr<matrix::Permutation<IndexType>> factor_perm;

    array<IndexType> row_factor_starts;

    std::unique_ptr<matrix::Dense<ValueType>> m_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_neg_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_zero;

    std::unique_ptr<array<ValueType>> x_tmp;

    IndexType m;


    SptrsvrspiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper}, diag{matrix->extract_diagonal()}
    {
        const auto n = matrix->get_size()[0];

        scaled_p_m =
            matrix::Csr<ValueType, IndexType>::create(exec, gko::dim<2>(n, n));
        // scaled_p_m->copy_from(matrix);

        if (!unit_diag) {
            diag->inverse_apply(matrix, scaled_p_m.get());
        }

        scaled_transp =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const auto nnz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        generate_factormin_perm(exec, matrix, scaled_transp.get());

        // FREE
        scaled_transp.reset();


        scaled_p_m = scaled_p_m->permute(factor_perm);

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("pkust11.input.rdpi.mtx");
        // gko::write(output, scaled_p_m);
        // output.close();


        unit_inv_p_m =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const dim3 block_size_0(default_block_size, 1, 1);
        const dim3 grid_size_0(ceildiv(n, block_size_0.x), 1, 1);
        sptrsvppi_negate_rdpi_kernel<<<grid_size_0, block_size_0>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n);


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(n, block_size_2.x), 1, 1);
        sptrsvppi_invert_rdpi_kernel<<<grid_size_2, block_size_2>>>(
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            as_cuda_type(scaled_p_m->get_values()),
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n, factor_offsets->get_size(),
            nnz);

        // FREE
        scaled_p_m.reset();



        // for(auto i = 0; i < m; ++i){
        //     printf(" %d ", (int32)factor_offsets->get_const_data()[i]);
        // }
        // printf("\n");

        // now in csr
        unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
            unit_inv_p_m->transpose());

        row_factor_starts = gko::array<IndexType>(exec, n);
        sptrsvppi_rowfactorsstarts_rspi_kernel<<<grid_size_2, block_size_2>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            factor_offsets->get_const_data(), row_factor_starts.get_data(), n,
            m);

        factor_offsets->set_executor(exec->get_master());


        x_tmp = std::make_unique<gko::array<ValueType>>(exec, n);

        m_neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
        m_one = gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
        m_zero = gko::initialize<gko::matrix::Dense<ValueType>>({0}, exec);
    }

    void generate_factormin_perm(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const matrix::Csr<ValueType, IndexType>* t_matrix)
    {
        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();

        array<IndexType> factor_sizes(exec, n);
        array<IndexType> factor_assignments_(exec, n);
        cudaMemset(factor_sizes.get_data(), 0, n * sizeof(IndexType));
        cudaMemset(factor_assignments_.get_data(), 0xFF, n * sizeof(IndexType));
        // factor_sizes.fill(zero<IndexType>());
        // factor_assignments_.fill(-one<IndexType>());

        sptrsvdrpi_create_ppi(
            exec, matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            t_matrix->get_const_row_ptrs(), t_matrix->get_const_col_idxs(),
            factor_sizes.get_data(), factor_assignments_.get_data(),
            // purely_diagonal_elements.get_const_data(),
            // purely_diagonal_count.get_const_data()[0],
            // &rem_preds[0],
            // counts.get_data(),
            (IndexType)n,
            // (IndexType)nz,
            &m, one<ValueType>());


        components::prefix_sum_nonnegative(exec, factor_sizes.get_data(), m);


        array<IndexType> perm(exec, n);
        thrust::sequence(thrust::device, perm.get_data(), perm.get_data() + n);
        // thrust::stable_sort_by_key(thrust::device, levels.get_data(),
        // levels.get_data() + n, level_perm.get_data());
        thrust::stable_sort_by_key(
            thrust::device, factor_assignments_.get_data(),
            factor_assignments_.get_data() + n, perm.get_data());


        factor_perm = matrix::Permutation<IndexType>::create(exec, perm);
        factor_offsets =
            std::make_unique<array<IndexType>>(std::move(factor_sizes));
        factor_assignments =
            std::make_unique<array<IndexType>>(std::move(factor_assignments_));

        // INFO
        printf("PPi m is %d\n", (int32)m);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        auto new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());
        new_b = new_b->permute(factor_perm, matrix::permute_mode::rows);
        x->fill(nan<ValueType>());
        x_tmp->fill(nan<ValueType>());

        // const dim3 block_size_2(default_block_size, 1,1);
        // const dim3 grid_size_2(ceildiv(n * nrhs, block_size_2.x), 1, 1);
        // // DEBUG
        // sptrsvppi_debug_bb_kernel<<<grid_size_2,block_size_2>>>(
        //     as_cuda_type(b->get_const_values()),
        //     as_cuda_type(new_b->get_const_values()), n, one<IndexType>());


        // WIP BELOW

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(2 * n, block_size.x), 1, 1); // FIXME here for factor threads
        sptrsvppi_solve_rspi_kernel<<<grid_size, block_size>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_const_values()),
            row_factor_starts.get_const_data(),
            as_cuda_type(new_b->get_const_values()),
            as_cuda_type(x_tmp->get_data()), as_cuda_type(x->get_values()), n);

        const auto final_x =
            x->permute(factor_perm, matrix::permute_mode::inverse_rows);
        cudaMemcpy(x->get_values(), final_x->get_const_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);


        // WIP ABOVE


        // DEBUG BELOW

        //     // Initialize x to all NaNs.
        //     const auto clone_x = matrix::Dense<ValueType>::create(exec);
        //     clone_x->copy_from(b);
        //     dense::fill(exec, clone_x.get(), nan<ValueType>());

        //     array<bool> nan_produced(exec, 1);
        //     array<IndexType> atomic_counter(exec, 1);
        //     sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
        //                                  atomic_counter.get_data());

        //    const dim3 block_size_3(default_block_size, 1, 1);
        //             const dim3 grid_size_3(ceildiv(n, block_size_3.x), 1, 1);
        //     sptrsv_naive_caching_kernel<false><<<grid_size_3,block_size_3>>>(
        //         matrix->get_const_row_ptrs(),
        //         matrix->get_const_col_idxs(),
        //         as_cuda_type(matrix->get_const_values()),
        //         as_cuda_type(b->get_const_values()), b->get_stride(),
        //         as_cuda_type(clone_x->get_values()), clone_x->get_stride(),
        //         n, nrhs, false, nan_produced.get_data(),
        //         atomic_counter.get_data());

        //     const auto clone_final_x = clone_x->permute(factor_perm,
        //     matrix::permute_mode::rows); const auto computed_final_x =
        //     x->permute(factor_perm, matrix::permute_mode::rows);

        //     const auto dbg_tmp_0 = 0;
        //     sptrsvppi_debug_kernel<<<grid_size_3,block_size_3>>>(
        //         as_cuda_type(clone_final_x->get_const_values()),
        //         as_cuda_type(computed_final_x->get_const_values()), n,
        //         one<IndexType>());


        // DEBUG ABOVE
    }
};


// rscpi for recursive single csc product inverse
template <typename ValueType, typename IndexType>
struct SptrsvrscpiSolveStruct : gko::solver::SolveStruct {
    bool is_upper;

    // Acutally never used after gen, but stored here anyway, because very
    // critical
    IndexType ticks_prod;

    std::unique_ptr<gko::matrix::Diagonal<ValueType>> diag;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> unit_inv_p_m;
    std::unique_ptr<gko::matrix::Csr<ValueType, IndexType>> scaled_transp;


    std::unique_ptr<array<IndexType>> factor_offsets;
    std::unique_ptr<array<IndexType>> factor_assignments;
    std::unique_ptr<matrix::Permutation<IndexType>> factor_perm;

    array<IndexType> col_factor_ends; // n values
    array<int32> row_counts; // 2n values
    mutable array<int32> row_counts_clone; // 2n values

    std::unique_ptr<matrix::Dense<ValueType>> m_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_neg_one;
    std::unique_ptr<matrix::Dense<ValueType>> m_zero;

    std::unique_ptr<array<ValueType>> x_tmp;

    IndexType m;


    SptrsvrscpiSolveStruct(std::shared_ptr<const gko::CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* matrix,
                          size_type, bool is_upper, bool unit_diag)
        : is_upper{is_upper}, 
        diag{matrix->extract_diagonal()}
    {
        const auto n = matrix->get_size()[0];

        scaled_p_m =
            matrix::Csr<ValueType, IndexType>::create(exec, gko::dim<2>(n, n));
        // scaled_p_m->copy_from(matrix);

        if (!unit_diag) {
            diag->inverse_apply(matrix, scaled_p_m.get());
        }

        scaled_transp =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const auto nnz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();


        generate_factormin_perm(exec, matrix, scaled_transp.get());

        // FREE
        scaled_transp.reset();


        scaled_p_m = scaled_p_m->permute(factor_perm);

        // DEBUG
        // std::basic_ofstream<char> output;
        // output.open("pkust11.input.rdpi.mtx");
        // gko::write(output, scaled_p_m);
        // output.close();


        unit_inv_p_m =
            gko::as<matrix::Csr<ValueType, IndexType>>(scaled_p_m->transpose());


        const dim3 block_size_0(default_block_size, 1, 1);
        const dim3 grid_size_0(ceildiv(n, block_size_0.x), 1, 1);
        sptrsvppi_negate_rdpi_kernel<<<grid_size_0, block_size_0>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n);


        const dim3 block_size_2(default_block_size, 1, 1);
        const dim3 grid_size_2(ceildiv(n, block_size_2.x), 1, 1);
        sptrsvppi_invert_rdpi_kernel<<<grid_size_2, block_size_2>>>(
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            as_cuda_type(scaled_p_m->get_values()),
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_values()),
            factor_offsets->get_const_data(),
            factor_assignments->get_const_data(), n, factor_offsets->get_size(),
            nnz);

        



        // for(auto i = 0; i < m; ++i){
        //     printf(" %d ", (int32)factor_offsets->get_const_data()[i]);
        // }
        // printf("\n");

        // if commented out, unit_inv_p_m stays in csc
        // unit_inv_p_m = gko::as<matrix::Csr<ValueType, IndexType>>(
        //     unit_inv_p_m->transpose());

        col_factor_ends = gko::array<IndexType>{exec, n};
        row_counts = gko::array<int32>{exec, 2 * n};
        row_counts_clone = gko::array<int32>{exec, 2 * n};
        const dim3 block_size_243(default_block_size, 1, 1);
        const dim3 grid_size_243(ceildiv(2 * n, block_size_2.x), 1, 1);
        sptrsvppi_rowfactorsstarts_rscpi_kernel<<<grid_size_243, block_size_243>>>(
            unit_inv_p_m->get_const_row_ptrs(), unit_inv_p_m->get_const_col_idxs(),
            scaled_p_m->get_const_row_ptrs(), scaled_p_m->get_const_col_idxs(),
            factor_offsets->get_const_data(), 
            col_factor_ends.get_data(), 
            row_counts.get_data(),
            n, m);
            
        // FREE
        scaled_p_m.reset();

        factor_offsets->set_executor(exec->get_master());


        x_tmp = std::make_unique<gko::array<ValueType>>(exec, n);

        m_neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1}, exec);
        m_one = gko::initialize<gko::matrix::Dense<ValueType>>({1}, exec);
        m_zero = gko::initialize<gko::matrix::Dense<ValueType>>({0}, exec);
    }

    void generate_factormin_perm(
        std::shared_ptr<const CudaExecutor> exec,
        const matrix::Csr<ValueType, IndexType>* matrix,
        const matrix::Csr<ValueType, IndexType>* t_matrix)
    {
        const auto n = matrix->get_size()[0];
        const auto nz = matrix->get_num_stored_elements();
        const auto nrhs = one<gko::size_type>();

        array<IndexType> factor_sizes(exec, n);
        array<IndexType> factor_assignments_(exec, n);
        cudaMemset(factor_sizes.get_data(), 0, n * sizeof(IndexType));
        cudaMemset(factor_assignments_.get_data(), 0xFF, n * sizeof(IndexType));
        // factor_sizes.fill(zero<IndexType>());
        // factor_assignments_.fill(-one<IndexType>());

        sptrsvdrpi_create_ppi(
            exec, matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            t_matrix->get_const_row_ptrs(), t_matrix->get_const_col_idxs(),
            factor_sizes.get_data(), factor_assignments_.get_data(),
            // purely_diagonal_elements.get_const_data(),
            // purely_diagonal_count.get_const_data()[0],
            // &rem_preds[0],
            // counts.get_data(),
            (IndexType)n,
            // (IndexType)nz,
            &m, one<ValueType>());


        components::prefix_sum_nonnegative(exec, factor_sizes.get_data(), m);


        array<IndexType> perm(exec, n);
        thrust::sequence(thrust::device, perm.get_data(), perm.get_data() + n);
        // thrust::stable_sort_by_key(thrust::device, levels.get_data(),
        // levels.get_data() + n, level_perm.get_data());
        thrust::stable_sort_by_key(
            thrust::device, factor_assignments_.get_data(),
            factor_assignments_.get_data() + n, perm.get_data());


        factor_perm = matrix::Permutation<IndexType>::create(exec, perm);
        factor_offsets =
            std::make_unique<array<IndexType>>(std::move(factor_sizes));
        factor_assignments =
            std::make_unique<array<IndexType>>(std::move(factor_assignments_));

        // INFO
        printf("PPi m is %d\n", (int32)m);
    }

    void solve(std::shared_ptr<const CudaExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* matrix,
               const matrix::Dense<ValueType>* b,
               matrix::Dense<ValueType>* x) const
    {
        const auto n = matrix->get_size()[0];
        const auto nrhs = b->get_size()[1];

        auto new_b = matrix::Dense<ValueType>::create(exec, gko::dim<2>(n, 1));
        new_b->copy_from(b);
        diag->inverse_apply(b, new_b.get());
        new_b = new_b->permute(factor_perm, matrix::permute_mode::rows);
        x->fill(zero<ValueType>());
        x_tmp->fill(zero<ValueType>());

        // const dim3 block_size_2(default_block_size, 1,1);
        // const dim3 grid_size_2(ceildiv(n * nrhs, block_size_2.x), 1, 1);
        // // DEBUG
        // sptrsvppi_debug_bb_kernel<<<grid_size_2,block_size_2>>>(
        //     as_cuda_type(b->get_const_values()),
        //     as_cuda_type(new_b->get_const_values()), n, one<IndexType>());


        // WIP BELOW

        exec->copy(2 * n, row_counts.get_const_data(), row_counts_clone.get_data());

        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(ceildiv(2 * n, block_size.x), 1, 1);
        sptrsvppi_solve_rscpi_kernel<<<grid_size, block_size>>>(
            unit_inv_p_m->get_const_row_ptrs(),
            unit_inv_p_m->get_const_col_idxs(),
            as_cuda_type(unit_inv_p_m->get_const_values()),
            factor_assignments->get_const_data(),
            col_factor_ends.get_const_data(),
            row_counts_clone.get_data(),
            as_cuda_type(new_b->get_const_values()),
            as_cuda_type(x_tmp->get_data()), as_cuda_type(x->get_values()), n);

        const auto final_x =
            x->permute(factor_perm, matrix::permute_mode::inverse_rows);
        cudaMemcpy(x->get_values(), final_x->get_const_values(),
                   n * sizeof(decltype(as_cuda_type(zero<ValueType>()))),
                   cudaMemcpyDeviceToDevice);


        // WIP ABOVE


        // DEBUG BELOW

        //     // Initialize x to all NaNs.
        //     const auto clone_x = matrix::Dense<ValueType>::create(exec);
        //     clone_x->copy_from(b);
        //     dense::fill(exec, clone_x.get(), nan<ValueType>());

        //     array<bool> nan_produced(exec, 1);
        //     array<IndexType> atomic_counter(exec, 1);
        //     sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
        //                                  atomic_counter.get_data());

        //    const dim3 block_size_3(default_block_size, 1, 1);
        //             const dim3 grid_size_3(ceildiv(n, block_size_3.x), 1, 1);
        //     sptrsv_naive_caching_kernel<false><<<grid_size_3,block_size_3>>>(
        //         matrix->get_const_row_ptrs(),
        //         matrix->get_const_col_idxs(),
        //         as_cuda_type(matrix->get_const_values()),
        //         as_cuda_type(b->get_const_values()), b->get_stride(),
        //         as_cuda_type(clone_x->get_values()), clone_x->get_stride(),
        //         n, nrhs, false, nan_produced.get_data(),
        //         atomic_counter.get_data());

        //     const auto clone_final_x = clone_x->permute(factor_perm,
        //     matrix::permute_mode::rows); const auto computed_final_x =
        //     x->permute(factor_perm, matrix::permute_mode::rows);

        //     const auto dbg_tmp_0 = 0;
        //     sptrsvppi_debug_kernel<<<grid_size_3,block_size_3>>>(
        //         as_cuda_type(clone_final_x->get_const_values()),
        //         as_cuda_type(computed_final_x->get_const_values()), n,
        //         one<IndexType>());


        // DEBUG ABOVE
    }
};


// TODO: This namespace commented out for ffi
// }  // namespace
}  // namespace cuda
}  // namespace kernels


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::generate<ValueType, IndexType>(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix,
    std::shared_ptr<solver::SolveStruct>& solve_struct,
    const gko::size_type num_rhs,
    const gko::solver::trisolve_strategy* strategy, bool is_upper,
    bool unit_diag)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if (strategy->type == gko::solver::trisolve_type::sparselib) {
        if (gko::kernels::cuda::cusparse::is_supported<ValueType,
                                                       IndexType>::value) {
            solve_struct = std::make_shared<
                gko::kernels::cuda::CudaSolveStruct<ValueType, IndexType>>(
                exec, matrix, num_rhs, is_upper, unit_diag);
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else if (strategy->type == gko::solver::trisolve_type::level) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvlrSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::winv) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebrwiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::wvar) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebrwvSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::thinned) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebcrnmSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, strategy->thinned_m);
    } else if (strategy->type == gko::solver::trisolve_type::block) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::BlockedSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, strategy->block_inner);
    } else if (strategy->type == gko::solver::trisolve_type::udiag) {
        solve_struct =
            std::make_shared<gko::kernels::cuda::SptrsvebcrnutsSolveStruct<
                ValueType, IndexType>>(exec, matrix, num_rhs, is_upper);
    } else if (strategy->type == gko::solver::trisolve_type::pdi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebrpdiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, 8);
    } else if (strategy->type == gko::solver::trisolve_type::syncfree) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::ppi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvppiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::rdpi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvrdpiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::rspi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvrspiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::cscb) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebccbnSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::csc) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvebccnSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::rscpi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvrscpiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag);
    } else if (strategy->type == gko::solver::trisolve_type::rdmpi) {
        solve_struct = std::make_shared<
            gko::kernels::cuda::SptrsvrdmpiSolveStruct<ValueType, IndexType>>(
            exec, matrix, num_rhs, is_upper, unit_diag, strategy->rdmpi_m);
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_GENERATE(_vtype, _itype)        \
    void gko::solver::SolveStruct::generate(                           \
        std::shared_ptr<const CudaExecutor> exec,                      \
        const matrix::Csr<_vtype, _itype>* matrix,                     \
        std::shared_ptr<solver::SolveStruct>& solve_struct,            \
        const gko::size_type num_rhs,                                  \
        const gko::solver::trisolve_strategy* strategy, bool is_upper, \
        bool unit_diag)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_GENERATE);


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::solve(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix,
    matrix::Dense<ValueType>* trans_b, matrix::Dense<ValueType>* trans_x,
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x) const
{
    if (matrix->get_size()[0] == 0 || b->get_size()[1] == 0) {
        return;
    }
    if (auto sptrsvebcrn_struct =
            dynamic_cast<const gko::kernels::cuda::SptrsvebcrnSolveStruct<
                ValueType, IndexType>*>(this)) {
        sptrsvebcrn_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvlr_struct =
                   dynamic_cast<const gko::kernels::cuda::SptrsvlrSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvlr_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvebcrnm_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebcrnmSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvebcrnm_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvebrwi_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebrwiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvebrwi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvb_struct =
                   dynamic_cast<const gko::kernels::cuda::BlockedSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvb_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvwv_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebrwvSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvwv_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvud_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebcrnutsSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvud_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvppi_struct =
                   dynamic_cast<const gko::kernels::cuda::SptrsvppiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvppi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvrdpi_struct =
                   dynamic_cast<const gko::kernels::cuda::SptrsvrdpiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvrdpi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvrspi_struct =
                   dynamic_cast<const gko::kernels::cuda::SptrsvrspiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvrspi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvrdmpi_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvrdmpiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvrdmpi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvpi_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebrpdiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvpi_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvcscb_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebccbnSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvcscb_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvcsc_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvebccnSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvcsc_struct->solve(exec, matrix, b, x);
    } else if (auto sptrsvcscrspi_struct = dynamic_cast<
                   const gko::kernels::cuda::SptrsvrscpiSolveStruct<
                       ValueType, IndexType>*>(this)) {
        sptrsvcscrspi_struct->solve(exec, matrix, b, x);
    }else if (gko::kernels::cuda::cusparse::is_supported<
                   ValueType,
                   IndexType>::value) {  // Must always be last check
        if (auto cuda_solve_struct =
                dynamic_cast<const gko::kernels::cuda::CudaSolveStruct<
                    ValueType, IndexType>*>(this)) {
            cuda_solve_struct->solve(matrix, b, x, trans_b, trans_x);
        } else {
            GKO_NOT_SUPPORTED(this);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_SOLVE(_vtype, _itype)            \
    void gko::solver::SolveStruct::solve(                               \
        std::shared_ptr<const CudaExecutor> exec,                       \
        const matrix::Csr<_vtype, _itype>* matrix,                      \
        matrix::Dense<_vtype>* trans_b, matrix::Dense<_vtype>* trans_x, \
        const matrix::Dense<_vtype>* b, matrix::Dense<_vtype>* x) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_SOLVE);


template <typename ValueType, typename IndexType>
bool gko::solver::SolveStruct::check_solvability(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* matrix) const
{
    if (auto sptrsv_solve_struct =
            dynamic_cast<const gko::kernels::cuda::SptrsvebcrnSolveStruct<
                ValueType, IndexType>*>(this)) {
        return sptrsv_solve_struct->check_solvability(exec, matrix);
    } else {
        return false;
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_CHECK_SOLVABILITY(_vtype, _itype) \
    bool gko::solver::SolveStruct::check_solvability(                    \
        std::shared_ptr<const CudaExecutor> exec,                        \
        const matrix::Csr<_vtype, _itype>* matrix) const

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_CHECK_SOLVABILITY);


template <typename ValueType, typename IndexType>
IndexType* gko::solver::SolveStruct::get_row_mappings(
    std::shared_ptr<const CudaExecutor> exec, ValueType dummy)
{
    if (auto sptrsv_solve_struct = dynamic_cast<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>*>(
            this)) {
        sptrsv_solve_struct->row_mappings.set_executor(exec->get_master());
        return sptrsv_solve_struct->row_mappings.get_data();
    } else {
        return NULL;
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_GET_ROW_MAPPINGS(_vtype, _itype) \
    _itype* gko::solver::SolveStruct::get_row_mappings(                 \
        std::shared_ptr<const CudaExecutor> exec, _vtype dummy)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_GET_ROW_MAPPINGS);


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::set_row_mappings(
    std::shared_ptr<const CudaExecutor> exec, IndexType* new_mappings, uint32 n,
    ValueType dummy)
{
    if (auto sptrsv_solve_struct = dynamic_cast<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>*>(
            this)) {
        cudaMemcpy(new_mappings, sptrsv_solve_struct->row_mappings.get_data(),
                   n * sizeof(IndexType), cudaMemcpyHostToDevice);
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_SET_ROW_MAPPINGS(_vtype, _itype) \
    void gko::solver::SolveStruct::set_row_mappings(                    \
        std::shared_ptr<const CudaExecutor> exec, _itype* new_mappings, \
        uint32 n, _vtype dummy)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_SET_ROW_MAPPINGS);


template <typename ValueType, typename IndexType>
IndexType* gko::solver::SolveStruct::get_row_mapping_starts(
    std::shared_ptr<const CudaExecutor> exec, ValueType dummy)
{
    if (auto sptrsv_solve_struct = dynamic_cast<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>*>(
            this)) {
        sptrsv_solve_struct->row_mapping_starts.set_executor(
            exec->get_master());
        return sptrsv_solve_struct->row_mapping_starts.get_data();
    } else {
        return NULL;
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_GET_ROW_MAPPING_STARTS(_vtype, _itype) \
    _itype* gko::solver::SolveStruct::get_row_mapping_starts(                 \
        std::shared_ptr<const CudaExecutor> exec, _vtype dummy)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_GET_ROW_MAPPING_STARTS);


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::apply_swap(
    std::shared_ptr<const CudaExecutor> exec, uint32 a, uint32 b,
    ValueType dummy1, IndexType dummy2)
{
    if (auto sptrsv_solve_struct = dynamic_cast<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>*>(
            this)) {
        thrust::swap(sptrsv_solve_struct->row_mappings.get_data()[a],
                     sptrsv_solve_struct->row_mappings.get_data()[b]);
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_APPLY_SWAP(_vtype, _itype)     \
    void gko::solver::SolveStruct::apply_swap(                        \
        std::shared_ptr<const CudaExecutor> exec, uint32 a, uint32 b, \
        _vtype dummy1, _itype dummy2)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_APPLY_SWAP);


template <typename ValueType, typename IndexType>
void gko::solver::SolveStruct::sync_to_gpu(
    std::shared_ptr<const CudaExecutor> exec, ValueType dummy1,
    IndexType dummy2)
{
    if (auto sptrsv_solve_struct = dynamic_cast<
            gko::kernels::cuda::SptrsvebcrnSolveStruct<ValueType, IndexType>*>(
            this)) {
        sptrsv_solve_struct->sync_to_gpu(exec);
    }
}

#define GKO_DECLARE_SOLVER_SOLVESTRUCT_SYNC_TO_GPU(_vtype, _itype) \
    void gko::solver::SolveStruct::sync_to_gpu(                    \
        std::shared_ptr<const CudaExecutor> exec, _vtype dummy1,   \
        _itype dummy2)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOLVER_SOLVESTRUCT_SYNC_TO_GPU);


void should_perform_transpose_kernel(std::shared_ptr<const CudaExecutor> exec,
                                     bool& do_transpose)
{
    do_transpose = false;
}


}  // namespace gko
