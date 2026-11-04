#include "kernel/cpu/attention.h"
#include "kernel/cpu/primitives.h"
#include "kernel/cpu/rmsnorm.h"
#include "kernel/cpu/matvec.h"
#include "kernel/cpu/softmax.h"
#include "kernel/functional.h"
#include <cmath>

void kernel::attention(
        Vector<bf16>::ViewType o,
        Vector<bf16>::ConstViewType q,
        Tensor<bf16, 3>::ConstViewType key_cache,
        Tensor<bf16, 3>::ConstViewType value_cache,
        Matrix<f32>::ViewType scores,
        size_t layer,
        size_t pos,
        size_t head_size
) {
    const size_t dim = q.extent(0),
                 seq_len = key_cache.extent(1),
                 kv_dim = key_cache.extent(2);

    const size_t n_heads = dim / head_size,
                 n_kv_heads = kv_dim / head_size,
                 kv_mul = n_heads / n_kv_heads;

    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(head_size));
    #pragma omp parallel for
    for (std::ptrdiff_t hh = 0; hh < static_cast<std::ptrdiff_t>(n_heads); ++hh) {
        const size_t h = static_cast<size_t>(hh);

        const size_t kv_head = h / kv_mul,
                     kv_offset = kv_head * head_size;
        const std::span<const bf16> q_head(
            q.data_handle() + h * head_size,
            head_size
        );
        const KVHeadView k_head = std::submdspan(
            key_cache,
            layer,
            std::pair{size_t{0}, pos + 1},
            std::pair{kv_offset, kv_offset + head_size}
        );
        const KVHeadView v_head = std::submdspan(
            value_cache,
            layer,
            std::pair{size_t{0}, pos + 1},
            std::pair{kv_offset, kv_offset + head_size}
        );
        std::span<f32> head_scores(
            scores.data_handle() + h * seq_len,
            pos + 1
        );
        for (size_t t = 0; t <= pos; ++t) {
            const std::span<const bf16> k_row(
                k_head.data_handle() + t * k_head.stride(0),
                head_size
            );
            head_scores[t] = kernel::dot(q_head, k_row) * scale;
        }
        softmax(head_scores);
        std::span<bf16> out_head(
            o.data_handle() + h * head_size,
            head_size
        );
        weightedSum(out_head, v_head, head_scores);
    }
} 