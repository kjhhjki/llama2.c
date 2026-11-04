#include "ops/attention.h"

#ifdef CUDA
#else  // CUDA
void attention(
    Vector<bf16> &o,
    const Vector<bf16> &q,
    const Tensor<bf16, 3> &key_cache,
    const Tensor<bf16, 3> &value_cache,
    Matrix<f32> &scores,
    size_t layer,
    size_t pos,
    size_t head_size
) {
    if (q.empty() || head_size == 0) {
        throw std::invalid_argument("Invalid query dimensions");
    }
    if (key_cache.shape() != value_cache.shape()) {
        throw std::invalid_argument("K/V cache shapes must match");
    }
    const size_t dim = q.size(),
                 kv_dim = key_cache.shape()[2];
    if (kv_dim == 0 || dim % head_size != 0 || kv_dim % head_size != 0) {
        throw std::invalid_argument("Invalid attention dimensions");
    }
    const size_t n_heads = dim / head_size,
                 n_kv_heads = kv_dim / head_size;
    if (n_heads % n_kv_heads != 0) {
        throw std::invalid_argument("Invalid GQA configuration");
    }
    const size_t n_layers = key_cache.shape()[0],
                 seq_len = key_cache.shape()[1];
    if (layer >= n_layers) {
        throw std::out_of_range("Layer index out of bounds");
    }
    if (pos >= seq_len) {
        throw std::out_of_range("Position index out of bounds");
    }
    if (o.size() != dim) {
        o.resize(dim);
    }

    const Matrix<f32>::ShapeType expected_shape{
        n_heads, seq_len
    };
    if (scores.shape() != expected_shape) {
        scores.resize(expected_shape);
    }
    kernel::attention(
        o.view(),
        q.view(),
        key_cache.view(),
        value_cache.view(),
        scores.view(),
        layer,
        pos,
        head_size
    );
}
#endif // CUDA