#pragma once
#include <stdexcept>

class Config 
{
public:
    constexpr Config(int dim, int hidden_dim, int n_layers, 
                     int n_heads, int n_kv_heads, 
                     int vocab_size, int seq_len)
        : dim(dim), hidden_dim(hidden_dim), n_layers(n_layers), 
          n_heads(n_heads), n_kv_heads(n_kv_heads), 
          vocab_size(vocab_size), seq_len(seq_len) {
            if (n_heads <= 0 || n_kv_heads <= 0 || dim <= 0 || hidden_dim <= 0 || 
                n_layers <= 0 || vocab_size <= 0 || seq_len <= 0 ) {
                throw std::invalid_argument("All configuration parameters must be positive integers");
            }
            if (dim % n_heads != 0) {
                throw std::invalid_argument("dim must be divisible by n_heads");
            }
            if (dim % n_kv_heads != 0) {
                throw std::invalid_argument("dim must be divisible by n_kv_heads");
            }
            if (n_heads % n_kv_heads != 0) {
                throw std::invalid_argument("n_heads must be divisible by n_kv_heads");
            }
        }

    constexpr int get_dim() const { return dim; }
    constexpr int get_hidden_dim() const { return hidden_dim; }
    constexpr int get_n_layers() const { return n_layers; }
    constexpr int get_n_heads() const { return n_heads; }
    constexpr int get_n_kv_heads() const { return n_kv_heads; }
    constexpr int get_vocab_size() const { return vocab_size; }
    constexpr int get_seq_len() const { return seq_len; }
    constexpr int get_head_size() const { return dim / n_heads; }
    constexpr int get_kv_dim() const { return (dim / n_heads) * n_kv_heads; }
    constexpr int get_kv_mul() const { return n_heads / n_kv_heads; } // integer multiplier of the kv sharing in multiquery

private:
    int dim;            // transformer dimension
    int hidden_dim;     // for ffn layers
    int n_layers;       // number of layers
    int n_heads;        // number of query heads
    int n_kv_heads;     // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;     // vocabulary size, usually 256 (byte-level)
    int seq_len;        // max sequence length
};