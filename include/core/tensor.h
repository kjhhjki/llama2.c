#pragma once
#include "core/types.h"
#include <array>
#include <concepts>
#include <mdspan>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <limits>
#include <utility>

#ifdef CUDA

template <typename T, std::size_t N>
class Tensor { };

#else
template <typename T, std::size_t N>
requires (N > 0)
class Tensor 
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using ShapeType = std::array<size_type, N>;
    using Extents = std::dextents<size_type, N>;
    using ViewType = std::mdspan<value_type, Extents, std::layout_right>;
    using ConstViewType = std::mdspan<const value_type, Extents, std::layout_right>;
    using VectorViewType = std::mdspan<value_type, std::dextents<size_type, 1>, std::layout_right>;
    using ConstVectorViewType = std::mdspan<const value_type, std::dextents<size_type, 1>, std::layout_right>;

private:
    std::vector<value_type> data_;
    ShapeType shape_{};

    static size_type calc_size(const ShapeType& shape) {
        size_type result = 1;
        for (size_type dim : shape) {
#ifdef CHECK_BOUNDS
            if (dim != 0 && result > std::numeric_limits<size_type>::max() / dim) {
                throw std::length_error("Tensor size overflow");
            }
#endif
            result *= dim;
        }
        return result;
    }

    template <std::integral I>
    static size_type to_size(I value) {
#ifdef CHECK_BOUNDS
        if constexpr (std::signed_integral<I>) {
            if (value < 0) {
                throw std::out_of_range("Negative tensor index or extent");
            }
        }
#endif
        return static_cast<size_type>(value);
    }

    template <size_type... I>
    requires (sizeof...(I) == N)
    ViewType make_view(std::index_sequence<I...>) & noexcept 
    { return ViewType(data_.data(), shape_[I]...); }

    template <size_type... I>
    requires (sizeof...(I) == N)
    ConstViewType make_view(std::index_sequence<I...>) const & noexcept 
    { return ConstViewType(data_.data(), shape_[I]...); }

    template <size_type... I>
    requires (sizeof...(I) == N)
    value_type& access(const ShapeType& indices, std::index_sequence<I...>) & 
    { return view()[indices[I]...]; }

    template <size_type... I>
    requires (sizeof...(I) == N)
    const value_type& access(const ShapeType& indices, std::index_sequence<I...>) const & 
    { return view()[indices[I]...]; }

public:
    explicit Tensor(const ShapeType& shape) : 
    data_(calc_size(shape)), shape_(shape) {}
    ViewType view() & noexcept 
    { return make_view(std::make_index_sequence<N>{}); }
    ConstViewType view() const & noexcept 
    { return make_view(std::make_index_sequence<N>{}); }
    ViewType view() && = delete;
    ConstViewType view() const && = delete;
    operator ViewType() & noexcept { return view(); }
    operator ConstViewType() const & noexcept { return view(); }

    template <std::integral... Indices>
    requires (sizeof...(Indices) == N)
    value_type& operator()(Indices... indices) & 
    { return view()[to_size(indices)...]; }
    template <std::integral... Indices>
    requires (sizeof...(Indices) == N)
    const value_type& operator()(Indices... indices) const & 
    { return view()[to_size(indices)...]; }
    value_type& operator()(const ShapeType& indices) & 
    { return access(indices, std::make_index_sequence<N>{}); }
    const value_type& operator()(const ShapeType& indices) const & 
    { return access(indices, std::make_index_sequence<N>{}); }

    template <std::integral... Indices>
    requires (sizeof...(Indices) == N)
    value_type& at(Indices... indices) & 
    { return at(ShapeType{to_size(indices)...}); }
    template <std::integral... Indices>
    requires (sizeof...(Indices) == N)
    const value_type& at(Indices... indices) const & 
    { return at(ShapeType{to_size(indices)...}); }
    value_type& at(const ShapeType& indices) & {
#ifdef CHECK_BOUNDS
        for (size_type i = 0; i < N; ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
#endif
        return access(indices, std::make_index_sequence<N>{});
    }
    const value_type& at(const ShapeType& indices) const & {
#ifdef CHECK_BOUNDS
        for (size_type i = 0; i < N; ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
#endif
        return access(indices, std::make_index_sequence<N>{});
    }

    template <std::integral... Indices>
    requires (N > 1 && sizeof...(Indices) == N - 1)
    VectorViewType row(Indices... indices) & {
        const std::array<size_type, N - 1> idx{to_size(indices)...};
        size_type offset = 0;
        for (size_type i = 0; i < N - 1; ++i) {
    #ifdef CHECK_BOUNDS
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Tensor row out of bounds");
            }
    #endif
            offset = offset * shape_[i] + idx[i];
        }
        offset *= shape_[N - 1];
        auto* ptr = data_.data();
        if (offset != 0) {
            ptr += offset;
        }
        return VectorViewType(ptr, shape_[N - 1]);
    }
    template <std::integral... Indices>
    requires (N > 1 && sizeof...(Indices) == N - 1)
    ConstVectorViewType row(Indices... indices) const & {
        const std::array<size_type, N - 1> idx{to_size(indices)...};
        size_type offset = 0;
        for (size_type i = 0; i < N - 1; ++i) {
    #ifdef CHECK_BOUNDS
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Tensor row out of bounds");
            }
    #endif
            offset = offset * shape_[i] + idx[i];
        }
        offset *= shape_[N - 1];
        const auto* ptr = data_.data();
        if (offset != 0) {
            ptr += offset;
        }
        return ConstVectorViewType(ptr, shape_[N - 1]);
    }

    template <typename... Dims>
    requires (sizeof...(Dims) == N)
    void reshape(Dims... dims) { reshape(ShapeType{to_size(dims)...}); }
    void reshape(const ShapeType& shape) {
#ifdef CHECK_BOUNDS
        if (calc_size(shape) != data_.size()) {
            throw std::invalid_argument("New shape must preserve tensor size");
        }
#endif
        shape_ = shape;
    }
    template <typename... Dims>
    requires (sizeof...(Dims) == N)
    void resize(Dims... dims) { resize(ShapeType{to_size(dims)...}); }
    void resize(const ShapeType& shape) { data_.resize(calc_size(shape)); shape_ = shape; }

    const ShapeType& shape() const noexcept { return shape_; }
    size_type size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }
    value_type* data() noexcept { return data_.data(); }
    const value_type* data() const noexcept { return data_.data(); }
};

#endif

template <typename T> 
using Vector = Tensor<T, 1>;
template <typename T> 
using Matrix = Tensor<T, 2>; 