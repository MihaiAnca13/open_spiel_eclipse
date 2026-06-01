//
// Created by Mihai on 01/06/2026.
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_FIXED_VECTOR_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_FIXED_VECTOR_H_

#include <array>
#include "open_spiel/json/include/nlohmann/json.hpp"

namespace open_spiel {
namespace eclipse {

template<typename T, size_t N>
class FixedVector {
private:
    std::array<T, N> data_;
    size_t size_ = 0;

public:
    using iterator = typename std::array<T, N>::iterator;
    using const_iterator = typename std::array<T, N>::const_iterator;

    FixedVector() : size_(0) {
        data_.fill(T{});
    }

    void clear() { size_ = 0; }
    void push_back(const T& val) {
        if (size_ < N) {
            data_[size_++] = val;
        }
    }
    void pop_back() {
        if (size_ > 0) {
            size_--;
        }
    }
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
    size_t capacity() const { return N; }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T& front() { return data_[0]; }
    const T& front() const { return data_[0]; }

    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.begin(); }
    iterator end() { return data_.begin() + size_; }
    const_iterator end() const { return data_.begin() + size_; }

    const std::array<T, N>& data() const { return data_; }
};

template<typename T, size_t N>
inline void to_json(nlohmann::json& j, const FixedVector<T, N>& v) {
    j = nlohmann::json::array();
    for (size_t i = 0; i < v.size(); ++i) {
        j.push_back(v[i]);
    }
}

template<typename T, size_t N>
inline void from_json(const nlohmann::json& j, FixedVector<T, N>& v) {
    v.clear();
    for (const auto& el : j) {
        v.push_back(el.get<T>());
    }
}

} // namespace eclipse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_ECLIPSE_FIXED_VECTOR_H_
