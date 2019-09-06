/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_CONSTEXPR_MAP_H
#define SHOGUN_CONSTEXPR_MAP_H

#include <array>
#include <string_view>

constexpr size_t hash_string(std::string_view str)
{
    // For this example, I'm requiring size_t to be 64-bit, but you could
    // easily change the offset and prime used to the appropriate ones
    // based on sizeof(size_t).
    static_assert(sizeof(size_t) == 8);
    // FNV-1a 64 bit algorithm
    size_t result = 0xcbf29ce484222325; // FNV offset basis

    for (char c : str) {
        result ^= c;
        result *= 1099511628211; // FNV prime
    }

    return result;
}

template <typename T, size_t size>
class HashTable
{
public:
    struct Pair{
        std::string_view   key;
        T                  value;
    };

    constexpr HashTable(const std::initializer_list<Pair>& objs): m_array()
    {
        for (auto &x : m_array)
            x = T();

        for(const Pair &p : objs)
            push_(p.key, p.value);
    }

    constexpr bool collisions() const{
        return collisions_;
    }

    constexpr T& operator[](std::string_view key) const{
        auto const bucket = bucket__(key);

        return m_array[bucket];
    }

    constexpr T& at(std::string_view key) const{
        auto const bucket = bucket__(key);
        return m_array[bucket];
    }

    constexpr T* begin() const {
        return m_array.begin();
    }

    constexpr T* end() const {
        return m_array.end();
    }

private:
    constexpr void push_(std::string_view key, const T &value){
        auto const bucket = bucket__(key);

        auto &element = m_array[bucket];

        if (element != T())
            collisions_ = true;

        element = value;
    }

    constexpr static size_t bucket__(std::string_view key){
        return hash_string(key) % size;
    }

    std::array<T, size> m_array;
    bool collisions_ = false;
};

#endif //SHOGUN_CONSTEXPR_MAP_H
