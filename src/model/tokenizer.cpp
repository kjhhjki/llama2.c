#include "model/tokenizer.h"
#include <queue>
#include <numeric>
#include <string_view>
#include <cstdio>

namespace 
{
Tokenizer tokenizer;

class Dsu
{
private:
    std::vector<size_t> pre, sz, l;
public:
    Dsu(size_t n) : pre(n) {
        sz.assign(n, 1);
        std::iota(pre.begin(), pre.end(), 0);   
        l = pre;
    }
    size_t find(size_t x) {
        while (x != pre[x]) {
            x = pre[x] = pre[pre[x]];
        }
        return x;
    }
    bool merge(size_t x, size_t y) {
        x = find(x); y = find(y);
        if (x == y) {
            return false;
        }
        if (sz[x] < sz[y]) {
            std::swap(x, y);
        }
        pre[y] = x;
        l[x] = std::min(l[x], l[y]);
        sz[x] += sz[y];
        return true;
    }
    size_t size(size_t x) {
        return sz[find(x)];
    }
    std::pair<size_t, size_t> range(size_t x) {
        x = find(x);
        return {l[x], l[x] + sz[x]};
    }
    size_t left(size_t x) {
        return l[find(x)];
    }
};

using T = std::tuple<f32, size_t, TokenID, TokenID, TokenID>;
struct cmp {
    bool operator()(const T& a, const T& b) {
        if (std::get<0>(a) == std::get<0>(b)) {
            return std::get<1>(a) > std::get<1>(b);
        }
        return std::get<0>(a) < std::get<0>(b);
    }
};

void bpe(std::vector<TokenID>& tokens) 
{
    // (score, p, l, r, merged)
    std::priority_queue<T, std::vector<T>, cmp> q;
    Dsu dsu(tokens.size());

    auto push = [&q, &dsu, &tokens](size_t pl, size_t pr) {
        auto [ll, lr] = dsu.range(pl);
        auto [rl, rr] = dsu.range(pr);
        if (ll == rl || lr != rl) {
            return;
        }
        const TokenID left_id = tokens[ll], right_id = tokens[rl];
        const std::string merged =
                tokenizer.vocab[left_id] +
                tokenizer.vocab[right_id];
        auto it = tokenizer.vocab_map.find(merged);
        if (it == tokenizer.vocab_map.end()) {
            return;
        }
        const TokenID merged_id = it->second;
        q.emplace(tokenizer.bpe_scores[merged_id], lr, left_id, right_id, merged_id);
    };

    for (size_t i = 1; i < tokens.size(); i++) {
        push(i - 1, i);
    }
    while (!q.empty()) { 
        auto [_, p, _l, _r, m] = q.top(); q.pop();
        if (m == -1) {
            continue;
        }
        auto [ll, lr] = dsu.range(p - 1);
        auto [rl, rr] = dsu.range(p);
        if (lr != p || rl != p || 
            tokens[ll] != _l || tokens[rl] != _r) {
            continue;
        }
        dsu.merge(p - 1, p);
        auto [pl, pr] = dsu.range(p);
        tokens[pl] = m;
        if (pl > 0) {
            push(pl - 1, pl);
        }
        if (pr < tokens.size()) {
            push(pr - 1, pr);
        }
    }
    std::vector<TokenID> new_tokens;
    for (size_t l = 0; l < tokens.size(); l = dsu.range(l).second) {
        new_tokens.push_back(tokens[l]);
    }
    std::swap(tokens, new_tokens);
}

constexpr TokenID bos_id = 1;
constexpr TokenID eos_id = 2;

}

void init_tokenizer(Tokenizer &&initializer) {
    tokenizer = std::move(initializer);
}

std::vector<TokenID> encode(
    std::string_view text, 
    bool bos = true, 
    bool eos = true
) {
    std::vector<TokenID> tokens;
    tokens.reserve(text.size() + 1 +
                    static_cast<size_t>(bos) +
                    static_cast<size_t>(eos));
    if (bos) {
        tokens.push_back(bos_id);
    }
    if (!text.empty()) {
        auto it = tokenizer.vocab_map.find(std::string(text.substr(0, 1)));
        if (it == tokenizer.vocab_map.end()) {
            throw std::invalid_argument("Missing dummy prefix token");
        }
        tokens.push_back(it->second);
    }
    for (size_t i = 0, len = 0; i < text.length();) {
        const unsigned char head = text[i];
        len = 1 + (head >= 0xC2) + (head >= 0xE0) + (head >= 0xF0);
        bool valid = i + len <= text.size();
        for (size_t j = 1; valid && j < len; ++j) {
            valid = (text[i + j] & 0xC0) == 0x80;
        }
        if (valid && len > 2) {
            const unsigned char byte1 = text[i + 1];
            if (len == 3) {
                valid = !(head == 0xE0 && byte1 < 0xA0) &&
                        !(head == 0xED && byte1 >= 0xA0);
            } else if (len == 4) {
                valid = !(head == 0xF0 && byte1 < 0x90) &&
                        !(head == 0xF4 && byte1 >= 0x90);
            }
        }
        if (!valid) {
            len = 1;    
        }
        const std::string piece(text.substr(i, len));
        auto it = tokenizer.vocab_map.find(piece);
        if (it != tokenizer.vocab_map.end()) {
            tokens.push_back(it->second);
        } else {
            for (size_t j = 0; j < len; ++j) {
                const unsigned char byte = text[i + j];
                tokens.push_back(static_cast<TokenID>(byte) + 3);
            }
        }
        i += len;
    }
    bpe(tokens);
    if (eos) {
        tokens.push_back(eos_id);
    }
    return tokens;
}

std::string decode(std::span<const TokenID> tokens) 
{
    std::string text;
    for (TokenID prev_token = -1; const TokenID token : tokens) {
        if (token < 0 ||
            static_cast<size_t>(token) >= tokenizer.vocab.size()) {
            throw std::out_of_range("Invalid Token ID");
        }
        if (token == bos_id || token == eos_id) {
            prev_token = token;
            continue;
        }
        const std::string &piece = tokenizer.vocab[token];
        size_t offset = prev_token == bos_id && !piece.empty() && piece[0] == ' ';
        unsigned char byte;
        if (sscanf(piece.c_str() + offset, "<0x%02hhX>", &byte) == 1) {
            text.push_back(byte);
        } else {
            text.append(piece, offset, std::string::npos);
        }
        prev_token = token;
    }
    return text;
}