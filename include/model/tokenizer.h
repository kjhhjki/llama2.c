#pragma once
#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, TokenID> vocab_map;
    std::vector<f32> bpe_scores;
};

void init_tokenizer(Tokenizer &&initializer);
std::vector<TokenID> encode(const std::string&, bool, bool);
std::string decode(std::span<const TokenID>);