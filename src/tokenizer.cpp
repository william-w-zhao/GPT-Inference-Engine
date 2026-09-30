#include "tokenizer.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

std::vector<std::string> load_vocab(std::string &path)
{
    std::ifstream f(path, std::ifstream::binary);
    if (!f)
    {
        throw std::runtime_error("unable to open vocab file");
    }

    int32_t count;
    f.read(reinterpret_cast<char *>(&count), sizeof(count));

    std::vector<std::string> vocab(count);

    for (int i = 0; i < count; i++)
    {
        int32_t len;
        f.read(reinterpret_cast<char *>(&len), sizeof(len));
        vocab[i].resize(len);
        f.read(vocab[i].data(), len);
    }
    return vocab;
}