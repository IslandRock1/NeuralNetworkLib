
// This file is written by ChatGPT.

#ifndef NEURALNETWORKLIB_MNIST_LOADER_HPP
#define NEURALNETWORKLIB_MNIST_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

namespace mnist
{

struct MNISTDataset
{
    int numImages = 0;
    int rows = 0;
    int cols = 0;

    // Each image is a flat vector of size rows*cols
    std::vector<std::vector<float>> images;

    // Labels in range [0,9]
    std::vector<uint8_t> labels;
};

// ------------------------------------------------------------
// Helper: read 32-bit big-endian integer
// ------------------------------------------------------------
inline int readBigEndianInt(std::ifstream& file)
{
    uint8_t b[4];
    file.read(reinterpret_cast<char*>(b), 4);

    return (int(b[0]) << 24) |
           (int(b[1]) << 16) |
           (int(b[2]) << 8)  |
            int(b[3]);
}

// ------------------------------------------------------------
// Load MNIST dataset
// ------------------------------------------------------------
inline MNISTDataset load(
    const std::string& imageFile,
    const std::string& labelFile)
{
    MNISTDataset dataset;

    // =======================
    // Load images
    // =======================
    std::ifstream img(imageFile, std::ios::binary);
    if (!img)
        throw std::runtime_error("MNIST: cannot open image file");

    int magic = readBigEndianInt(img);

    if (magic != 2051) {
        throw std::runtime_error("MNIST: invalid image file magic");
    }

    dataset.numImages = readBigEndianInt(img);
    dataset.rows      = readBigEndianInt(img);
    dataset.cols      = readBigEndianInt(img);

    const int imageSize = dataset.rows * dataset.cols;
    dataset.images.resize(dataset.numImages,
                          std::vector<float>(imageSize));

    std::vector<uint8_t> buffer(imageSize);

    for (int i = 0; i < dataset.numImages; ++i)
    {
        img.read(reinterpret_cast<char*>(buffer.data()), imageSize);

        for (int j = 0; j < imageSize; ++j)
        {
            // Normalize to [0,1]
            dataset.images[i][j] = buffer[j] / 255.0f;
        }
    }

    // =======================
    // Load labels
    // =======================
    std::ifstream lbl(labelFile, std::ios::binary);
    if (!lbl)
        throw std::runtime_error("MNIST: cannot open label file");

    magic = readBigEndianInt(lbl);
    if (magic != 2049)
        throw std::runtime_error("MNIST: invalid label file magic");

    int numLabels = readBigEndianInt(lbl);
    if (numLabels != dataset.numImages)
        throw std::runtime_error("MNIST: image/label count mismatch");

    dataset.labels.resize(numLabels);
    lbl.read(reinterpret_cast<char*>(dataset.labels.data()), numLabels);

    return dataset;
}

inline const mnist::MNISTDataset& getMNIST(
    const std::string& imageFile,
    const std::string& labelFile)
{
    static mnist::MNISTDataset dataset = mnist::load(imageFile, labelFile);
    return dataset;
}

inline const mnist::MNISTDataset& getMNISTValidation(
const std::string& imageFile,
const std::string& labelFile)
{
    static mnist::MNISTDataset dataset = mnist::load(imageFile, labelFile);
    return dataset;
}

} // namespace mnist


#endif //NEURALNETWORKLIB_MNIST_LOADER_HPP