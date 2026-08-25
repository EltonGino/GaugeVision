#include "inference.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gaugevision {

namespace {
constexpr char kMagic[4] = {'G', 'V', 'P', 'D'};
constexpr int kFormatVersion = 1;

// ImageNet normalization constants — must match gaugevision.anomaly.padim.preprocess.
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
}  // namespace

PaDiMEdgeModel::PaDiMEdgeModel(const std::string& onnx_path, const std::string& stats_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "gaugevision-edge"),
      session_(env_, onnx_path.c_str(), Ort::SessionOptions{nullptr}) {
  Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(0, allocator_);
  Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(0, allocator_);
  input_name_ = input_name.get();
  output_name_ = output_name.get();

  load_stats(stats_path);
}

void PaDiMEdgeModel::load_stats(const std::string& stats_path) {
  std::ifstream f(stats_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("could not open stats file: " + stats_path);
  }

  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, kMagic, 4) != 0) {
    throw std::runtime_error("stats file has wrong magic bytes: " + stats_path);
  }

  int32_t version = 0;
  f.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != kFormatVersion) {
    throw std::runtime_error("stats file has unsupported format version");
  }

  int32_t n_patches, d, grid_h, grid_w;
  f.read(reinterpret_cast<char*>(&n_patches), sizeof(n_patches));
  f.read(reinterpret_cast<char*>(&d), sizeof(d));
  f.read(reinterpret_cast<char*>(&grid_h), sizeof(grid_h));
  f.read(reinterpret_cast<char*>(&grid_w), sizeof(grid_w));
  n_patches_ = n_patches;
  d_ = d;
  grid_h_ = grid_h;
  grid_w_ = grid_w;

  f.read(reinterpret_cast<char*>(&threshold_), sizeof(threshold_));

  feature_indices_.resize(d_);
  f.read(reinterpret_cast<char*>(feature_indices_.data()), d_ * sizeof(int32_t));

  mean_.resize(static_cast<size_t>(n_patches_) * d_);
  f.read(reinterpret_cast<char*>(mean_.data()), mean_.size() * sizeof(float));

  cov_inv_.resize(static_cast<size_t>(n_patches_) * d_ * d_);
  f.read(reinterpret_cast<char*>(cov_inv_.data()), cov_inv_.size() * sizeof(float));

  if (!f) {
    throw std::runtime_error("stats file ended unexpectedly (truncated?): " + stats_path);
  }
}

std::vector<float> PaDiMEdgeModel::preprocess(const std::string& image_path) const {
  // Grayscale, matching how this project reads MVTec images throughout
  // (cv2.IMREAD_GRAYSCALE in the Python training/API code), then
  // replicated to 3 channels — mirrors gaugevision.anomaly.padim.preprocess.
  cv::Mat gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (gray.empty()) {
    throw std::runtime_error("could not read image: " + image_path);
  }

  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);

  // NCHW float32, ImageNet-normalized, channel replicated from the single
  // grayscale channel (R=G=B), matching the Python path's np.stack([img]*3).
  std::vector<float> tensor(static_cast<size_t>(3) * kInputSize * kInputSize);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < kInputSize; ++y) {
      for (int x = 0; x < kInputSize; ++x) {
        float pixel = resized.at<uint8_t>(y, x) / 255.0f;
        float normalized = (pixel - kMean[c]) / kStd[c];
        tensor[static_cast<size_t>(c) * kInputSize * kInputSize + y * kInputSize + x] = normalized;
      }
    }
  }
  return tensor;
}

float PaDiMEdgeModel::mahalanobis_score(const std::vector<float>& embedding, int total_dim) const {
  // embedding is (n_patches_, total_dim), row-major. For each patch, select
  // the d_ feature_indices_, compute (diff)^T * cov_inv * diff, take sqrt;
  // the image-level score is the max over patches — mirrors
  // gaugevision.anomaly.padim.mahalanobis_score exactly.
  float max_dist = 0.0f;
  std::vector<float> diff(d_);

  for (int p = 0; p < n_patches_; ++p) {
    const float* patch_full = &embedding[static_cast<size_t>(p) * total_dim];
    const float* patch_mean = &mean_[static_cast<size_t>(p) * d_];
    for (int i = 0; i < d_; ++i) {
      diff[i] = patch_full[feature_indices_[i]] - patch_mean[i];
    }

    const float* cov_inv_p = &cov_inv_[static_cast<size_t>(p) * d_ * d_];
    float dist_sq = 0.0f;
    for (int i = 0; i < d_; ++i) {
      float row_sum = 0.0f;
      const float* cov_row = &cov_inv_p[static_cast<size_t>(i) * d_];
      for (int j = 0; j < d_; ++j) {
        row_sum += cov_row[j] * diff[j];
      }
      dist_sq += diff[i] * row_sum;
    }
    float dist = std::sqrt(std::max(dist_sq, 0.0f));
    max_dist = std::max(max_dist, dist);
  }
  return max_dist;
}

InferenceResult PaDiMEdgeModel::predict(const std::string& image_path) {
  auto start = std::chrono::steady_clock::now();

  std::vector<float> input_tensor = preprocess(image_path);

  std::array<int64_t, 4> input_shape = {1, 3, kInputSize, kInputSize};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_value = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};

  auto output_tensors =
      session_.Run(Ort::RunOptions{nullptr}, input_names, &input_value, 1, output_names, 1);

  const Ort::Value& output = output_tensors.front();
  Ort::TensorTypeAndShapeInfo shape_info = output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = shape_info.GetShape();  // (1, C, H, W)
  int64_t channels = shape[1];
  int64_t out_h = shape[2];
  int64_t out_w = shape[3];

  if (out_h != grid_h_ || out_w != grid_w_) {
    throw std::runtime_error("backbone output grid size doesn't match fitted stats");
  }

  const float* output_data = output.GetTensorData<float>();

  // Reorder from NCHW to (n_patches, total_dim) row-major, matching the
  // Python path's embedding.reshape(1,c,h*w).permute(0,2,1) layout.
  int total_dim = static_cast<int>(channels);
  std::vector<float> embedding(static_cast<size_t>(n_patches_) * total_dim);
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t p = 0; p < n_patches_; ++p) {
      embedding[static_cast<size_t>(p) * total_dim + c] = output_data[c * out_h * out_w + p];
    }
  }

  float score = mahalanobis_score(embedding, total_dim);

  auto end = std::chrono::steady_clock::now();
  double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

  return InferenceResult{score, latency_ms};
}

}  // namespace gaugevision
