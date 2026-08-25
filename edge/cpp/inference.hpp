// PaDiM edge inference — CLAUDE.md §4.8 (Phase 6).
//
// Minimal proof that inference works outside the Python runtime: loads the
// ONNX-exported PaDiM backbone via the ONNX Runtime C++ API, runs the same
// Mahalanobis-distance anomaly scoring the Python path uses (against
// statistics exported by export_cpp_stats.py), and reports score + latency
// for one image. Not a port of the whole project — no training, no
// measurement pipeline, no API — just image in, score + latency out.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace gaugevision {

struct InferenceResult {
  float score;
  double latency_ms;  // wall time for preprocess + inference + scoring
};

class PaDiMEdgeModel {
 public:
  // input_size is fixed at 224x224 to match this project's fixed PaDiM
  // input_size default (see gaugevision/anomaly/padim.py) — not encoded in
  // the exported stats file to keep that format minimal.
  PaDiMEdgeModel(const std::string& onnx_path, const std::string& stats_path);

  InferenceResult predict(const std::string& image_path);

 private:
  static constexpr int kInputSize = 224;

  Ort::Env env_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_name_;
  std::string output_name_;

  int n_patches_ = 0;
  int d_ = 0;
  int grid_h_ = 0;
  int grid_w_ = 0;
  float threshold_ = 0.0f;
  std::vector<int32_t> feature_indices_;  // (d,)
  std::vector<float> mean_;               // (n_patches * d)
  std::vector<float> cov_inv_;            // (n_patches * d * d)

  void load_stats(const std::string& stats_path);
  std::vector<float> preprocess(const std::string& image_path) const;
  float mahalanobis_score(const std::vector<float>& embedding, int total_dim) const;
};

}  // namespace gaugevision
