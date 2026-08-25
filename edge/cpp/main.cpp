// GaugeVision edge inference demo — CLAUDE.md §4.8.
//
// A minimal C++/ONNX Runtime edge inference implementation is included to
// demonstrate deployment outside the Python runtime. Not a port of the
// whole project: image in, anomaly score + latency out.
//
// Usage:
//   gaugevision_edge_demo <backbone.onnx> <stats.bin> <image_path> [image_path...]

#include "inference.hpp"

#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(
        stderr,
        "usage: %s <backbone.onnx> <stats.bin> <image_path> [image_path...]\n",
        argv[0]);
    return 1;
  }

  const std::string onnx_path = argv[1];
  const std::string stats_path = argv[2];

  try {
    gaugevision::PaDiMEdgeModel model(onnx_path, stats_path);

    for (int i = 3; i < argc; ++i) {
      const std::string image_path = argv[i];
      gaugevision::InferenceResult result = model.predict(image_path);
      std::printf(
          "%-40s score=%.4f  latency_ms=%.2f\n",
          image_path.c_str(),
          result.score,
          result.latency_ms);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
