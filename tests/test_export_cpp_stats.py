import struct

import numpy as np

from gaugevision.anomaly.padim import PaDiMStats
from gaugevision.deployment.export_cpp_stats import (
    FORMAT_VERSION,
    MAGIC,
    export_stats_for_cpp,
)


def test_export_cpp_stats_binary_layout(tmp_path, monkeypatch):
    n_patches, d = 4, 3
    stats = PaDiMStats(
        mean=np.arange(n_patches * d, dtype=np.float64).reshape(n_patches, d),
        cov_inv=np.stack([np.eye(d)] * n_patches).astype(np.float64),
        feature_indices=np.array([1, 3, 5]),
        grid_size=(2, 2),
        input_size=(224, 224),
    )

    def fake_load_stats(path):
        return stats, 1.2345

    monkeypatch.setattr(
        "gaugevision.deployment.export_cpp_stats.load_stats", fake_load_stats
    )

    output_path = tmp_path / "stats.bin"
    result = export_stats_for_cpp(model_path="unused.npz", output_path=output_path)

    assert result["n_patches"] == n_patches
    assert result["d"] == d
    assert result["grid_size"] == (2, 2)

    with open(output_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        (version,) = struct.unpack("<i", f.read(4))
        assert version == FORMAT_VERSION
        header = struct.unpack("<iiii", f.read(16))
        assert header == (n_patches, d, 2, 2)
        (threshold,) = struct.unpack("<f", f.read(4))
        assert threshold == np.float32(1.2345)

        feature_indices = np.frombuffer(f.read(d * 4), dtype=np.int32)
        assert list(feature_indices) == [1, 3, 5]

        mean = np.frombuffer(f.read(n_patches * d * 4), dtype=np.float32).reshape(n_patches, d)
        assert np.allclose(mean, stats.mean.astype(np.float32))

        cov_inv = np.frombuffer(f.read(n_patches * d * d * 4), dtype=np.float32).reshape(
            n_patches, d, d
        )
        assert np.allclose(cov_inv, stats.cov_inv.astype(np.float32))

        assert f.read() == b""  # no trailing bytes
