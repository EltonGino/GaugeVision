from gaugevision.calibration.lens_calibration import (
    calibrate_camera,
    generate_synthetic_checkerboard_views,
    undistort,
)


def test_synthetic_checkerboard_calibration_converges():
    pattern_size = (9, 6)
    views = generate_synthetic_checkerboard_views(
        pattern_size=pattern_size, n_views=12, seed=0
    )
    calibration = calibrate_camera(views, pattern_size=pattern_size)

    assert calibration.camera_matrix.shape == (3, 3)
    assert calibration.dist_coeffs.size >= 4
    # Homography-warped views of a flat board are a weaker calibration set
    # than real multi-angle photos of a physical checkerboard (planar
    # homographies under-constrain the pinhole model), so this is a loose
    # "did calibration converge at all" bound, not an accuracy claim — see
    # CLAUDE.md §4.1a: capability demonstration, not a real-camera calibration.
    assert calibration.rms_reprojection_error < 15.0


def test_undistort_preserves_image_shape():
    pattern_size = (9, 6)
    views = generate_synthetic_checkerboard_views(pattern_size=pattern_size, n_views=12, seed=1)
    calibration = calibrate_camera(views, pattern_size=pattern_size)

    undistorted = undistort(views[0], calibration)
    assert undistorted.shape == views[0].shape
