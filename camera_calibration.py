import glob
import os

import cv2
import numpy as np


IMAGE_DIR = "/home/alien/YourDirectoryID_p3/Code/log/frames"
IMAGE_PATTERN = "*.png"

# Physical spacing between grid positions (m or mm; only scale matters)
SQUARE_SIZE = 0.03  # 3 cm, change if needed

# We treat the L pattern as living on a 7x6 grid of *possible* positions.
# From the template, the white squares are at the following (col,row) indices:
# (computed once from the clean PNG you sent)
L_PATTERN_INDICES = sorted([
    (0, 0),
    (2, 0),
    (4, 0),
    (6, 0),
    (1, 1),
    (3, 1),
    (5, 1),
    (0, 2),
    (1, 3),
    (0, 4),
    (1, 5),
])
NUM_POINTS = len(L_PATTERN_INDICES)  # 11


def detect_l_squares_centers(gray):
    """
    Detect centers of the white squares forming the L-shape pattern.
    Returns:
        centers: list of (x, y) in image pixels, length should be NUM_POINTS (11)
                 or None if detection fails.
    """
    # 1) Threshold to isolate bright (white) squares
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use high threshold to capture white squares.
    _, th = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)

    # 2) Find contours of bright blobs
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        # Keep roughly square blobs
        if 0.5 < ar < 1.5:
            cx = x + w / 2.0
            cy = y + h / 2.0
            centers.append((cx, cy))

    if len(centers) != NUM_POINTS:
        # Not the expected number of squares -> fail this image
        return None

    # 3) Sort centers into a consistent order: by row (y), then by column (x)
    centers = np.array(centers, dtype=np.float32)

    # Cluster y-values into 6 rows using k-means (rows 0..5)
    ys = centers[:, 1].reshape(-1, 1)
    # criteria: stop after 100 iters or epsilon 0.1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    K_rows = 6
    compactness, labels_row, centers_row = cv2.kmeans(
        ys.astype(np.float32), K_rows, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    # Row index = rank of cluster center by y
    row_centers_sorted_idx = np.argsort(centers_row[:, 0])
    row_map = {cluster_id: rank for rank, cluster_id in enumerate(row_centers_sorted_idx)}

    # Cluster x-values into 7 columns using k-means (cols 0..6)
    xs = centers[:, 0].reshape(-1, 1)
    K_cols = 7
    compactness, labels_col, centers_col = cv2.kmeans(
        xs.astype(np.float32), K_cols, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    col_centers_sorted_idx = np.argsort(centers_col[:, 0])
    col_map = {cluster_id: rank for rank, cluster_id in enumerate(col_centers_sorted_idx)}

    # 4) For each detected square center, assign discrete (c, r) indices
    indexed_points = []
    for i, (cx, cy) in enumerate(centers):
        r_cluster = labels_row[i, 0]
        c_cluster = labels_col[i, 0]
        r = row_map[r_cluster]
        c = col_map[c_cluster]
        indexed_points.append((c, r, cx, cy))

    # 5) Keep only those (c, r) that match the known L pattern, sorted in that order
    indexed_points_dict = {(c, r): (cx, cy) for (c, r, cx, cy) in indexed_points}

    img_points_ordered = []
    for (c, r) in L_PATTERN_INDICES:
        if (c, r) not in indexed_points_dict:
            # pattern mismatch -> fail
            return None
        img_points_ordered.append(indexed_points_dict[(c, r)])

    img_points_ordered = np.array(img_points_ordered, dtype=np.float32).reshape(-1, 1, 2)
    return img_points_ordered


def build_object_points():
    """
    Build 3D object points for the fixed L pattern.
    All points lie on Z=0, with spacing SQUARE_SIZE in both directions.
    """
    objp = []
    for (c, r) in L_PATTERN_INDICES:
        X = c * SQUARE_SIZE
        Y = r * SQUARE_SIZE
        objp.append([X, Y, 0.0])
    objp = np.array(objp, dtype=np.float32)
    return objp


def main():
    objpoints = []  # 3D points in world (same for all images)
    imgpoints = []  # 2D points in each image

    objp_template = build_object_points()

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, IMAGE_PATTERN)))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR} with pattern {IMAGE_PATTERN}")
        return

    print(f"Found {len(image_paths)} images for calibration")

    used_images = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_pts = detect_l_squares_centers(gray)
        if img_pts is None:
            print(f"[!!] L-pattern NOT detected in: {os.path.basename(img_path)}")
            continue

        objpoints.append(objp_template)
        imgpoints.append(img_pts)
        used_images += 1
        print(f"[OK] L-pattern detected in: {os.path.basename(img_path)}")

        # Optional: visualize detections
        # vis = img.copy()
        # for p in img_pts:
            # cv2.circle(vis, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), -1)
        # cv2.imshow("L pattern", vis)
        # cv2.waitKey(0)

    print(f"\nTotal images used: {used_images}")
    if used_images < 5:
        print("Not enough valid views for calibration. "
              "Try more images or ensure the L pattern is clearly visible.")
        return

    # Image size from first image
    img_example = cv2.imread(image_paths[0])
    gray_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)
    image_size = gray_example.shape[::-1]  # (width, height)

    print("\nRunning cv2.calibrateCamera ...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\n=== Calibration Results ===")
    print(f"RMS re-projection error (from calibrateCamera): {ret}")
    print("\nCamera Matrix (K):")
    print(camera_matrix)
    print("\nDistortion coefficients [k1, k2, p1, p2, k3, ...]:")
    print(dist_coeffs.ravel())

    # Explicit mean reprojection error
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        n = len(imgpoints2)
        total_error += error * error
        total_points += n

    mean_error = np.sqrt(total_error / total_points)
    print(f"\nMean reprojection error (pixels): {mean_error}")

    np.savez(
        "camera_calibration_Lpattern.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_size=image_size,
        rms_error=ret,
        mean_error=mean_error,
    )
    print("\nCalibration saved to camera_calibration_Lpattern.npz")


if __name__ == "__main__":
    main()
