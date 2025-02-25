import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import cv2
import numpy as np

from roboreg.util import (
    mask_dilate_with_kernel,
    mask_distance_transform,
    mask_erode_with_kernel,
    mask_exponential_decay,
    mask_extract_boundary,
    mask_extract_extended_boundary,
    overlay_mask,
)


def test_dilate_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    dilated_mask = mask_dilate_with_kernel(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("dilated_mask", dilated_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_distance_transform() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )

    # show distance map
    distance_map = mask_distance_transform(mask)
    distance_map = (distance_map / distance_map.max() * 255.0).astype(
        np.uint8
    )  # normalize for visualization
    cv2.imshow("mask", mask)
    cv2.imshow("distance_map", distance_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show inverse distance map
    inverse_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    inverse_distance_map = mask_distance_transform(inverse_mask)
    inverse_distance_map = (
        inverse_distance_map / inverse_distance_map.max() * 255.0
    ).astype(
        np.uint8
    )  # normalize for visualization
    cv2.imshow("inverse_mask", inverse_mask)
    cv2.imshow("inverse_distance_map", inverse_distance_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_erode_with_kernel() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    eroded_mask = mask_erode_with_kernel(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("eroded_mask", eroded_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_exponential_decay() -> None:
    idx = 1
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    exponential_decay = mask_exponential_decay(mask)
    cv2.imshow("mask", mask)
    cv2.imshow("exponential_decay", exponential_decay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_extract_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/assets/lbr_med7/zed2i/left_image_{idx}.png")
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    boundary_mask = mask_extract_boundary(mask)
    overlay = overlay_mask(img, boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imshow("mask", mask)
    cv2.imshow("boundary_mask", boundary_mask)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_extract_extended_boundary() -> None:
    idx = 1
    img = cv2.imread(f"test/assets/lbr_med7/zed2i/left_image_{idx}.png")
    mask = cv2.imread(
        f"test/assets/lbr_med7/zed2i/mask_sam2_left_image_{idx}.png",
        cv2.IMREAD_GRAYSCALE,
    )
    extended_boundary_mask = mask_extract_extended_boundary(
        mask, dilation_kernel=np.ones([2, 2]), erosion_kernel=np.ones([10, 10])
    )
    overlay = overlay_mask(img, extended_boundary_mask, mode="b", alpha=1.0, scale=1.0)
    cv2.imshow("mask", mask)
    cv2.imshow("extended_boundary_mask", extended_boundary_mask)
    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_dilate_with_kernel()
    test_distance_transform()
    test_erode_with_kernel()
    test_exponential_decay()
    test_extract_boundary()
    test_extract_extended_boundary()
