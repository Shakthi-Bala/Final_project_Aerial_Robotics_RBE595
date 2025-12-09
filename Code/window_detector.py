import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import cv2
from scipy import ndimage
from control import pid
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2


def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum: top-left has smallest sum, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diff: top-right has smallest diff, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def separate_overlapping_windows(mask):
    """
    Separate overlapping windows using watershed algorithm
    """
    # 1. Ensure binary mask
    binary = (mask > 0.5).astype(np.uint8) * 255
    
    # 2. Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Sure background area (dilate)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 4. Sure foreground area (distance transform + threshold)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 5. Unknown region (background - foreground)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 6. Label markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Add 1 so background is not 0
    markers[unknown == 255] = 0  # Mark unknown regions as 0
    
    # 7. Apply watershed
    # Need 3-channel image for watershed
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(binary_3ch, markers)
    
    # 8. Extract individual windows
    individual_windows = []
    for label in range(2, markers.max() + 1):  # Skip 0 (unknown) and 1 (background)
        window_mask = np.zeros_like(mask)
        window_mask[markers == label] = 255
        individual_windows.append(window_mask)
    
    return individual_windows


def separate_by_erosion(mask, erosion_size=5):
    """
    Separate windows by eroding to break connections
    """
    # 1. Binary mask
    binary = (mask > 0.5).astype(np.uint8) * 255
    
    # 2. Erode to separate touching windows
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=2)
    
    # 3. Find connected components in eroded image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    # 4. For each component, dilate back to original size
    individual_windows = []
    for label in range(1, num_labels):  # Skip background (0)
        # Create mask for this component
        component_mask = (labels == label).astype(np.uint8) * 255
        
        # Dilate back (approximately)
        dilated = cv2.dilate(component_mask, kernel, iterations=2)
        
        # Mask with original to get actual window pixels
        window_mask = cv2.bitwise_and(binary, dilated)
        
        individual_windows.append(window_mask)
    
    return individual_windows


def robust_window_separation(mask):
    """
    Robust separation with fallback
    """
    # Try watershed first
    try:
        windows = separate_overlapping_windows(mask)
        if len(windows) > 0:
            print("Separated using watershed")
            return windows
        
    except:
        pass
    
    # Fallback to erosion
    try:
        windows = separate_by_erosion(mask)
        if len(windows) > 0:
            return windows
    except:
        pass
    
    # Last resort: return original mask
    return [mask]

def find_closest_window(individual_windows, img_shape):
    """
    Among separated windows, find the closest (largest area)
    """
    max_area = 0
    closest_window = None
    
    for window_mask in individual_windows:
        contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
            
        contour = contours[0]
        area = cv2.contourArea(contour)
        
        if area > max_area:
            max_area = area
            closest_window = window_mask
    
    return closest_window


val_transform = A.Compose([
    A.Resize(height=640, width=640),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2() # Converts (H, W, C) -> (C, H, W) and handles tensor conversion
])

def preprocess(image_np):
    """
    Applies the *exact* same validation transforms as the training pipeline.
    Expects a numpy array loaded with cv2.imread().
    """
    
    # 1. Correct BGR to RGB (since cv2 loads as BGR)
    #    Your training pipeline used PIL, which loads as RGB.
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # 2. Apply all validation transforms
    #    Albumentations takes and returns a dictionary
    transformed = val_transform(image=image_rgb)
    image_tensor = transformed['image']
    
    # 3. Add batch dimension (N, C, H, W) and send to device
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    return image_tensor

def extract_window_featues(model, rgb_image, test_mask=None):
    # Non maximum suppression can be applied here if needed 

    # TODO check if canny edge detection is needed before finding contours

    if test_mask is None:
        with torch.no_grad():  
            print("running model inference")
            mask_tensor = model(preprocess(rgb_image))
        
        prob_tensor = torch.sigmoid(mask_tensor)
        # 1. Convert mask tensor to binary mask
        prob_np= torch.squeeze(prob_tensor).cpu().numpy()

        orig_h, orig_w = rgb_image.shape[:2]
        prob_np_resized = cv2.resize(prob_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        mask_squeeze = (prob_np_resized > 0.5).astype(np.uint8) * 255

        output_path = "model_output_mask.png"
        cv2.imwrite(output_path, mask_squeeze)
    
    else:
        mask_squeeze = test_mask
    # 2. Separate overlapping windows
    individual_windows = robust_window_separation(mask_squeeze)

    output_path = "separated_windows.png"
    cv2.imwrite(output_path, np.hstack(individual_windows))
    
    # 3. Find closest window (largest area = closest)
    closest_mask = find_closest_window(individual_windows, rgb_image.shape)
    
    if closest_mask is None:
        return None, None, None, None
    
    # 4. Extract features from closest window only
    contours, _ = cv2.findContours(closest_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    
    # 5. Get centroid, area, corners
    M = cv2.moments(largest)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = cv2.contourArea(largest)
    
    rect = cv2.minAreaRect(largest)
    corners = cv2.boxPoints(rect)


    # Create a new blank mask and draw the largest contour on it
    new_mask = np.zeros_like(mask_squeeze)

    cv2.polylines(new_mask, [np.int32(corners)], isClosed=True, color=(255), thickness=2)
    cv2.circle(new_mask, (cx, cy), radius=1, color=(255), thickness=-1)

    return new_mask, (cx, cy), area, corners