# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:41:13 2025

@author: Adiepoer
"""
import os
import cv2
import numpy as np
import pandas as pd

# Define ROI and scale
roi_x_min = 100
roi_x_max = 698
roi_y_min = 48
roi_y_max = 582
max_frequency = 20  # MHz

def y_to_frequency(y_pixel, roi_y_min, roi_y_max, max_frequency):
    norm_y = (y_pixel - roi_y_min) / (roi_y_max - roi_y_min)
    return (1 - norm_y) * max_frequency

def mask_to_frequency(mask_path, output_csv_path):
    # Load binary mask (already thresholded)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)  # Ensure binary

    # Crop to ROI
    roi_mask = mask[roi_y_min:roi_y_max, roi_x_min:roi_x_max]  # shape: (534, 598)

    time_points = np.linspace(0, 24, roi_x_max - roi_x_min)
    frequency_values = []

    for col in range(roi_mask.shape[1]):
        column = roi_mask[:, col]  # vertical slice
        white_pixels = np.where(column == 1)[0]  # y positions in ROI

        if len(white_pixels) > 0:
            topmost_y = white_pixels[0] + roi_y_min
            freq = y_to_frequency(topmost_y, roi_y_min, roi_y_max, max_frequency)
        else:
            freq = np.nan  # or 0 or -1

        frequency_values.append(freq)

    # Save to CSV
    df = pd.DataFrame({
        'time_hour': time_points,
        'foF2_frequency_MHz': frequency_values
    })
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved frequency data to {output_csv_path}")

mask_path = 'D:/My Project/auto_trace/learning/detected_edges.png'
output_csv = 'D:/My Project/auto_trace/learning/detected_edges.csv'

mask_to_frequency(mask_path, output_csv)