import math
import time
import sys
import cv2
import easyocr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from itertools import combinations
from math import ceil
from PIL import Image
from skimage import io, filters
from skimage.color import rgb2lab, lab2rgb, rgb2gray, lab2lch, lch2lab
from skimage.io import imread
from skimage.morphology import skeletonize
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from textwrap import wrap
from tkinter import N
from scipy.ndimage.morphology import distance_transform_edt

import colour



# Helper functions
def lch2rgb(lch_arr):
    """
    Converts LCH color space to RGB color space.

    Args:
        lch_arr (np.ndarray): Input array in LCH color space.

    Returns:
        np.ndarray: Output array in RGB color space.
    """

    return lab2rgb(lch2lab(lch_arr))


def rgb2lch(rgb_arr):
    """
    Converts RGB color space to LCH color space.

    Args:
        rgb_arr (np.ndarray): Input array in RGB color space.

    Returns:
        np.ndarray: Output array in LCH color space.
    """
    return lab2lch(rgb2lab(rgb_arr))


# Mask generation functions
def mask_gen(mask_size, bright_area, Hdark_i, Hbright_i):
    """
    Generates a mask with a specified size and brightness.

    Args:
        mask_size (int): Size of the mask.
        bright_area (int): Area of the mask that should be bright.
        Hdark_i (int): Dark intensity.
        Hbright_i (int): Bright intensity.

    Returns:
        np.ndarray: Generated mask.
    """
    mask = np.full((mask_size, mask_size), Hdark_i)
    bright_center = mask_size // 2
    if bright_area > 1:
        assert bright_area <= (mask_size - 2), "make sure right_area<=(mask_size-2)"
        if bright_area % 2 == 1:  # * odd
            upper = int(bright_center - (bright_area - np.ceil(bright_area / 2)))
            lower = int(bright_center + (bright_area - np.ceil(bright_area / 2)) + 1)
        if bright_area % 2 == 0:
            upper = int(bright_center - bright_area / 2)
            lower = int(bright_center + bright_area / 2)
        mask[upper:lower, upper:lower] = Hbright_i
    else:
        mask[mask_size // 2, mask_size // 2] = Hbright_i

    return mask


def edge_mask_gen(mask_size, bright_area, Hdark_i, Hbright_i):
    """
    Generates an edge mask with a specified size and brightness.

    Args:
        mask_size (int): Size of the mask.
        bright_area (int): Area of the mask that should be bright.
        Hdark_i (int): Dark intensity.
        Hbright_i (int): Bright intensity.

    Returns:
        np.ndarray: Generated edge mask.
    """
    mask = np.full((mask_size, mask_size), Hdark_i)
    mask[[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4], [0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4]] = Hbright_i
    return mask

def grid_gen(img, mask_size, bright_size, pattern=2):
    """
    Generates a grid.

    Args:
        img (np.ndarray): Input image.
        mask_size (int): Size of the mask.
        bright_size (int): Size of the bright area.
        pattern (int, optional): Pattern for the grid. Defaults to 2.

    Returns:
        np.ndarray: Generated grid.
    """
    Hdark_i = 0
    Hbright_i = 1
    if pattern == 1:
        mask = np.array([[Hdark_i, Hbright_i], [Hbright_i, Hdark_i]])
    # NOTE: odd number masking
    if pattern == 2:

        mask = mask_gen(mask_size, bright_size, Hdark_i, Hbright_i)
    # * edge generator
    if pattern == 3:
        mask = edge_mask_gen(mask_size, bright_size, Hdark_i, Hbright_i)
        plt.imshow(mask, cmap="gray")
        plt.show()
    row_dups = int((np.ceil(img.shape[0] / mask_size)))
    cols_dups = int(np.ceil(img.shape[1] / mask_size))
    mask = np.tile(mask, (row_dups, cols_dups))

    mask = mask[: img.shape[0], : img.shape[1]]
    return mask

def dest_image(img, mask, bg_illum):
    """
    Modifies the image based on the mask and background illumination.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Mask.
        bg_illum (float): Background illumination.

    Returns:
        np.ndarray: Output image.
    """
    output = np.copy(img)
    row, col = np.where((mask == 0))

    output[row, col] = bg_illum

    return output

def dest_image2(img, mask, fg_area, bg_avg_rgb, similar_contrast):
    """
    Modifies the image based on the mask, foreground area, 
    average background color and similar contrast.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Mask.
        fg_area (np.ndarray): Foreground area.
        bg_avg_rgb (list): Average background color in RGB.
        similar_contrast (float): Similar contrast.

    Returns:
        np.ndarray: Output image.
    """
    output = np.copy(img)
    row, col = np.where((mask == 0))

    # * layers superimposition
    output[row, col] = bg_avg_rgb
    if np.max(bg_avg_rgb) >= 1.0:
        bg_illum = rgb2lab(bg_avg_rgb / 255.0)[0]
    else:
        bg_illum = rgb2lab(bg_avg_rgb)[0]

    # print(f"bg_illum: {bg_illum}")

    if np.max(output) >= 1.0:
        lab_output = rgb2lab(output / 255.0)
    else:
        lab_output = rgb2lab(output)
    row, col = np.where(fg_area & (mask == 1))

    assert bg_illum >= np.max(
        lab_output[fg_area][:, 0]
    ), "background illuminance must be brighter than the background's"
    # print(f"before contrast improvements: {lab_output[595:600,440:460,0]}")

    # contrast_diff = abs(bg_illum - lab_output[fg_area][:, 0])
    # fg_illum = lab_output[fg_area][:, 0] + contrast_diff * similar_contrast / 10

    contrast_diff = abs(bg_illum)
    fg_illum = contrast_diff / 100 * similar_contrast
    lab_output[row, col, 0] = fg_illum
    # print(f"after contrast improvements: {lab_output[595:600,440:460,0]}")

    return lab_output


# Image processing functions
def derive_edge(bg_image, image):
    """
    Derives edges from an image.

    Args:
        bg_image (np.ndarray): Background image.
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Derived edges.
    """
    assert len(image.shape) == 3, "the image must have three dimensions"
    assert np.max(image) > 100, "the image must be RGB format"
    tgt_img = np.mean(image, axis=2)
    thresh = filters.threshold_sauvola(tgt_img)
    edges = tgt_img < thresh
    filtered_edges = bg_image * edges
    return filtered_edges

def segment_bg(image):
    """
    Segments background and foreground of the image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        tuple: Background and foreground masks.
    """
    assert len(image.shape) == 3, "the image must have three dimensions"
    assert np.max(image) > 100, "the image must be RGB format"
    # tgt_img = rgb2lab(image)[..., 0]
    tgt_img = np.mean(image, axis=2)
    threshold = filters.threshold_li(tgt_img)
    fg = tgt_img < threshold
    bg = tgt_img >= threshold
    # plt.imshow(tgt_img < threshold, cmap='gray')
    return bg, fg

def dest_edges(img, mask, edges, bg_illum):
    """
    Modifies the image based on the mask, edges and background illumination.

    Args:
        img (np.ndarray): Input image.
        mask (np.ndarray): Mask.
        edges (np.ndarray): Edges.
        bg_illum (float): Background illumination.

    Returns:
        np.ndarray: Output image.
    """
    output = np.copy(img)
    row, col = np.where((mask == 0) * edges)

    output[row, col] = bg_illum
    return output


# Image contour functions
def get_contours(fg_area):
    """
    Modifies the input image based on the given contour information and a specified radius.
    
    Args:
        image (np.array): A 2D numpy array representing the input image.
        contour (list): A list of 2D points representing the contour of an object.
        radius (int): The radius within which to modify the image. 
    
    Returns:
        output (np.array): A 2D numpy array representing the modified image.
    """
    return cv2.findContours(fg_area.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

def dest_contours(fg_area, contours, interval):
    """
    Function to draw contours with a specific interval on an image.

    Args:
        fg_area (np.array): An array representing the foreground area of the image.
        contours (list): A list of contour arrays.
        interval (int): The interval at which to draw contours.

    Returns:
        tuple: Two numpy arrays. The first contains the image with contours drawn at the specified interval.
               The second contains the image with all contours drawn.
    """
    contours_np = np.vstack(contours)
    length = contours_np.shape[0]
    a = np.zeros(fg_area.shape)
    b = np.zeros(fg_area.shape).astype(np.uint8)

    cv2.drawContours(a, contours_np[np.arange(0, length, interval), ::], -1, 1, 1)
    cv2.drawContours(b, contours_np, -1, 1, 1)
    return a, b


def text_ocr(image):
    """
    Function to perform OCR on an image.

    Args:
        image (np.array): An array representing the image.

    Returns:
        tuple: Two lists. The first contains bounding box coordinates for each detected text.
               The second contains the detected texts.
    """
    reader = easyocr.Reader(["en"])
    image = cv2.blur(image, (3, 3))
    result = reader.readtext(image, text_threshold=0.5, low_text=0.3)
    # * bbox:[tl, tr, br, bl]
    bbox = [i[0] for i in result]
    texts = [i[1] for i in result]
    return bbox, texts

def text_mask_gen(img, cell_size, text_width):
    """
    Function to generate a text mask for an image.

    Args:
        img (np.array): An array representing the image.
        cell_size (int): Size of each cell in the mask.
        text_width (int): The width of the text.

    Returns:
        np.array: A mask for the image.
    """
    Hdark_i = 0
    Hbright_i = 1

    mask_area = 3
    mask = np.array([Hdark_i] * (mask_area * mask_area)).reshape((mask_area, mask_area))
    if text_width <= 2:
        mask[[0, 2], [0, 2]] = Hbright_i
    else:
        mask[[1], [1]] = Hbright_i
    # mask = np.array([[Hbright_i, Hdark_i,],[Hdark_i, Hbright_i]])

    mask = np.tile(mask, (img.shape[0] // cell_size, img.shape[1] // cell_size))
    a = np.repeat(mask, cell_size, axis=0)
    b = np.repeat(a, cell_size, axis=1)
    mask = b[: img.shape[0], : img.shape[1]]
    return mask

def dest_text(img, mask, bg_color, fg_color, fg_area):
    """
    Function to modify the color of the text and background in an image.

    Args:
        img (np.array): An array representing the image.
        mask (np.array): A mask for the image.
        bg_color (list): The RGB color to apply to the background.
        fg_color (list): The RGB color to apply to the text.
        fg_area (np.array): An array representing the foreground area of the image.

    Returns:
        np.array: The image with modified text and background colors.
    """
    bg_illum = bg_color[0]
    fg_illum = fg_color[0]
    TEXT_PLUS = 5
    text_illum = (bg_illum + fg_illum) / 2 + TEXT_PLUS
    text_color = np.array([text_illum, fg_color[1], fg_color[2]])
    output = np.copy(img)
    output[np.where(mask == 0)] = bg_color
    output[np.where(fg_area * (mask == 1))] = text_color
    return output

def dest_text2(img, mask, bg_color):
    """
    Function to modify the color of the background in an image.

    Args:
        img (np.array): An array representing the image.
        mask (np.array): A mask for the image.
        bg_color (list): The RGB color to apply to the background.
        fg_color (list): The RGB color to apply to the text.

    Returns:
        np.array: The image with the modified background color.
    """
    output = np.copy(img)
    output[np.where(mask == 0)] = bg_color
    return output

def stroke_width(text_fg):
    """
    Function to calculate the stroke width of the text in an image.

    Args:
        text_fg (np.array): An array representing the foreground text of the image.

    Returns:
        tuple: The average stroke width of the text and a skeletonized representation of the text.
    """
    padded_img = np.pad(text_fg, 2)
    distances = distance_transform_edt(padded_img)
    skeleton = skeletonize(padded_img)
    center_pixel_distances = distances[skeleton]
    # * get the stroke width, twice the average of center pixel distances b/c supposedly,the width is symmetry
    stroke_width = np.mean(center_pixel_distances) * 2
    return stroke_width, skeleton

def dest_contour(image, contour, radius):
    """
    Modifies the input image based on the given contour information and a specified radius.
    
    Args:
        image (np.array): A 2D numpy array representing the input image.
        contour (list): A list of 2D points representing the contour of an object.
        radius (int): The radius within which to modify the image. 
    
    Returns:
        output (np.array): A 2D numpy array representing the modified image.
    """
    output = np.copy(image)
    contours_np = np.concatenate([i.reshape(-1, 2) for i in contour])
    tgt_list = []
    for contour_pt in contours_np:
        row = contour_pt[1]
        col = contour_pt[0]
        if output[row, col] == 0:
            continue
        # * keep only one point
        min_row = row - radius
        max_row = row + radius + 1
        min_col = col - radius
        max_col = col + radius + 1
        if min_row < 0 or max_row > image.shape[0] or min_col < 0 or max_col > image.shape[1]:
            print("out of limitation")
            continue

        mask_size = 2 * radius + 1
        if mask_size == 1:
            return image

        temp = np.copy(image[min_row:max_row, min_col:max_col])
        if mask_size != 3:
            temp[:mask_size:2, 1:mask_size:2] = 0
            temp[1:mask_size:2, :] = 0

            mask_arr = np.zeros((mask_size, mask_size))
            mask_arr[radius - 2 : radius + 3, radius - 2 : radius + 3] = 1

        else:
            temp[:mask_size:2, 1:mask_size:2] = 0
            temp[1:mask_size:2, :mask_size:2] = 0
            mask_arr = np.ones((mask_size, mask_size))

        temp = temp * mask_arr
        r, c = np.where(temp == 1)
        offsets = np.vstack((r - radius, c - radius)).T
        bright_index_arr = offsets + np.array([row, col])

        output[min_row:max_row, min_col:max_col][:mask_size:2, 1:mask_size:2] = 0
        # output[min_row:max_row, min_col:max_col][1:mask_size:2, 0:mask_size:2] = 0
        output[min_row:max_row, min_col:max_col][1:mask_size:2, :] = 0
        output[min_row:max_row, min_col:max_col] = output[min_row:max_row, min_col:max_col] * mask_arr

        tgt_list += bright_index_arr.tolist()
    # print(f"mask size: {mask_size}")
    for arr in tgt_list:

        row = arr[0]
        col = arr[1]

        output[row, col] = 1
    return output

# OCR operations
def run_ocr_in_chart(image, output, bg_avg_lab, bboxes):
    """
    Performs Optical Character Recognition (OCR) on a given image and modifies the output image accordingly. 
    
    Args:
        image (np.array): A 3D numpy array representing the input image.
        output (np.array): A 3D numpy array representing the output image.
        bg_avg_lab (np.array): A 3D numpy array representing the average color of the background in LAB color space.
        bboxes (list): A list of bounding boxes for detected text regions in the image.
        
    Returns:
        output (np.array): The modified output image.
        bboxes (list): A list of bounding boxes for detected text regions in the image.
    """
    if not bboxes:
        bboxes, texts = text_ocr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for bbox in bboxes:
        tl, tr, br, bl = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        text_region = image[tl[1] : br[1], tl[0] : br[0]]
        text_thresh = filters.threshold_li(rgb2gray(text_region))
        text_fg = rgb2gray(text_region) < text_thresh
        _, fg_area = segment_bg(text_region)
        fg_avg_lab = np.mean(rgb2lab(text_region[fg_area]), axis=0)

        radius = int(np.min(text_region.shape[:2]) / 10)
        if radius >= 5:
            radius = 4
        text_fg_pad = np.pad(text_fg, radius)
        fg_area_pad = np.pad(fg_area, radius)
        text_width, skeleton = stroke_width(fg_area_pad)

        skeleton = skeleton[2:-2, 2:-2]

        row, col = np.where(skeleton == True)
        skeleton_cont = np.vstack((col, row)).T
        result = dest_contour(text_fg_pad, skeleton_cont, radius)[radius:-radius, radius:-radius]
        # output2 = test_dest_edges(image, result, bg_avg_rgb)
        text_output = dest_text2(rgb2lab(text_region), result, bg_avg_lab, fg_avg_lab)

        output[tl[1] : br[1], tl[0] : br[0]] = (lab2rgb(text_output) * 255).astype(np.uint8)

    return output, bboxes

