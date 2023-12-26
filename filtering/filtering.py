"""
Module for image filtering operations.

This module contains functions for applying convolutional kernels on images.
"""
import numpy as np


def apply_filter(image: np.array, kernel: np.array) -> np.array:
    """
    Apply convolutional kernel on image.

    Args:
        image (array): Input image (RGB or gray).
        kernel (array): Convolutional kernel.

    Returns:
        array: Image with applied kernel.
    """
    # A given image has to have either 2 (grayscale) or 3 (RGB) dimensions
    assert image.ndim in [2, 3]
    # A given filter has to be 2 dimensional and square
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

    if kernel.shape[0] % 2 == 0:
        kernel = np.pad(kernel, ((0, 1), (0, 1)), mode='constant', constant_values=0)


    if image.ndim == 2:
        return gray_kernel(image, kernel)

    return RGB_kernel(image, kernel)


def gray_kernel(image: np.array, kernel: np.array) -> np.array:
    """
    Apply convolutional kernel on gray image.

    Args:
        image (array): Input image (gray).
        kernel (array): Convolutional kernel.

    Returns:
        array: Image with applied kernel.
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    # padding
    padding = int(kernel.shape[0] / 2)
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # proccessing
    proccessed_image = np.empty((image_height, image_width), dtype=np.uint8)
    for i in range(0, image_height):
        for j in range(0, image_width):
            new_pixel = 0
            for k in range(0, kernel_height):
                for n in range(0, kernel_width):
                    new_pixel += padded_image[i + k][j + n] * kernel[k][n]
            if new_pixel > 255:
                proccessed_image[i][j] = 255
            elif new_pixel < 0:
                proccessed_image[i][j] = 0
            else:
                proccessed_image[i][j] = new_pixel
    return proccessed_image


def RGB_kernel(image: np.array, kernel: np.array) -> np.array:
    """
     Apply convolutional kernel on RGB image.

     Args:
         image (array): Input image (RGB).
         kernel (array): Convolutional kernel.

     Returns:
         array: Image with applied kernel.
     """
    image_height, image_width, image_depth = image.shape
    # padding
    padding = int(kernel.shape[0] / 2)
    padded_image = np.empty((image_height + padding * 2, image_width + padding * 2, image_depth), dtype=np.uint8)
    for i in range(0, image_height + padding * 2):
        for j in range(0, image_width + padding * 2):
            for k in range(0, image_depth):
                padded_image[i][j][k] = 0 if (i in range(0, padding)) or (j in range(0, padding)) or (
                            i in range(image_height + padding, image_height + padding * 2)) or (
                                                         j in range(image_width + padding,
                                                                    image_width + padding * 2)) else \
                image[i - padding][j - padding][k]
    # proccessing
    proccessed_image = np.empty((image_height, image_width, image_depth), dtype=np.uint8)
    for i in range(0, image_height):
        for j in range(0, image_width):
            for d in range(0, image_depth):
                new_pixel = 0
                for k in range(0, kernel.shape[0]):
                    for n in range(0, kernel.shape[0]):
                        new_pixel += padded_image[i + k][j + n][d] * kernel[k][n]
                if new_pixel > 255:
                    proccessed_image[i][j][d] = 255
                elif new_pixel < 0:
                    proccessed_image[i][j][d] = 0
                else:
                    proccessed_image[i][j][d] = new_pixel
    return proccessed_image
