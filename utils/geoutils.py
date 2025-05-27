import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.mask
from PIL import Image
import cv2
import torch

def check_quality(filename, threshold=200000):
    """
    Detect if an image is blurry using the Laplacian variance method.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): Variance threshold below which the image is considered blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
        float: The variance of the Laplacian.
    """
    pil_image = Image.open(filename)

    new_size = (500, 500)
    pil_image = pil_image.resize(new_size)
    image = np.array(pil_image)
    
    width, height = pil_image.size

    # Define box size
    box_width, box_height = 250, 250

    # Calculate top-left corner of the centered box
    x = (width - box_width) // 2
    y = (height - box_height) // 2

    # Extract ROI
    image = image[y:y+box_height, x:x+box_width]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 50, 200, apertureSize=5) 

    # Determine if the image is blurry
    return edges.sum() > threshold
    


def crop_tile(shape, scale, in_file, out_file):  
    """
    Crops the aerial image using a scaled minimum rotated rectangle of the input shape.

    Args:
        shape (GeoDataFrame): Geometry to crop around.
        scale (float): Scale factor for expanding the shape.
        in_file (str): Input raster file path.
        out_file (str): Output raster file path.

    Returns:
        None
    """
    
    shape.geometry = shape.geometry.apply(lambda x: x.minimum_rotated_rectangle)
    shape.geometry = shape.geometry.scale(scale, scale)

    with rio.open(in_file) as src:
        shape = shape.to_crs(src.crs)
        out_image, out_transform = rasterio.mask.mask(
            src, [shape.iloc[0].geometry], crop=True
        )
        out_meta = src.meta
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": 0
            }
        )
    with rio.open(out_file, "w", **out_meta) as dest:
        dest.write(out_image)