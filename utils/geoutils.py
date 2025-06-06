import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.mask
from PIL import Image
from tqdm import tqdm
import imagehash
import torch
import cv2

def remove_duplicates(
    data: pd.DataFrame,
    hash_size: int = 8,
    image_dir: str = "data/images/",
    similarity = 95
):
    hashes = []
    threshold = 1 - similarity/100
    diff_limit = int(threshold*(hash_size**2))
    duplicates = []

    duplicates = []
    for index in tqdm(range(len(data)), total=len(data)):
        item = data.iloc[index]
        image = Image.open(item.filepath)
        temp_hash = imagehash.average_hash(image, hash_size).hash

        found = False
        for hash_ in hashes:
            if np.count_nonzero(temp_hash != hash_) <= diff_limit:
                duplicates.append(item["UID"])
                found = True
                break
            
        if not found:
            hashes.append(temp_hash)
            
    #data = data[~data.uid.isin(duplicates)]
    data["duplicate"] = False
    data.loc[data.UID.isin(duplicates), "duplicate"] = True
    return data


def inspect_quality(filename: str, threshold: float = 200000):
    """Assesses the sharpness of an image by measuring edge strength in a central region.

    This function opens an image file, resizes it to 500x500 pixels, extracts a 
    250x250 pixel region centered in the image, converts it to grayscale, and then 
    applies Canny edge detection. It returns `True` if the sum of detected edge 
    intensities exceeds a specified threshold, indicating that the image is likely 
    sharp; otherwise, it returns `False`.

    Args:
        filename (str): Path to the image file.
        threshold (int, optional): Edge intensity threshold to classify an image as 
            sharp or blurry. Default is 200000.

    Returns:
        bool: True if the image is considered sharp, False if considered blurry.
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
    


def crop_tile(shape: gpd.GeoDataFrame, scale: float, in_file: str, out_file: str):  
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

    from shapely.geometry import Point
    from math import sqrt
    
    def to_square(polygon):
        minx, miny, maxx, maxy = polygon.bounds
        centroid = [(maxx+minx)/2, (maxy+miny)/2]
        diagonal = sqrt((maxx-minx)**2+(maxy-miny)**2)
        
        return Point(centroid).buffer(diagonal/sqrt(2.)/2., cap_style=3)
    
    #shape.geometry = shape.geometry.apply(lambda x: x.minimum_rotated_rectangle)
    shape.geometry = shape.geometry.map(to_square)
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