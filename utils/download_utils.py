import os
import re
import operator
import logging
import requests
import urllib.request
import subprocess
import zipfile
import leafmap
import geojson

from tqdm import tqdm 
import geopandas as gpd
from rapidfuzz import fuzz

import pandas as pd
from shapely.geometry import shape

iso_codes_url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"


class DownloadProgressBar(tqdm):
    """
    A subclass of tqdm to show progress bar for file downloads.

    Methods:
        update_to(b, bsize, tsize): Updates the progress bar.
    """
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        """
        Update the progress bar.

        Args:
            b (int): Number of blocks transferred. Defaults to 1.
            bsize (int): Size of each block. Defaults to 1.
            tsize (int, optional): Total size of the download in bytes. Defaults to None.

        Returns:
            None
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_buildings(
    iso_code: str, 
    vector_dir: str = "data/aerial/vector/", 
    source: str = "ms", 
    country: str = None,
    out_file: str = None,
    verbose: bool = False
) -> None:
    """
    Download building footprints from Microsoft or Google datasets for a given ISO country code.

    Args:
        iso_code (str): ISO alpha-3 country code (e.g., "KEN" for Kenya).
        vector_dir (str, optional): Directory to store the output GeoJSON files. Defaults to "data/aerial/vector/".
        source (str, optional): Data source to use. Either "ms" (Microsoft) or "google". Defaults to "ms".
        verbose (bool, optional): If True, prints progress and debug information. Defaults to False.

    Returns:
        None
    """

    microsoft_url = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    google_url = "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson"
    
    def get_country(iso_code, source: str = "ms"): 
        msf_links = pd.read_csv(microsoft_url)
        
        codes = pd.read_csv(iso_codes_url)
        subcode = codes.query(f"`alpha-3` == '{iso_code}'")
        country = subcode["name"].values[0].replace(" ", "").split("(")[0]
    
        match = None
        max_score = 0
        for msf_country in msf_links.Location.unique():
            # Adjust the Microsoft country name for better matching
            msf_country_ = re.sub(r"(\w)([A-Z])", r"\1 \2", msf_country)
        
            # Calculate the similarity score between the country names
            score = fuzz.token_sort_ratio(country, msf_country_)
            if score > max_score:
                max_score = score
                match = msf_country
        
        return match
        
    # Create output directories
    out_dir = os.path.join(vector_dir, iso_code)
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, f"{source}_buildings")
    os.makedirs(temp_dir, exist_ok=True)

    if country is None:
        country = get_country(iso_code, source)

    # Download building data if the output file doesn't already exist
    if out_file is None:
        out_file = str(os.path.join(out_dir, f"{source}_{iso_code}.geojson"))
    if not os.path.exists(out_file):
        quiet = operator.not_(verbose)
        try:
            if source == "ms":
                download_ms_buildings(
                    country,
                    microsoft_url,
                    out_dir=temp_dir,
                    merge_output=out_file,
                    quiet=quiet,
                    overwrite=True,
                )
            elif source == "google":
                download_google_buildings(
                    country,
                    google_url,
                    out_dir=temp_dir,
                    merge_output=out_file,
                    quiet=quiet,
                    overwrite=True,
                )
        except Exception as e:
            logging.info(e)
            
    return out_file


def download_ms_buildings(
    location: str,
    building_url: str,
    out_dir: str = None,
    merge_output: str = None,
    head=None,
    quiet: bool = False,
    **kwargs,
) -> list:
    """
    Download Microsoft building footprints for a specific country.

    Args:
        location (str): Name of the location (e.g., "Kenya") as listed in the Microsoft dataset.
        building_url (str): URL pointing to the Microsoft building dataset index.
        out_dir (str, optional): Directory to save downloaded GeoJSON files. Defaults to current working directory.
        merge_output (str, optional): If provided, merges individual GeoJSON files into a single file. Defaults to None.
        head (int, optional): If specified, limits the number of files to download. Defaults to None.
        quiet (bool, optional): If True, suppresses console output. Defaults to False.
        **kwargs: Additional arguments passed to `gpd.to_file()`.

    Returns:
        list: List of file paths of the downloaded (and optionally merged) GeoJSON files.
    """

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_links = pd.read_csv(building_url)
    country_links = dataset_links[dataset_links.Location == location]

    if not quiet:
        print(f"Found {len(country_links)} links for {location}")
    if head is not None:
        country_links = country_links.head(head)

    filenames = []
    i = 1

    for _, row in country_links.iterrows():
        if not quiet:
            print(f"Downloading {i} of {len(country_links)}: {row.QuadKey}.geojson")
        i += 1
        filename = os.path.join(out_dir, f"{row.QuadKey}.geojson")
        filenames.append(filename)
        if os.path.exists(filename):
            print(f"File {filename} already exists, skipping...")
            continue
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(shape)
        gdf = gpd.GeoDataFrame(df, crs=4326)
        gdf.to_file(filename, driver="GeoJSON")

    if merge_output is not None:
        if os.path.exists(merge_output):
            print(f"File {merge_output} already exists, skip merging...")
            return filenames
        leafmap.merge_vector(filenames, merge_output, quiet=quiet)

    return merge_output


def download_google_buildings(
    location: str,
    building_url: str,
    out_dir: str = None,
    merge_output: str = None,
    head: str = None,
    keep_geojson: bool = False,
    overwrite: bool = False,
    quiet: bool = False,
    crs: str = "EPSG:4326",
    **kwargs,
) -> list:
    """
    Download Google Open Buildings dataset for a specific location.

    Args:
        location (str): Country name as listed in the Natural Earth dataset (e.g., "Kenya").
        building_url (str): URL to the Google Open Buildings tiles.geojson index.
        out_dir (str, optional): Output directory for downloaded CSV/GeoJSON files. Defaults to current working directory.
        merge_output (str, optional): If provided, merges output into a single GeoJSON file. Defaults to None.
        head (str, optional): If specified, limits the number of building tiles to download. Defaults to None.
        keep_geojson (bool, optional): If True, saves converted GeoJSON files. Defaults to False.
        overwrite (bool, optional): If True, allows overwriting of existing merged file. Defaults to False.
        quiet (bool, optional): If True, suppresses progress messages. Defaults to False.
        crs (str, optional): Coordinate reference system for GeoDataFrame. Defaults to "EPSG:4326".
        **kwargs: Additional arguments passed to `gpd.to_file()`.

    Returns:
        list: List of paths to downloaded files or merged output.
    """
    
    country_url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )

    if out_dir is None:
        out_dir = os.getcwd()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    building_gdf = gpd.read_file(building_url)
    country_gdf = gpd.read_file(country_url)

    country = country_gdf[country_gdf["NAME"] == location]

    if len(country) == 0:
        country = country_gdf[country_gdf["NAME_LONG"] == location]
        if len(country) == 0:
            raise ValueError(f"Could not find {location} in the Natural Earth dataset.")

    gdf = building_gdf[building_gdf.intersects(country.geometry.iloc[0])]
    gdf.sort_values(by="size_mb", inplace=True)

    print(f"Found {len(gdf)} links for {location}.")
    if head is not None:
        gdf = gdf.head(head)

    if len(gdf) > 0:
        links = gdf["tile_url"].tolist()
        leafmap.download_files(links, out_dir=out_dir, quiet=quiet, **kwargs)
        filenames = [os.path.join(out_dir, os.path.basename(link)) for link in links]

        gdfs = []
        for filename in filenames:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(filename)

            # Create a geometry column from the "geometry" column in the DataFrame
            df["geometry"] = df["geometry"].apply(wkt.loads)

            # Convert the pandas DataFrame to a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry="geometry")
            gdf.crs = crs
            if keep_geojson:
                gdf.to_file(
                    filename.replace(".csv.gz", ".geojson"), driver="GeoJSON", **kwargs
                )
            gdfs.append(gdf)

        if merge_output:
            if os.path.exists(merge_output) and not overwrite:
                print(f"File {merge_output} already exists, skip merging...")
            else:
                if not quiet:
                    print("Merging GeoDataFrames ...")
                gdf = gpd.GeoDataFrame(
                    pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
                )
                gdf.to_file(merge_output, **kwargs)

    else:
        print(f"No buildings found for {location}.")

    return merge_output


def download_geoboundary(
    iso_code: str = None,
    out_dir: str = "data/aerial/vector/",
    adm_level: str = "ADM0",
    region: str = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Download administrative boundary from geoBoundaries for a specified ISO country code and admin level.

    Args:
        iso_code (str, optional): ISO alpha-3 country code (e.g., "KEN"). Defaults to None.
        out_dir (str, optional): Directory to save downloaded boundary file. Defaults to "data/aerial/vector/".
        adm_level (str, optional): Administrative level to retrieve (e.g., "ADM0", "ADM1"). Defaults to "ADM0".
        crs (str, optional): Coordinate reference system of the output GeoDataFrame. Defaults to "EPSG:4326".

    Returns:
        geopandas.GeoDataFrame: The downloaded administrative boundary as a GeoDataFrame.
    """
    
    # Define the output file path for the GeoJSON file
    out_file = os.path.join(out_dir, f"{adm_level}_{iso_code}.geojson")

    # Download the GeoJSON if it doesn't already exist locally
    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    gbhumanitarian_url = "https://www.geoboundaries.org/api/current/gbHumanitarian/"
    gbopen_url = "https://www.geoboundaries.org/api/current/gbOpen/"
    natural_earth_url = "https://raw.githubusercontent.com/eFrane/admin0/refs/heads/master/{}/{}.geojson"
    
    url = f"{gbhumanitarian_url}{iso_code}/{adm_level}/"
    download_path = None
    try:
        r = requests.get(url)
        download_path = r.json()["gjDownloadURL"]
    except:
        pass
    
    # Fallback to GBOpen URL if GBHumanitarian URL fails
    if download_path is None:
        try:
            url = f"{gbopen_url}{iso_code}/{adm_level}/"
            r = requests.get(url)
            download_path = r.json()["gjDownloadURL"]
        except:
            codes = pd.read_csv(iso_codes_url)
            subcode = codes.query(f"`alpha-3` == '{iso_code}'")
            iso_code2 = subcode["alpha-2"].iloc[0]
            download_path = natural_earth_url.format(region, iso_code2)
            print(download_path)

    # Download and save the GeoJSON data
    if not os.path.exists(out_file):
        geoboundary = requests.get(download_path).json()
        with open(out_file, "w") as file:
            geojson.dump(geoboundary, file)

    # Read the downloaded GeoJSON into a GeoDataFrame
    geoboundary = gpd.read_file(out_file).fillna("")
    geoboundary = geoboundary.to_crs(crs)

    # Select relevant columns and rename them
    geoboundary = geoboundary[["geometry"]]
    geoboundary.to_file(out_file)
    logging.info(f"{adm_level} geoboundary for {iso_code} saved to {out_file}")

    return geoboundary


def download_ghsl_smod(out_dir: str = "data/aerial/raster/") -> None:
    """
    Download and extract the Global Human Settlement Layer (GHSL) Settlement Model (SMOD) raster file for 2030.

    Args:
        out_dir (str, optional): Directory where the raster data will be stored. 
            Defaults to "data/aerial/raster/".

    Returns:
        str: Path to the extracted GHSL SMOD TIFF file.
    """
    
    # Create the GHSL folder in the rasters directory
    ghsl_dir = os.path.join(out_dir, "GHSL")
    os.makedirs(ghsl_dir, exist_ok=True)

    ghsl_smod_file = "GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V1_0.tif"
    ghsl_smod_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2030_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V1_0.zip"
    ghsl_path = os.path.join(ghsl_dir, ghsl_smod_file)

    # Download and extract the GHSL data if it doesn't already exist
    if not os.path.exists(ghsl_path):
        ghsl_zip = os.path.join(ghsl_dir, "ghsl_smod.zip")

        if not os.path.exists(ghsl_zip):
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="GHSL SMOD") as pbar:
                urllib.request.urlretrieve(ghsl_smod_url, ghsl_zip, reporthook=pbar.update_to)

        if not os.path.exists(ghsl_path):
            with zipfile.ZipFile(ghsl_zip, 'r') as zip_ref:
                zip_ref.extractall(ghsl_dir)

    return ghsl_path


def download_imagery(iso_code: str, out_file: str, image_urls: list = [], out_dir: str = "data/aerial/raster/"):
    """Downloads aerial imagery for a given country using provided URLs.

    Supports both single-image downloads and multi-image mosaics.
    If multiple images are provided, a VRT is built using GDAL tools,
    and the result is converted to a single GeoTIFF file.

    Args:
        iso_code (str): ISO 3-letter country code used to structure directory paths.
        out_file (str): Filename for the final output image (e.g., 'image.tif').
        image_urls (list): List of image URLs to download. If more than one,
                           a mosaic will be built.
        out_dir (str): Base directory where imagery will be saved.
                       Default is "data/aerial/raster/".

    Returns:
        str: Full path to the final saved GeoTIFF image.
    """
    # Handle single image
    if len(image_urls) == 1:
        image_dir = os.path.join(os.getcwd(), out_dir, iso_code)
        os.makedirs(image_dir, exist_ok=True)
        image_file = os.path.join(image_dir, out_file)
        if not os.path.exists(image_file):
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=image_urls[0]) as pbar:
                urllib.request.urlretrieve(image_urls[0], image_file, reporthook=pbar.update_to)
        return image_file

    # Handle multiple images
    image_dir = os.path.join(os.getcwd(), out_dir, iso_code, out_file.split(".")[0])
    os.makedirs(image_dir, exist_ok=True)

    images = []
    for image_url in image_urls:
        image_name = image_url.split("/")[-1]
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=image_name) as pbar:
                urllib.request.urlretrieve(image_url, image_path, reporthook=pbar.update_to)
        images.append(image_path.replace("\\", "/"))

    # Write file list for GDAL
    text_file = os.path.join(image_dir, out_file.split(".")[0]+".txt")
    if not os.path.exists(text_file):
        with open(text_file, 'w') as f:
            for image in images:
                f.write(f"{image}\n")

    # Build VRT mosaic
    vrt_file = os.path.join(image_dir, out_file.split(".")[0]+".vrt")
    if not os.path.exists(vrt_file):
        subprocess.call(f"gdalbuildvrt -input_file_list {text_file} {vrt_file}", shell=True)  

    # Translate VRT to GeoTIFF
    image_file = os.path.join(os.getcwd(), out_dir, iso_code, out_file)
    if not os.path.exists(image_file):
        subprocess.call(f"gdal_translate -of GTiff {vrt_file} {image_file}", shell=True)  
    logging.info(f"Image saved to {image_file}")
    
    return image_file