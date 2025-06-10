import os
import logging
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.mask

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from utils import download_utils
from utils import geoutils

logging.basicConfig(level=logging.INFO)


class DataSampler:
    def __init__(
        self,
        iso_code: str,
        bldgs_file: str = None,
        bldgs_src: str = None,
        aerial_image_file: str = None,
        image_urls: list = [],
        grid_length: float = 500,
        grid_width: float = 500,
        vector_dir: str = "data/aerial/vector/",
        raster_dir: str = "data/aerial/raster/",
        n_samples: int = 100,
        rurban_ratio: int = 2.5,
        scale: float = 1.5,
        min_area: float = 25,
        crs: str = "EPSG:4326",
        country: str = None,
        region: str = None,
    ):
        """
        Initializes the DataSampler object and prepares the sampling grid, building samples, and raster tiles.

        Args:
            iso_code (str): ISO country code.
            bldgs_file (str, optional): Path to a building footprint file. Defaults to None.
            bldgs_src (str, optional): Online source for downloading building data. Defaults to None.
            aerial_image_file (str, optional): Filename of the aerial imagery file. Defaults to None.
            grid_length (float, optional): Length of each grid cell in meters. Defaults to 500.
            grid_width (float, optional): Width of each grid cell in meters. Defaults to 500.
            vector_dir (str, optional): Directory path for vector data. Defaults to "data/aerial/vector/".
            raster_dir (str, optional): Directory path for raster data. Defaults to "data/aerial/raster/".
            n_samples (int, optional): Number of grid samples to draw. Defaults to 100.
            rurban_ratio (float, optional): Ratio of rural to urban samples. Defaults to 1.25.
            scale (float, optional): Scale factor for cropping tiles. Defaults to 1.5.
            crs (str, optional): Coordinate Reference System for output data. Defaults to "EPSG:4326".
            region (str, optional: Region of the country as specified in https://github.com/eFrane/admin0/tree/master
        """

        self.iso_code = iso_code
        self.region = region
        self.min_area = min_area
        self.crs = crs

        self.vector_dir = os.path.join(os.getcwd(), vector_dir, iso_code)
        os.makedirs(self.vector_dir, exist_ok=True)

        self.raster_dir = os.path.join(os.getcwd(), raster_dir)
        os.makedirs(self.raster_dir, exist_ok=True)

        self.bldgs = self.load_bldgs(bldgs_file, bldgs_src, country)
        self.grid = self.generate_grid(grid_length, grid_width)
        self.grid_samples = self.sample_grids(self.grid, n_samples, rurban_ratio)

        self.bldgs_samples = gpd.sjoin(
            self.bldgs,
            self.grid_samples[["geometry", "grid_id"]],
            predicate="intersects",
        )
        self.bldgs_samples = self.bldgs_samples.drop(["index_right"], axis=1)
        self.bldgs_samples = self.bldgs_samples.drop_duplicates("geometry")
        self.aerial_image_file = self.load_aerial_image(aerial_image_file, image_urls)
        self.bldgs_samples = self.generate_tiles(
            self.bldgs_samples, self.aerial_image_file, scale
        )

    def plot_samples(self):
        """
        Plots the grid and sampled grid cells using Matplotlib.

        This method creates a matplotlib figure and plots two layers:
        - The full grid (in blue)
        - The sampled grid cells (in red), with a legend

        It also logs the total number of building samples relative to the full dataset,
        and provides a count of the urban vs. rural classification.

        Logs:
            - Total number of building samples over the total buildings
            - Urban/rural distribution in the sampled buildings
        """
        fig, ax = plt.subplots(figsize=(13, 13))
        self.grid.plot(ax=ax, edgecolor="blue", facecolor=None)
        self.grid_samples.plot(ax=ax, edgecolor="red", facecolor=None, legend=True)
        logging.info(
            f"Total number of building samples: {len(self.bldgs_samples)}/{len(self.bldgs)}"
        )
        logging.info(
            f"Urban/rural ratio: {self.bldgs_samples['rurban'].value_counts()}"
        )

    def generate_tiles(self, bldgs_samples, in_file, scale):
        """
        Generates image tiles by cropping the aerial imagery using sampled buildings.

        Args:
            bldgs_samples (GeoDataFrame): GeoDataFrame containing sampled buildings.
            in_file (str): Path to the input aerial imagery file.
            scale (float): Scaling factor to enlarge the bounding box for cropping.

        """

        def shuffle_grouped_dataframe(df, group_column):
            """
            Groups a Pandas DataFrame by a column and shuffles the groups.

            Args:
                df (pd.DataFrame): The input DataFrame.
                group_column (str): The column to group by.

            Returns:
                pd.DataFrame: A new DataFrame with shuffled groups.
            """
            grouped = df.groupby(group_column)
            group_keys = df[group_column].unique()
            random.shuffle(group_keys)

            shuffled_df = pd.concat([grouped.get_group(key) for key in group_keys])
            shuffled_df = gpd.GeoDataFrame(shuffled_df, geometry="geometry")
            shuffled_df = shuffled_df.reset_index(drop=True)
            return shuffled_df

        out_dir = os.path.join(os.getcwd(), self.raster_dir, self.iso_code, "tiles")
        os.makedirs(out_dir, exist_ok=True)

        total = len(bldgs_samples)
        filenames, exists = [], []
        for index, bldg in tqdm(
            bldgs_samples.iterrows(),
            total=total,
            desc=f"Processing {total} image tiles",
        ):
            filename = f"{bldg.UID}_{bldg.grid_id}.tif"
            out_file = os.path.join(out_dir, filename)
            filenames.append(filename)

            exist = True
            if not os.path.exists(out_file):
                try:
                    geoutils.crop_tile(
                        bldgs_samples[bldgs_samples.UID == bldg.UID].copy(),
                        scale,
                        in_file,
                        out_file,
                    )
                except Exception as e:
                    exist = False
                    pass
            exists.append(exist)
        logging.info(f"Building tiles saved to {out_dir}.")

        out_file = os.path.join(
            os.getcwd(), self.vector_dir, f"tiles_{self.iso_code}.geojson"
        )
        if not os.path.exists(out_file):
            bldgs_samples["filename"] = filenames
            bldgs_samples["exists"] = exists
            bldgs_samples = bldgs_samples[bldgs_samples.exists == True].drop(
                columns=["exists"]
            )

            shuffle_grouped_dataframe(bldgs_samples, "grid_id").to_file(out_file)
            logging.info(f"Building sample geojson saved to {out_file}.")

        bldgs_samples = gpd.read_file(out_file)

        out_file = os.path.join(
            os.getcwd(), self.vector_dir, f"tiles_{self.iso_code}.geojson"
        )
        if not os.path.exists(out_file):
            bldgs_samples.to_file(out_file, driver="GeoJSON")

        return bldgs_samples

    def sample_grids(
        self, grid: gpd.GeoDataFrame, n_samples: int = 100, rurban_ratio: float = 1.5
    ):
        """
        Samples rural and urban grid cells according to the specified ratio.

        Args:
            grid (GeoDataFrame): Grid over which to sample.
            n_samples (int, optional): Total number of samples. Defaults to 100.
            rurban_ratio (float, optional): Ratio of rural to urban samples. Defaults to 1.5.

        Returns:
            GeoDataFrame: Sampled grid cells.
        """
        n_samples = min(n_samples, len(self.grid))
        logging.info(f"Sampling {n_samples} grid tiles for {self.iso_code}...")
        out_file = os.path.join(self.vector_dir, f"ann_grid_{self.iso_code}.geojson")
        if os.path.exists(out_file):
            logging.info(f"Reading grid sample file: {out_file}")
            return gpd.read_file(out_file)

        urban = grid[grid["rurban"] == "urban"]
        rural = grid[grid["rurban"] == "rural"]

        n_urban_samples = int(n_samples / (rurban_ratio + 1))
        urban_samples = urban.sample(n_urban_samples)
        rural_samples = rural.sample(n_samples - n_urban_samples)

        samples = pd.concat([urban_samples, rural_samples])
        samples = gpd.GeoDataFrame(samples, geometry="geometry")

        samples.to_file(out_file)
        logging.info(f"Grid samples saved to {out_file}")
        return samples

    def generate_grid(
        self, length: float = 500, width: float = 500
    ) -> gpd.GeoDataFrame:
        """
        Creates a spatial grid over the country boundary and assigns rural/urban labels.

        Args:
            length (float, optional): Grid length in meters. Defaults to 500.
            width (float, optional): Grid width in meters. Defaults to 500.

        Returns:
            GeoDataFrame: Generated spatial grid with rural/urban classifications.
        """
        logging.info(f"Generating {length} x {width} grid for {self.iso_code}...")
        geoboundary = download_utils.download_geoboundary(
            iso_code=self.iso_code,
            out_dir=self.vector_dir,
            crs=self.crs,
            region=self.region,
        )

        out_file = os.path.join(self.vector_dir, f"grid_{self.iso_code}.geojson")
        if os.path.exists(out_file):
            logging.info(f"Reading grid file {out_file}")
            return gpd.read_file(out_file)

        geoboundary = geoboundary.to_crs(geoboundary.estimate_utm_crs())
        xmin, ymin, xmax, ymax = geoboundary.total_bounds

        cols = list(np.arange(xmin, xmax + width, width))
        rows = list(np.arange(ymin, ymax + length, length))

        polygons = []
        for x in cols[:-1]:
            for y in rows[:-1]:
                polygons.append(
                    Polygon(
                        [
                            (x, y),
                            (x + width, y),
                            (x + width, y + length),
                            (x, y + length),
                        ]
                    )
                )

        grid = gpd.GeoDataFrame({"geometry": polygons}, crs=geoboundary.crs)
        grid = gpd.sjoin(grid, geoboundary, predicate="intersects").drop(
            columns=["index_right"]
        )

        if self.bldgs is not None:
            grid = grid.to_crs(self.bldgs.crs)
            grid = gpd.sjoin(grid, self.bldgs, predicate="contains")

        grid = grid[["geometry"]].drop_duplicates()
        grid = self.assign_rurban(grid)
        grid = grid.reset_index()
        grid["grid_id"] = grid.index

        grid.to_file(out_file, driver="GeoJSON")
        logging.info(
            f"{length} x {width} grids for {self.iso_code} saved to {out_file}"
        )

        return grid

    def assign_rurban(self, data: gpd.GeoDataFrame):
        """
        Assigns rural/urban classification to geometries based on GHSL SMOD data.

        Args:
            data (GeoDataFrame): Input data for classification.

        Returns:
            GeoDataFrame: Data with added 'rurban' classification.
        """

        ghsl_smod_file = download_utils.download_ghsl_smod(out_dir=self.raster_dir)
        with rio.open(ghsl_smod_file) as ghsl_smod:
            data = data.to_crs(ghsl_smod.crs)
            coords = data.centroid.get_coordinates().values
            data["ghsl_smod"] = [x[0] for x in ghsl_smod.sample(coords)]

        # Define rural class codes
        rural = [10, 11, 12, 13]
        data["rurban"] = "urban"
        data.loc[data["ghsl_smod"].isin(rural), "rurban"] = "rural"
        data = data.to_crs(self.crs)
        return data

    def load_aerial_image(self, aerial_image_file, image_urls):
        """Loads or downloads the aerial image for the specified country.

        If the aerial image file does not exist locally, it will be downloaded
        using predefined URLs.

        Args:
            aerial_image_file (str): The name of the aerial image file.
            image_urls (dict): A dictionary mapping ISO country codes to their aerial imagery URLs.

        Returns:
            str: The full path to the local aerial image file.
        """
        aerial_image_file = os.path.join(
            os.getcwd(), self.raster_dir, self.iso_code, aerial_image_file
        )
        if not os.path.exists(aerial_image_file):
            download_utils.download_imagery(
                self.iso_code, aerial_image_file, image_urls
            )
        return aerial_image_file

    def load_bldgs(
        self, bldgs_file: str = None, bldgs_src: str = None, country: str = None
    ):
        """
        Loads building footprints from file or online source and assigns rural/urban classification.

        Args:
            iso_code (str): ISO country code.
            bldgs_file (str, optional): Path to the building file. Defaults to None.
            bldgs_src (str, optional): Source to download building data from. Defaults to None.

        Returns:
            GeoDataFrame: Loaded and classified building data.
        """
        bldgs_file = os.path.join(self.vector_dir, bldgs_file)
        if not os.path.exists(bldgs_file):
            logging.info(f"Downloading buildings for {self.iso_code}...")
            bldgs_file = download_utils.download_buildings(
                iso_code=self.iso_code,
                source=bldgs_src,
                country=country,
                out_file=bldgs_file,
            )

        logging.info(f"Loading buildings for {self.iso_code}...")
        bldgs = gpd.read_file(bldgs_file)
        bldgs = bldgs.to_crs(bldgs.estimate_utm_crs())
        bldgs["area"] = bldgs.area
        bldgs = bldgs.query(f"area > {self.min_area}")
        bldgs = bldgs.to_crs(self.crs)

        if "UID" not in bldgs.columns:
            bldgs["UID"] = bldgs.reset_index().index
        bldgs = bldgs[["UID", "geometry"]]
        bldgs = self.assign_rurban(bldgs)
        bldgs = bldgs.to_crs(self.crs)

        return bldgs
