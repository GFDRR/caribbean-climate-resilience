import os
import time
import logging
import datetime
import shutil
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd

import traitlets
import ipywidgets as widgets
from ipywidgets import Layout, GridspecLayout, Button, Image, Tab
from IPython.display import display
from ipyannotations.images import MulticlassLabeller
from ipyannotations.controls import togglebuttongroup as tbg

from PIL import Image as PILImage
from matplotlib import pyplot as plt

from utils import data_utils
from utils import embed_utils

logging.basicConfig(level=logging.INFO)


class DataAnnotator:
    """Interactive image annotation tool for aerial or street view data."""

    def __init__(
        self,
        labels: dict,
        path_to_images: str,
        path_to_file: str,
        path_to_embeddings: str = None,
        embeds_dir: str = None,
        index: int = 0,
        mode: str = "aerial",
    ) -> None:
        """
        Initializes the annotation tool.

        Args:
            labels (dict): Dictionary of {label_category: [class1, class2, ...]}.
            path_to_images (str): Directory containing images.
            path_to_file (str): Path to shapefile or GeoJSON metadata.
            path_to_embeddings (str, optional): Path to save embeddings. Defaults to None.
            embeds_dir (str, optional): Directory for embeddings. Defaults to None.
            index (int, optional): Starting index for annotation. Defaults to 0.
            mode (str, optional): Mode of operation, either 'aerial' or 'streetview'. Defaults to "aerial".
        """
        # Initialize variables
        self.path_to_embeddings = path_to_embeddings
        self.path_to_images = path_to_images
        self.path_to_file = path_to_file
        self.labels = labels
        self.index = index
        self.mode = mode
        self.embeddings = None

        # Initialize tracking variables
        self.total_annotations = 0
        self.start_time = self.begin_timer()

        # Load data with verification
        self.data = self.load_data()

        # Build UI
        self.widget = self.show_annotator()

    def begin_timer(self) -> float:
        """
        Starts the annotation timer.

        Returns:
            float: The current time in seconds since the epoch.
        """
        self.total_annotations = 0  # Reset count of annotations
        return time.time()  # Return current time in seconds since epoch

    def end_timer(self) -> None:
        """
        Logs the elapsed time and annotation rate.
        """
        elapsed_time = time.time() - self.start_time  # Calculate time difference
        logging.info(f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}")

        # Log average annotation time if any annotations were done
        if self.total_annotations > 0:
            annotation_rate = elapsed_time / self.total_annotations
            logging.info(
                f"Annotation rate: {annotation_rate:.2f} seconds per annotation"
            )

        # Log total annotations done
        logging.info(f"Total annotations for this session: {self.total_annotations}")

    def _parse_bbox(self, bbox_str: str) -> tuple:
        """
        Converts a bounding box string to a tuple of floats.

        Args:
            bbox_str (str): Bounding box string.

        Returns:
            tuple: Tuple of floats representing the bounding box.
        """
        # Strip parentheses or brackets, split by comma, convert each to float and return as tuple
        return tuple(map(float, bbox_str.strip("()[]").split(",")))

    def _is_annotated(self, row) -> bool:
        """
        Checks if the row has any annotation.

        Args:
            row (Series): A row from the DataFrame.

        Returns:
            bool: True if annotated, False otherwise.
        """
        # Check if any label exists in row and has a non-null value
        return any(
            label in row and pd.notnull(row[label])
            for label in self.labels
            if label in row.index
        )

    def save_data(self) -> bool:
        """Safely saves data with verification."""
        try:
            # Prepare data for saving
            save_data = self.data.copy()
            save_data = save_data.replace("nan", np.nan)
            if "filepath" in save_data.columns:
                save_data = save_data.drop(columns=["filepath"])

            # Save based on mode
            if self.mode == "aerial":
                save_data.to_file(self.path_to_file, driver="GeoJSON")
            else:
                save_data.to_csv(self.path_to_file, index=False)

            # Verify save by reading back
            if self.mode == "aerial":
                # test_data = gpd.read_file(self.path_to_file)
                test_data = self.load_data()
            else:
                pd.read_csv(self.path_to_file)

            if len(test_data) != len(self.data):
                raise ValueError("Saved data length mismatch")

            logging.info(f"Saved {len(test_data)} records")
            return True

        except Exception as e:
            logging.error(f"Save failed: {str(e)}")
            return False

    def load_data(self) -> gpd.GeoDataFrame:
        """
        Loads the geospatial data and attaches full file paths.

        Returns:
            GeoDataFrame or DataFrame: Loaded data with additional columns.
        """
        cwd = os.getcwd()  # Current working directory
        file_path = os.path.join(cwd, self.path_to_file)  # Full path to metadata file
        image_dir = os.path.join(
            cwd, self.path_to_images
        )  # Full path to images directory

        # Load data differently based on mode
        if self.mode == "aerial":
            data = gpd.read_file(file_path)
        elif self.mode == "streetview":
            data = pd.read_csv(file_path)
            data["bbox"] = data["bbox"].apply(self._parse_bbox)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Initialize required columns
        if "UID" not in data.columns:
            data["UID"] = data.index

        # Attach full file path to each image filename
        if "filepath" not in data.columns:
            data["filepath"] = data["filename"].apply(
                lambda x: os.path.join(self.path_to_images, x)
            )

        # Initialize special columns if missing
        cols_to_add = {
            "annotated": lambda d: d.apply(self._is_annotated, axis=1),
            "clean": lambda d: d["filepath"].apply(data_utils.inspect_quality),
            "duplicate": lambda d: pd.Series(False, index=d.index),
        }

        modified = False
        for col, init_func in cols_to_add.items():
            if col not in data.columns:
                data[col] = init_func(data)
                modified = True

        # Save if we modified the structure
        if modified and not self.save_data():
            raise RuntimeError("Failed to save initialized data")

        return data  # Return loaded and processed data

    def update_title(self) -> str:
        """
        Generates HTML title with filename and current labels.

        Returns:
            str: HTML string for the title.
        """
        row = self.data.iloc[self.index]  # Current row

        # Main title with filename
        title = f'<h2 style="text-align:center;">{row.filename} <br> Index: {self.index}</h2>'
        title = title + '<h3 style="text-align:center;">'  # Subtitle with labels

        # Iterate through labels in reverse order to add label: value pairs if present
        for label in list(self.labels)[::-1]:
            if label in self.data.columns:
                value = self.data.iloc[self.index][label]
                if value is not None:
                    label = label.replace("_", " ").title()
                    value = value.replace("_", " ").title()
                    title += f"{label}: {value}<br>"
        title += "</h3>"
        return title

    def show_prev_annotation(self) -> None:
        """
        Navigates to the previous image and updates widget display.
        """
        if self.index == 0:
            logging.info("Already at the beginning of the dataset.")
            return

        self.index -= 1  # Decrement index to previous image
        self.widget.children[0].value = self.update_title()  # Update widget title
        self.widget.display(
            self.data.iloc[self.index].filepath
        )  # Display previous image

    def _move_to_next_valid(self) -> bool:
        """Moves to next valid image, returns True if successful."""
        remaining = self.data[
            (self.data.index > self.index) & (self.data.clean) & (~self.data.duplicate)
        ]

        if not remaining.empty:
            self.index = remaining.index[0]
            return True
        return False

    def _update_display(self) -> None:
        """Updates widget display with current image."""
        self.widget.children[0].value = self.update_title()
        self.widget.display(self.data.iloc[self.index].filepath)

    def store_annotations(self, annotations: list) -> None:
        """
        Stores submitted annotations and moves to the next image.

        Args:
            annotations (list): List of annotations corresponding to labels.
        """
        if annotations:
            # Check if all categories have labels annotated
            if len(annotations) != len(self.labels):
                raise ValueError("Please provide all required annotations")

            # Validate each annotation against allowed categories and store it
            for label, ann in zip(self.labels, annotations):
                if ann not in self.labels[label]:
                    raise ValueError(f"Invalid value '{ann}' for label '{label}'")

                self.data.at[self.index, label] = ann
                self.data.at[self.index, "annotated"] = True

            # Save and verify
            if not self.save_data():
                raise RuntimeError("Failed to save annotations")
            self.total_annotations += 1

        if self._move_to_next_valid():
            self._update_display()
        else:
            logging.info("Reached end of dataset")

    def show_annotator(self) -> MulticlassLabeller:
        """
        Builds the annotation UI.

        Returns:
            MulticlassLabeller: The interactive widget for annotation.
        """

        # Load existing annotations
        all_categories = [cat for group in self.labels.values() for cat in group]
        label_keys = list(self.labels.keys())
        first_label = label_keys[0]

        # Set up the main labeling widget
        widget = MulticlassLabeller(options=self.labels[first_label])
        widget.class_selector.options = all_categories

        # Trim excess buttons for the first label
        widget.children[1].children = widget.children[1].children[
            : len(self.labels[first_label])
        ]

        # Rename undo button
        widget.children[-2].children[-1].children[1].description = "Previous"

        def create_toggle_group(label: str, start_index: int) -> tbg.ToggleButtonGroup:
            group = tbg.ToggleButtonGroup(options=self.labels[label])
            group.options = all_categories
            group.children = group.children[
                start_index : start_index + len(self.labels[label])
            ]
            traitlets.link((widget, "data"), (group, "value"))
            return group

        options_widgets = []
        start_index = len(self.labels[first_label])
        for label in label_keys[1:]:
            group = create_toggle_group(label, start_index)
            options_widgets.append(group)
            start_index += len(self.labels[label])

        title = widgets.HTML(value=self.update_title())
        widget.children = (
            (title,)
            + widget.children[:1]
            + tuple(options_widgets[::-1])
            + (widget.children[1],)
            + widget.children[2:]
        )

        # Register event handlers
        widget.on_submit(self.store_annotations)
        widget.on_undo(self.show_prev_annotation)
        widget.display(self.data.iloc[self.index].filepath)

        layout = Layout(
            display="flex",
            justify_content="center",
            align_self="center",
            padding="2.5% 0",
            width="20%",
        )
        widget.children[1].layout = layout

        return widget

    def visualize_annotations(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        index: int = 0,
        query: str = None,
        randomize: bool = False,
        show_filename: bool = True,
        show_clean_only: bool = True,
    ):
        """
        Displays a grid of annotated images with optional filtering and labeling.

        Shows images in a matplotlib subplot grid with titles that can include filenames,
        indices, and annotation labels. Allows filtering by query string, randomizing
        order, and showing only clean (non-duplicate) items.

        Args:
            n_rows (int, optional): Number of rows in the display grid. Defaults to 5.
            n_cols (int, optional): Number of columns in the display grid. Defaults to 5.
            index (int, optional): Starting index from which to display images. Defaults to 0.
            query (str, optional): Query string to filter the data before display. Defaults to None.
            randomize (bool, optional): If True, randomizes the order of displayed annotated images. Defaults to False.
            show_filename (bool, optional): Whether to show filenames in the image titles. Defaults to True.
            show_clean_only (bool, optional): If True, shows only items marked as clean and not duplicates. Defaults to True.

        Returns:
            None: Displays a matplotlib figure with the annotated images grid.
        """
        data = self.data.copy()
        if query:
            data = data.query(query)
        if index:
            data = data.iloc[index:]
        if randomize:
            data = data[data.annotated == True].sample(frac=1.0)
        if show_clean_only:
            data = data[data["clean"] == True]
            data = data[data["duplicate"] == False]

        index = 0
        uids = list(data.UID)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

        # Loop through the rows and columns of the grid
        for i in range(n_rows):
            for j in range(n_cols):
                if index >= len(uids):
                    break
                item = data.query(f"UID == {uids[index]}")
                image = data_utils.load_image(item.filepath.values[0])

                axes[i, j].imshow(image)
                axes[i, j].axis("off")
                if show_filename:
                    title = item.filename.values[0] + f"\nIndex: {item.index.values[0]}"
                else:
                    title = f"Index: {item.index.values[0]}"
                for label in self.labels.keys():
                    if label in data.columns:
                        title += f"\n{item[label].values[0]}"
                axes[i, j].set_title(title)
                index += 1

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.axis("off"),

        # Show the plot
        plt.show()

    def vector_search(
        self,
        query_index: int = 0,
        n: int = 10,
        model_name: str = "FMOW_RGB_GASSL",
        exclude_annotated: bool = True,
    ):
        """
        Performs vector search to find similar images.

        Args:
            query_index (int, optional): Index of the query image. Defaults to 0.
            n (int, optional): Number of similar images to retrieve. Defaults to 10.
            model_name (str, optional): Name of the model to use for embeddings. Defaults to "FMOW_RGB_GASSL".
            exclude_annotated (bool, optional): Whether to exclude already annotated images. Defaults to True.

        Returns:
            GridspecLayout: Grid layout of similar images for validation.
        """
        self.embeddings = embed_utils.generate_embeddings(
            data=self.data,
            image_dir=self.path_to_images,
            out_dir=self.path_to_embeddings,
            model_name=model_name,
        )

        query = self.embeddings.iloc[query_index, :-1].to_numpy()
        query_image = data_utils.load_image(self.data.iloc[query_index].filepath)

        embeddings = self.embeddings.copy()
        valid_uids = list(self.data.UID.unique())
        if exclude_annotated:
            valid_uids = list(
                self.data[
                    (self.data.clean == True)
                    & (self.data.annotated == False)
                    & (self.data.duplicate == False)
                ].UID.unique()
            )

        embeddings = embeddings[embeddings.UID.isin(valid_uids)]
        indexes = embeddings.index
        embeddings_vector = embeddings.iloc[:, :-1].to_numpy()

        indexes = embed_utils.top_n_similarity(
            query, embeddings_vector, indexes, n=n + 1
        )
        indexes = [index[0] for index in indexes[1:]]
        top_n_uids = embeddings.loc[indexes].UID

        data = self.data[self.data.UID.isin(top_n_uids)].reindex(indexes)

        uids = list(data.UID)
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        title = f"Query Image \nIndex: {query_index}"
        for label in self.labels.keys():
            if label in data.columns:
                title += f"\n{self.data.iloc[query_index][label]}"

        # Plot the main image
        ax.imshow(query_image)
        ax.axis("off")
        ax.set_title(title)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        return self.validate_data(data.copy(), query_index, n_rows=int(n / 5), n_cols=5)

    def validate_data(self, data, query_index, n_rows, n_cols) -> GridspecLayout:
        """
        Creates an interactive grid layout to display images alongside validation buttons.

        Each image in the grid corresponds to a data item, and users can toggle its
        annotated status by clicking the associated button. The button's appearance
        changes to reflect the selection state. Validation updates are saved to disk.

        Args:
            data (pd.DataFrame): DataFrame containing data items to be validated. Must include 'annotated' column and 'filepath' for images.
            query_index (int): Index of the reference row used to copy label values when selecting an item.
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.

        Returns:
            GridspecLayout: A widget grid displaying images with corresponding validation buttons.
        """
        row_inc = 3
        grid = GridspecLayout(n_rows * row_inc + n_rows + 1, n_cols)

        def add_image(item):
            image = data_utils.load_image(item.filepath)
            membuf = BytesIO()
            from PIL import Image

            Image.fromarray(image).save(membuf, format="png")

            from ipywidgets import Image

            image = Image(
                value=membuf.getvalue(),
                format="png",
                layout=Layout(
                    justify_content="center",
                    border="solid",
                ),
                width=250,
                height=250,
            )
            return image

        def on_button_click(button):
            # Function to handle button click events for validation
            index = int(button.description.split(" ")[0])
            item = self.data.iloc[index]

            change_value = True
            selected = "Selected"
            button_style = "warning"

            if item["annotated"] == True:
                selected = "Unselected"
                change_value = False
                button_style = "primary"
                for label in self.labels.keys():
                    if label in self.data.columns:
                        self.data.loc[index, label] = None
            else:
                for label in self.labels.keys():
                    if label in self.data.columns:
                        self.data.loc[index, label] = self.data.iloc[query_index][label]

            self.data.loc[index, "annotated"] = change_value
            # Save and verify
            if not self.save_data():
                raise RuntimeError("Failed to save annotations")

            button.button_style = button_style
            button.description = f"{index} {selected}"

        def create_button(index, item):
            # Function to create validation buttons
            val = item["annotated"]
            selected = "Unselected"
            if val == True:
                selected = "Selected"
                button_style = "warning"
            else:
                button_style = "primary"
            description = f"{index} {selected}"

            return Button(
                description=description,
                button_style=button_style,
                layout=Layout(
                    justify_content="center", border="solid", width="auto", height="10"
                ),
            )

        # Populate the grid with images and buttons
        row_index, col_index = 1, 0
        for index, item in data.iterrows():
            grid[row_index : row_index + row_inc, col_index] = add_image(item)
            button = create_button(index, item)
            button.on_click(on_button_click)
            grid[row_index + row_inc, col_index] = button

            col_index += 1
            if col_index >= n_cols:
                row_index += row_inc + 1
                col_index = 0

        return grid
