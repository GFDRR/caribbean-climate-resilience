import os
import time
import logging
import datetime

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd

import traitlets
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from ipyannotations.images import MulticlassLabeller
from ipyannotations.controls import togglebuttongroup as tbg

from PIL import Image
from matplotlib import pyplot as plt
from utils import geoutils
from utils import model_utils

from IPython.display import display
from ipywidgets import Layout, GridspecLayout, Button, Image, Tab

import cv2
from io import BytesIO

logging.basicConfig(level=logging.INFO)

class DataAnnotator:
    def __init__(
        self, 
        labels: dict,
        path_to_images: str, 
        path_to_file: str, 
        path_to_embeddings: str = None,
        embeds_dir: str = None,
        index: int = 0,
        mode: str = "aerial",
    ):
        """
        Initializes the annotation tool.

        Args:
            path_to_images (str): Directory containing images.
            path_to_file (str): Path to shapefile or GeoJSON metadata.
            labels (dict): Dictionary of {label_category: [class1, class2, ...]}.
            index (int): Index to start annotation from. Default is 0.
        """
        self.path_to_embeddings = path_to_embeddings
        self.path_to_images = path_to_images
        self.path_to_file = path_to_file
        
        self.labels = labels
        self.index = index
        self.mode = mode
        self.embeddings = None

        self.total_annotations = 0
        self.start_time = self.begin_timer()       

        self.data = self.load_data()            
        self.widget = self.show_annotator()
            
        
    def begin_timer(self) -> None:
        """Starts the annotation timer."""
        self.total_annotations = 0
        return time.time()
        

    def end_timer(self) -> None:
        """Logs the elapsed time and annotation rate."""
        elapsed_time = time.time() - self.start_time
        logging.info(f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}")

        if self.total_annotations > 0:
            annotation_rate = elapsed_time / self.total_annotations
            logging.info(f"Annotation rate: {annotation_rate:.2f} seconds per annotation")
            
        logging.info(f"Total annotations for this session: {self.total_annotations}")

    def _parse_bbox(self, bbox_str: str) -> tuple:
        """Converts a bbox string to a tuple of floats."""
        return tuple(map(float, bbox_str.strip("()[]").split(",")))

    
    def _is_annotated(self, row) -> bool:
        """Checks if the row has any annotation."""
        return any(
            label in row and pd.notnull(row[label])
            for label in self.labels
            if label in row.index
        )
    
    
    def load_data(self) -> gpd.GeoDataFrame:
        """Loads the geospatial data and attaches full file paths."""
        cwd = os.getcwd()
        file_path = os.path.join(cwd, self.path_to_file)
        image_dir = os.path.join(cwd, self.path_to_images)

        if self.mode == "aerial":
            data = gpd.read_file(file_path)
        elif self.mode == "streetview":
            data = pd.read_csv(file_path).reset_index(drop=True)
            data["UID"] = data.index
            data["bbox"] = data["bbox"].apply(self._parse_bbox)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        data["filepath"] = data["filename"].apply(lambda x: os.path.join(image_dir, x))

        if "annotated" not in data.columns:
            data["annotated"] = data.apply(self._is_annotated, axis=1)
            modified = True
        else:
            modified = False
    
        if "clean" not in data.columns:
            data["clean"] = data["filepath"].apply(geoutils.inspect_quality)
            modified = True
    
        if "duplicate" not in data.columns:
            data = geoutils.remove_duplicates(data, image_dir=self.path_to_images, similarity=100)
            modified = True
        
        if modified:
            if self.mode == "aerial":
                data.to_file(self.path_to_file)
            else:
                data.to_csv(self.path_to_file, index=False)
                
        return data
    
    def update_title(self) -> str:
        """Generates HTML title with filename and current labels."""
        row = self.data.iloc[self.index]
        title = f'<h2 style="text-align:center;">{row.filename} <br> Index: {self.index}</h2>'
        title = title + '<h3 style="text-align:center;">'
        for label in list(self.labels)[::-1]:
            if label in self.data.columns:
                value = self.data.iloc[self.index][label]
                if value is not None:
                    title += f"{label.replace('_', ' ').title()}: {value.replace('_', ' ').title()}<br>"
        title += "</h3>"
        return title

        
    def show_prev_annotation(self) -> None:
        """Navigates to the previous image and updates widget display."""
        if self.index == 0:
            logging.info("Already at the beginning of the dataset.")
            return
                
        self.index -= 1
        self.widget.children[0].value = self.update_title() 
        self.widget.display(self.data.iloc[self.index].filepath)


    def store_annotations(self, annotations: list) -> None:
        """Stores submitted annotations and moves to next image."""
        try:
            if annotations:
                if len(annotations) < len(self.labels.keys()):
                    raise ValueError(f"Please add a label for each category.")
                    return
                for label, ann in zip(self.labels, annotations):
                    if ann not in self.labels[label]:
                        raise ValueError(f"Invalid category '{ann}' for label '{label}'.")
                        return 
                        
                    self.data.loc[self.data.index == self.index, label] = ann
                    self.data.loc[self.data.index == self.index, "annotated"] = True
                self.total_annotations += 1

                drop_columns = ["filepath", "exists"]
                data = self.data.drop(columns=[col for col in drop_columns if col in self.data.columns])
                if self.mode == "aerial":
                    data.to_file(self.path_to_file)
                elif self.mode == "streetview":
                    data.to_csv(self.path_to_file)

            while self.index < len(self.data):
                self.index += 1
                if (self.data.iloc[self.index].clean and not self.data.iloc[self.index].duplicate):
                    break
            else:
                logging.info("Reached end of data.")
                return
                    
            self.widget.children[0].value = self.update_title() 
            self.widget.display(self.data.iloc[self.index].filepath)
            
        except IndexError:
            logging.info("Reached end of data.")

        
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
        widget.children[1].children = widget.children[1].children[:len(self.labels[first_label])]
    
        # Rename undo button
        widget.children[-2].children[-1].children[1].description = "Previous"

        def create_toggle_group(label: str, start_index: int) -> tbg.ToggleButtonGroup:
            group = tbg.ToggleButtonGroup(options=self.labels[label])
            group.options = all_categories
            group.children = group.children[start_index:start_index + len(self.labels[label])]
            traitlets.link((widget, "data"), (group, "value"))
            return group
    
        options_widgets = []
        start_index = len(self.labels[first_label])
        for label in label_keys[1:]:
            group = create_toggle_group(label, start_index)
            options_widgets.append(group)
            start_index += len(self.labels[label])
    
        title = widgets.HTML(value=self.update_title())
        widget.children = (title, ) + widget.children[:1] + tuple(options_widgets[::-1]) + (widget.children[1], ) + widget.children[2:]
        
        # Register event handlers
        widget.on_submit(self.store_annotations)
        widget.on_undo(self.show_prev_annotation)
        widget.display(self.data.iloc[self.index].filepath)
    
        layout = Layout(display='flex', justify_content='center', align_self='center', padding='2.5% 0', width='20%')
        widget.children[1].layout = layout
        
        return widget


    def vector_search(
        self, 
        query_index: int = 0, 
        n: int = 10, 
        model_name: str = "FMOW_RGB_GASSL",
        exclude_annotated: bool = True
    ):            
        self.embeddings = model_utils.generate_embeddings(
            data=self.data,
            image_dir=self.path_to_images,
            out_dir=self.path_to_embeddings,
            model_name=model_name
        )   

        query = self.embeddings.iloc[query_index, :-1].to_numpy()
        query_image = model_utils.load_image(self.data.iloc[query_index].filepath)
        
        embeddings = self.embeddings.copy()
        valid_uids = list(self.data.UID.unique())
        if exclude_annotated:
            valid_uids = list(self.data[
                (self.data.clean == True)
                & (self.data.annotated == False) 
                & (self.data.duplicate == False)
            ].UID.unique())
        
        embeddings = embeddings[embeddings.UID.isin(valid_uids)]
        indexes = embeddings.index
        embeddings_vector = embeddings.iloc[:, :-1].to_numpy()
        
        indexes = model_utils.top_n_similarity(query, embeddings_vector, indexes, n=n+1)
        indexes = [index[0] for index in indexes[1:]]
        top_n_uids = embeddings.loc[indexes].UID

        data = self.data[self.data.UID.isin(top_n_uids)].reindex(indexes)

        uids = list(data.UID)
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        title = f'Query Image \nIndex: {query_index}'
        for label in self.labels.keys():
            if label in data.columns:
                title += f'\n{self.data.iloc[query_index][label]}'

        # Plot the main image
        ax.imshow(query_image)
        ax.axis('off')
        ax.set_title(title)
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        return self.validate_data(data.copy(), query_index, n_rows=int(n/5), n_cols=5)


    def validate_data(self, data, query_index, n_rows, n_cols) -> GridspecLayout:
        # Create a GridspecLayout for displaying images and buttons
        row_inc= 3
        grid = GridspecLayout(n_rows * row_inc + n_rows+1, n_cols)
    
        def add_image(item):
            image = model_utils.load_image(item.filepath)
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
                height=250
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
            self.data.to_file(self.path_to_file)
            
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
        
        
    def visualize_annotations(
        self,
        n_rows: int = 5,
        n_cols: int = 5, 
        index: int = 0,
        query: str = None,
        randomize: bool = False,
        show_filename: bool = True,
        show_clean_only: bool = True
    ):
        """
        Displays a grid of annotated images.

        Args:
            n_rows (int): Number of rows. Default is 5.
            n_cols (int): Number of columns. Default is 5.
            index (int): Start index. Default is 0.
            query (str): Optional query filter.
        """

        data = self.data.copy()
        if query:
            data = data.query(query)
        if index:
            data = data.iloc[index:]
        if randomize:
            data = data[data.annotated==True].sample(frac=1.0)
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
                image = model_utils.load_image(item.filepath.values[0])
                
                axes[i, j].imshow(image) 
                axes[i, j].axis('off') 
                if show_filename:
                    title = item.filename.values[0] + f"\nIndex: {item.index.values[0]}"
                else:
                    title = f"Index: {item.index.values[0]}"
                for label in self.labels.keys():
                    if label in data.columns:
                        title += f'\n{item[label].values[0]}'
                axes[i, j].set_title(title)
                index += 1
        
        # Adjust spacing between subplots
        plt.tight_layout()
        plt.axis('off'),
        
        # Show the plot
        plt.show()