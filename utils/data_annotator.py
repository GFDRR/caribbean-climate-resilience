import os
import time
import logging
import datetime
import geopandas as gpd

import traitlets
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from ipyannotations.images import MulticlassLabeller
from ipyannotations.controls import togglebuttongroup as tbg

from PIL import Image
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

class DataAnnotator:
    def __init__(
        self, 
        path_to_images: str, 
        path_to_file: str, 
        labels: dict,
        index: int = 0
    ):
        """
        Initializes the annotation tool.

        Args:
            path_to_images (str): Directory containing images.
            path_to_file (str): Path to shapefile or GeoJSON metadata.
            labels (dict): Dictionary of {label_category: [class1, class2, ...]}.
            index (int): Index to start annotation from. Default is 0.
        """
        self.path_to_images = path_to_images
        self.path_to_file = path_to_file
        self.labels = labels
        self.index = index

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

    
    def load_data(self) -> gpd.GeoDataFrame:
        """Loads the geospatial data and attaches full file paths."""
        data = gpd.read_file(os.path.join(os.getcwd(), self.path_to_file))
        data["filepath"] = data.filename.apply(lambda x: os.path.join(os.getcwd(), self.path_to_images, x))
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
            self.data.drop(columns=[col for col in drop_columns if col in self.data.columns]).to_file(self.path_to_file)
            
            self.index += 1
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
        if "annotated" not in self.data.columns:
            self.data["annotated"] = False
    
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
        
    def visualize_annotations(
        self,
        n_rows: int = 5,
        n_cols: int = 5, 
        index: int = 0,
        query: str = None
    ):
        """
        Displays a grid of annotated images.

        Args:
            n_rows (int): Number of rows. Default is 5.
            n_cols (int): Number of columns. Default is 5.
            index (int): Start index. Default is 0.
            query (str): Optional query filter.
        """
        data = self.load_data()
        
        if query:
            data = data.query(query)
        if index:
            data = data.iloc[index:]

        index = 0
        uids = list(data.UID)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        
        # Loop through the rows and columns of the grid
        for i in range(n_rows):
            for j in range(n_cols):
                if index >= len(uids):
                    break
                item = data.query(f"UID == {uids[index]}")
                image = Image.open(item.filepath.values[0])
                axes[i, j].imshow(image) 
                axes[i, j].axis('off')  # Turn off axis labels and ticks
                title = item.filename.values[0] + f"\nIndex: {item.index.values[0]}"
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










