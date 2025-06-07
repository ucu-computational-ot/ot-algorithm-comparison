from uot.data.measure import DiscreteMeasure, GridMeasure

def load_csv_as_discrete(path: str) -> DiscreteMeasure:
    "Loads the discrete measure from the defined path to the data."
    raise NotImplementedError()


def load_image_as_grid(path: str, bins_per_channel: int = 32) -> GridMeasure:
    "Loads the image as the grid measure from the defined path."
    raise NotImplementedError()