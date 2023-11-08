import pickle as pkl
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.representation.abhye_hypergraph import ABHyeHypergraph
from src.data.representation.hypergraph import Hypergraph

def load_data(
    hye_file: str = "",
    weight_file: str = "",
    pickle_file: str = "",
) -> Hypergraph:
    """Load a hypergraph dataset.
    Utility function for loading hypergraph data provided in various formats.
    Currently two formats are supported:
    - a pair (hye_file, weight_file) specifying the hyperedges and relative weights
    - the path to a serialized hypergraph, to be loaded via the pickle package.

    The function raises an error if more than one of the options above is given as
    input.

    Parameters
    ----------
    hye_file: txt file containing the hyperedges in the dataset.
        If provided, also weight_file needs to be provided.
    weight_file:  txt file containing the hyperedge weights in the dataset.
        If provided, also hye_file needs to be provided.
    pickle_file: path to a .pkl file to be loaded via the pickle package.

    Returns
    -------
    The loaded hypergraph.
    """
    # Check that the data is provided exactly in one format:
    # - as a real real_dataset name
    # - in the form of two files, specifying the hyperedges and relative weights
    # - in the form of a pickle file, containing a serialized hypergraph
    inputs = (
        (bool(hye_file) or bool(weight_file)) + bool(pickle_file)
    )
    if inputs == 0:
        raise ValueError("No input hypergraph has been provided.")
    if inputs >= 2:
        raise ValueError("Provide only one valid input hypergraph format.")

    if pickle_file:
        with open(pickle_file, "rb") as file:
            return pkl.load(file)

    if hye_file or weight_file:
        if not hye_file and weight_file:
            raise ValueError("Provide both the hyperedge and weight files.")
        return ABHyeHypergraph.load(hye_file, weight_file)


def load_attributes(file_path: str, attributes: Optional[List]) -> np.ndarray:
    """Load attributes.
    Utility function for loading the attributes saved as .csv file. 
    Each attribute is one-hot encoded.
    
    Parameters
    ----------
    file_path: a path to a .csv file
    attributes: a list of column names of the .csv file to be used as attributes. 
        If not provided a Metadata column is used.

    Returns
    -------
    Numpy array with attributes.
    """
    if attributes is None:
        attributes = ['Metadata']

    df = pd.read_csv(file_path)
    Xs = []
    for attribute in attributes:
        X = pd.get_dummies(df[attribute], dtype=int).to_numpy()
        Xs.append(X)
    return np.hstack(Xs)
