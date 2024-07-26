# IMPORT MODULES
# Standard library
from typing import Callable, Dict, Optional

# Third party
import numpy as np
import plotly.graph_objects as go
from sklearn.utils import Bunch

def get_true_drc(
    data: Bunch,
    num_integration_samples: int = 65,
) -> np.ndarray:
    """
    Calculates the true dose-response curve (DRC) for the given data.

    Parameters:
    data (Bunch): The dataset containing the dose-response information.
    num_integration_samples (int, optional): The number of samples to use for numerical integration. Defaults to 65.

    Returns:
    np.ndarray: The calculated true DRC as a numpy array.
    """
    # Save observation according to quantile
    x = np.quantile(data.x[data.test_ids], q=0.5, axis=0).reshape(1, -1)

    # Multiply observation
    x = x.repeat(num_integration_samples, 0)
    
    # Get doses
    d = np.linspace(0, 1, num_integration_samples)
    
    # Define treatment
    t = np.zeros(num_integration_samples)
    
    return data.ground_truth(x, d, t)

def predict_drc(
    data: Bunch,
    model: Callable,
    num_integration_samples: int = 65,
):
    """
    Predicts the dose-response curve (DRC) for the given data using the provided model.

    Parameters:
    data (Bunch): The dataset containing the dose-response information.
    model (Callable): The machine learning model to be used for prediction.
    num_integration_samples (int, optional): The number of samples to use for numerical integration. Defaults to 65.

    Returns:
    np.ndarray: The predicted DRC as a numpy array.
    """
    # Save observation according to quantile
    x = np.quantile(data.x[data.test_ids], q=0.5, axis=0).reshape(1, -1)

    # Multiply observation
    x = x.repeat(num_integration_samples, 0)
    
    # Get treatments
    d = np.linspace(0, 1, num_integration_samples)
    
    # Define treatment
    t = np.zeros(num_integration_samples)
    
    return model.predict(x, d, t)

def plot_drc(
    curves: Dict,
    doses: Optional[np.ndarray] = None,
    num_integration_samples: int = 65,
    description: str = "",
) -> go.Figure:
    # Get treatments for x axis
    d = np.linspace(0, 1, num_integration_samples)

    # Generate figure
    figure = go.Figure()
    for key in curves.keys():
        figure.add_trace(go.Scatter(x=d, y=curves[key], line_shape="linear", name=key))
    
    # Generate histogram-style plot of dose distribution
    if doses is not None:
        bins = np.linspace(0, 1, num_integration_samples + 1)
        bars = np.histogram(doses, bins)[0]

        # Get max value of curves
        max_curves = max([max(curve) for curve in curves.values()])

        # Adjust height
        bars = ((bars) / np.max(bars)) * max_curves

        figure.add_trace(
            go.Bar(
                x=d,
                y=bars,
                name="Relative density",
                opacity=0.5,
                marker_color="lightblue",
            )
        )
        
    # Update layout
    figure.update_layout(
        barmode="group",
        bargap=0.0,
        bargroupgap=0.0,
        paper_bgcolor="white",
        plot_bgcolor="white",
        shapes=[
            go.layout.Shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.0,
                y0=0.0,
                x1=1.0,
                y1=1.0,
                line={"width": 1, "color": "black"},
            )
        ],
        title=("Summary of dose-response curves: " + description),
        xaxis_title="Dosage",
        yaxis_title="Probability",
        legend_title="Models",
    )

    return figure