"""
Utility Functions

Author: Muhammad Azeem Khan
University: Sukkur IBA University

This file generates a synthetic dataset
for binary classification.
"""

import numpy as np


def generate_data(samples=1000):
    """
    Generate random 2D points and classify
    whether they are inside or outside a circle.
    """

    # Generate random points
    X = np.random.uniform(-1, 1, (samples, 2))

    # Generate labels
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5).astype(int)

    return X, y