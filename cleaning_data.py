import os
import tkinter as tk
import pandas as pd
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, fft, fftfreq
from tkinter import filedialog, messagebox
from scipy import signal

def cleaning_data(coeffs):
    if coeffs.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {coeffs.ndim}D")

    cleaned = np.empty_like(coeffs, dtype=np.float64)
    n_rows, n_cols = coeffs.shape

    for i in range(n_rows):
        row = coeffs[i, :].copy()  # Get row as 1D array

        row = np.where(np.isinf(row), np.nan, row)

        if np.all(np.isnan(row)):
            cleaned[i, :] = np.zeros(n_cols, dtype=np.float64)
            continue

        interpolated = pd.Series(row).interpolate(
            method='linear',
            limit_direction='both'
        ).bfill().ffill()

        cleaned[i, :] = interpolated.values

    return cleaned