import pandas as pd
import edfio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from shrec.models import RecurrenceManifold
from shrec.utils import standardize_ts
import plotly.graph_objects as go
import random
import tensorflow as tf



def plot_3d_trajectory(coords, seizure_start, seizure_end, sampling_rate, title="3D Trajectory Plot"):
    seizure_index_start = int(seizure_start * sampling_rate)
    seizure_index_end   = int(seizure_end * sampling_rate)


    fig = go.Figure()

    # Vor Seizure
    fig.add_trace(go.Scatter3d(
        x=coords[:seizure_index_start, 0],
        y=coords[:seizure_index_start, 1],
        z=coords[:seizure_index_start, 2],
        mode='lines',
        line=dict(color='blue'),
        name='Pre-Seizure'
    ))

    # Seizure-Bereich farblich markieren (z.B. grün)
    fig.add_trace(go.Scatter3d(
        x=coords[seizure_index_start:seizure_index_end, 0],
        y=coords[seizure_index_start:seizure_index_end, 1],
        z=coords[seizure_index_start:seizure_index_end, 2],
        mode='lines',
        line=dict(color='limegreen'),
        name='Seizure'
    ))

    # Nach Seizure
    fig.add_trace(go.Scatter3d(
        x=coords[seizure_index_end:, 0],
        y=coords[seizure_index_end:, 1],
        z=coords[seizure_index_end:, 2],
        mode='lines',
        line=dict(color='black'),
        name='Post-Seizure'
    ))

    # Marker für Seizure Onset
    fig.add_trace(go.Scatter3d(
        x=[coords[seizure_index_start, 0]],
        y=[coords[seizure_index_start, 1]],
        z=[coords[seizure_index_start, 2]],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        name='Seizure Onset'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600,
        legend=dict(
            x=1,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )

    fig.show()


def extract_eeg_data(filepath):
    X_eeg = edfio.read_edf(filepath)
    X_data = None

    for sig in X_eeg.labels:
        if not "p " in sig.lower():
            if X_data is None:
                X_data = X_eeg.get_signal(sig).data
            else:
                X_data = np.vstack((X_data, X_eeg.get_signal(sig).data))
    return X_data


def plot_eeg(X_eeg):
    plt.figure(figsize=(15, 8))
    for i in range(X_eeg.shape[0]):
        plt.plot(-i + standardize_ts(X_eeg[i, :]), 'k', linewidth=1, alpha=0.5)
    plt.title("EEG Signals")
    plt.xlabel("Time (samples)")
    plt.ylabel("Channels (offset for visibility)")
    plt.plot()


def calc_and_plot_psd(X_eeg, y_reconstructed, sampling_rate: int = 256, channel: int = 0):
    f, Pxx_den = welch(standardize_ts(y_reconstructed), sampling_rate, nperseg=1024)
    f2, Pxx_den2 = welch(standardize_ts(X_eeg[channel, :]), sampling_rate, nperseg=1024)
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, Pxx_den, label='Reconstructed Driver', linewidth=1)
    plt.semilogy(f2, Pxx_den2, label=f'Original EEG Channel {channel}', linewidth=1)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.legend()
    plt.title('Power Spectral Density Comparison')
    plt.show()


def set_random_seeds(seed_value: int = 42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)