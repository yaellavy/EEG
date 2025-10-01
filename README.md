# EEG Experiment and Analysis

This repository contains Python scripts for **EEG-based experiments**, including:

- Presenting cognitive tasks to participants while recording EEG signals  
- Acquiring EEG data in real-time  
- Preprocessing and cleaning raw EEG recordings  
- Time-domain and frequency-domain analysis (averaging, FFT, STFT, binning)  

The code is designed for research use in neuroscience, cognitive science, and neuroimaging.

---

## Repository Structure

- **EEGexperiment.py** – Task presentation and synchronization with EEG recording  
- **main.py** – Example pipeline for running an experiment and recording data  
- **cleaning_data.py** – Scripts for cleaning and preprocessing EEG recordings  
- **original_data.py** – Utilities for handling raw EEG files  
- **FFT_concatenation.py** – Frequency-domain analysis using concatenated FFT  
- **STFT.py** – Time-frequency analysis using short-time Fourier transform  
- **averaging_time.py** – Time-domain averaging across trials  
- **averaging_frequency.py** – Frequency-domain averaging across trials  
- **averaging_time_for_STFT.py** – Averaging adapted for STFT output  
- **bining.py / main_bining_all.py** – Binning EEG data for grouped analysis  
- **main_averaging_frequency.py** – Example script for frequency averaging workflow  

 
