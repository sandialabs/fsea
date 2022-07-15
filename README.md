# fsea
The Fan Speed Extraction Algorithm (FSEA) was developed by Sandia National Laboratories' Multi-Informatics for Nuclear Operations Scenarios (MINOS) Venture team to determine the rotational rate of gearbox-equipped fans from infrasound measurements.  The algorithm is based on identifying the blade passing frequencies (BPFs), their harmonics, as well as the motor frequencies (MFs) for each fan in operation.  Using the algorithm, these frequencies can be automatically identified in the infrasound waveform’s short-time Fourier transform spectrogram. Attribution is aided by a pair of filters that rely on the unique spectral and temporal characteristics of fan operation, as well as the intrinsic frequency ratios of the BPF harmonics and the BPF/MF signals.

## Code Details
The code provided here includes:
- **fsea.py**:  The algorithm itself, which ingests acoustic waveform data and generates the most likely fan speed values as well as associated visualizations.

- **config.py**:  A configuration file to provide user-defined parameters to the algorithm.

- **WACO_Station_Acoustic_Data.parquet**:  An example dataset that contains an acoustic waveform collected from 13:55 to 18:05 UTC on 2018-09-16 at the Oak Ridge National Laboratory (ORNL) High Flux Isotope Reactor (HFIR) by seismo-acoustic station WACO operated by ORNL (see publication for further details).

- **Tachometer_Ground_Truth.parquet**:  The true speeds over time of the HFIR's four cooling tower fans during the acoustic measurement period. 

## Publication
A full description of the design principles behind the software and its application to two different nuclear research reactor cooling towers will be provided in the following publication:

**Eaton, S. W.; Cárdenas, E. S.; Hix, J. D.; Johnson, J. T.; Watson, S. M.; Chichester, D. L.; Garcés, M. A.; Magaña-Zook, S. A.; Maceira, M.; Marcillo, O. E.; Chai, C.; Reichardt, T. A. "An Algorithmic Approach to Predicting Mechanical Draft Cooling Tower Fan Speeds from Infrasound Signals" Applied Acoustics (submitted).**

## Operation & Expected Results
The configuration file already provides the necessary parameters to generate the following output (below).  Experiment with parameters controlling the short-time Fourier transform (samples_per_second, window, overlap) and the horizontal filter (total_filter_height, mid_filter_height) to generate different results.  Control the plotting range in time (start, stop), frequency (freq_lims), and sound pressure level (spl_lims) by adjusting the corresponding parameters.

<p align="center">
  <img src="https://user-images.githubusercontent.com/108030273/179288385-28907508-535b-4a7e-8a2c-412ec4afe80a.png" alt="Figure 1"/>
</p>
<b>Figure 1.</b>  A short-time Fourier transform of the acoustic waveform data.

<p align="center">
  <img src="https://user-images.githubusercontent.com/108030273/179288433-41af891b-fe1e-4e4a-a668-a9b5237658f0.png" alt="Figure 2"/>
</p>
<b>Figure 2.</b>  The same short-time Fourier transform after the identification of pixels most likely to correspond to the BPFs of operational fans (red highlight).  

<p align="center">
  <img src="https://user-images.githubusercontent.com/108030273/179288468-07a6e3e8-8b0a-49cb-b1a6-3adfa10c4365.png" alt="Figure 3"/>
</p>
<b>Figure 3.</b>  A comparison of the total active cooling capacity calculated from the identified BPFs and as measured directly by installed tachometers. 

## Current Authors
Samuel Eaton

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Copyright
Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

## Acknowledgements
This software was developed with funds from Defense Nuclear Nonproliferation Research and Development (DNN R&D).
