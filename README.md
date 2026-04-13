# 🐦 Neural Spectral Analyzer (Fourier-inspired Audio Decomposition)

## 📌 Overview

This project explores a novel approach to analyzing audio signals using a minimalist neural network architecture inspired by the Fourier series.

Instead of applying classical Fast Fourier Transform (FFT), this method leverages a neural network with sinusoidal activation to **learn the underlying frequency components of an audio signal directly from data**.

The result is an interpretable decomposition of sound into:

* Frequencies (angular)
* Phase shifts
* (implicitly) amplitudes

---

## 🧠 Core Idea

Given an audio signal (y(t)), the model approximates it using:

$$y(t) \approx \sum_{i=1}^{N} A_i \sin(\omega_i t + \phi_i)$$

Where:

* ($\omega_i$) → learned frequencies (from weights)
* ($\phi_i$) → learned phase shifts (from biases)
* ($A_i$) → learned amplitudes (from output layer)

This transforms the neural network into a **data-driven Fourier-like decomposition system**.

---

## 📁 Project Structure

```
.
├── bird.mp3           # Input audio file (example: bird sound)
├── analyzer.py        # Main script for signal analysis
├── frequencies.xlsx   # Output file containing extracted parameters
├── audio_signal.png   # Audio signal visualization
```

---

## ⚙️ How It Works

### 1. Audio Sampling

* The `.mp3` file is loaded using `librosa`
* The signal is converted into:

  * Time array (microseconds)
  * Sound signal values

### 2. Neural Approximation

A simple neural network is constructed:

* **Input:** time ($t$)
* **Hidden layer:** sinusoidal activation
* **Output layer:** linear combination

This effectively creates a sum of sinusoidal basis functions.

### 3. Training

* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Goal: Fit the waveform as closely as possible

### 4. Parameter Extraction

After training:

* Weights → frequencies ($\omega$)
* Biases → phase shifts ($\phi$)

These are exported into an Excel file.

---

## 📊 Output

The resulting `frequencies.xlsx` contains:

| Index    | Frequencies (ω) | Phase Shift (φ) | Amplitudes (A) |
| -------- | --------------- | --------------- | -------------- |
| Neuron_0 | ...             | ...             | ...            |
| Neuron_1 | ...             | ...             | ...            |
| ...      | ...             | ...             | ...            |

Each row corresponds to one sinusoidal component learned by the model.

---

## 🚀 Usage

1. Install dependencies:

```bash
pip install numpy pandas matplotlib librosa tensorflow openpyxl
```

2. Run the analyzer:

```bash
python analyzer.py
```

3. Output:

* `frequencies.xlsx` will be generated and opened automatically

---

## ⚠️ Notes

* Frequencies are in **angular form (rad/s)**
  Convert to Hz using:
  $f = \frac{\omega}{2\pi}$

* The model may produce:

  * Redundant frequencies
  * Low-amplitude components
  * Non-orthogonal basis functions

* This is **not a replacement for FFT**, but an alternative perspective:

  * Data-driven
  * Interpretable
  * Flexible

---

## 💡 Future Improvements

* Extract amplitudes from the output layer
* Apply regularization for sparse frequency selection
* Compare results with FFT
* Use windowing for better temporal resolution
* Extend to real-time audio analysis

---

## 🎯 Motivation

This project aims to bridge:

* Signal Processing
* Machine Learning
* Mathematical Physics

By treating neural networks not just as predictors, but as **tools for discovering structure in physical signals**.

---

## ✨ Closing Thought

> “What if a neural network could listen to sound not as data,
> but as a composition of pure mathematical waves?”

This project is a small step toward that idea.

## License
This project is under MIT license.

Created by Jovan, 2026