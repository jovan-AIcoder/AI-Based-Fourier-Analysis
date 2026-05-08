# AI-Based Fourier Analysis (aifourier)

> *“Machines can learn Fourier analysis.”*

A Python library that approximates Fourier decomposition using a sinusoidal neural network.

Instead of explicitly computing Fourier integrals, this library **learns** the frequency components of a signal through optimization.

---

## ✨ Features

* 🔊 Analyze audio signals (`.wav`, `.mp3`, `.flac`, `.ogg`)
* 🧠 Neural network with sinusoidal activation
* 📊 Extract:

  * Angular frequency
  * Phase shift
  * Amplitude
* ⚡ Simple one-line API
* 📁 Output as Pandas DataFrame

---

## 📦 Installation

```bash
pip install aifourier
```

---

## 🚀 Usage

```python
import aifourier as aif

df = aif.analyze("audio.mp3", max_modes=128, epochs=300)

print(df.head())
```

---

## 📊 Output

The result is a DataFrame containing:

| Column      | Description                        |
| ----------- | ---------------------------------- |
| Frequencies | Learned frequencies (Hz)    |
| Phase shift | Phase of each component            |
| Amplitudes  | Contribution strength of each mode |

---

## 🧠 How It Works

The signal is approximated as:

y(t) ≈ Σ Aᵢ sin(ωᵢ t + φᵢ)

Where:

* Aᵢ = amplitude
* ωᵢ = angular frequency
* φᵢ = phase

These parameters are learned by a neural network instead of computed analytically.

---

## ⚙️ Parameters

```python
aif.analyze(audio_path, max_modes=128, epochs=64,use_phase_shift=True)
```

* `audio_path` : Path to audio file
* `max_modes`  : Number of sinusoidal components
* `epochs`     : Training iterations (higher = better approximation)
* `use_phase_shift` : If this is set to `False`, all phase shifts are set to zero.

---

## 📁 Example

See the `examples/` folder for a complete demo:

```bash
cd examples
python example.py
```

This will:

* Analyze `bird.mp3`
* Generate frequency components
* Save results
* Plot the spectrum

---

## ⚖️ Comparison with FFT

| Method    | Approach                    |
| --------- | --------------------------- |
| FFT       | Analytical, deterministic   |
| aifourier | Learning-based, approximate |

This project explores whether neural networks can **discover Fourier structure from data**.

---

## 🚧 Limitations

* Approximation quality depends on training
* Slower than FFT
* Results may vary between runs

---

## 💡 Future Ideas

* Signal reconstruction from learned parameters
* FFT comparison mode
* Real-time signal analysis (oscilloscope / radio)
* Complex-valued extensions

---

## 👤 Author

Jovan

---

## 📜 License

MIT License

---

> *“What Fourier derives analytically, neural networks can approximate through learning.”*
