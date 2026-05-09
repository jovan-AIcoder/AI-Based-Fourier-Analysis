from setuptools import setup, find_packages

setup(
    name="aifourier",
    version="0.4.5",
    description="AI-based Fourier Analysis using sinusoidal neural networks",
    author="Jovan",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "tensorflow",
        "librosa"
    ],
    python_requires=">=3.8",
)