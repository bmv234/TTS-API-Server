from setuptools import setup, find_packages

setup(
    name="f5_tts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'numpy',
        'soundfile'
    ]
)