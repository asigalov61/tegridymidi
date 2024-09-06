from setuptools import setup, find_packages
import os

# Install fluidsynth using apt
os.system('apt-get update && apt-get install -y fluidsynth')

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tegridymidi",
    version="0.2.1",
    description="Tegridy MIDI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alex Lev",
    author_email="alexlev61@proton.me",
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,
    package_data={
        'tegridymidi': ['sample_midis/*'],
    },
    install_requires=[
        'tqdm',
        'numpy',
        'matplotlib',
        'scipy',
        'networkx',
        'scikit-learn',
    ],
    extras_require={
        'cuda': [
            'einops',
            'torch',
            'cupy-cuda12x',
            'torch-summary',
        ],
)
