from setuptools import setup, find_packages
import os

# Install fluidsynth using apt
os.system('apt-get update && apt-get install -y fluidsynth')

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tegridymidi",
    version="24.9.7",
    description="Tools for reading, writing, and manipulating MIDIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alex Lev",
    author_email="alexlev61@proton.me",
    url="https://github.com/asigalov61/tegridymidi",  # Add the URL of your project
    project_urls={
        "Examples": "https://github.com/asigalov61/tegridymidi/tegridymidi/example",
        "Issues": "https://github.com/asigalov61/tegridymidi/issues",
        "Documentation": "https://github.com/asigalov61/tegridymidi/docs",
        "Discussions": "https://github.com/asigalov61/tegridymidi/discussions",
        "Source Code": "https://github.com/asigalov61/tegridymidi",
    },
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,
    package_data={
        'tegridymidi': ['sample_midis/*', 'examples/*'],
    },
    install_requires=[
        'tqdm',
        'pillow',
        'numpy',
        'matplotlib',
        'scipy',
        'networkx',
        'scikit-learn',
    ],
    extras_require={
        'cuda': [
            'torch',
            'einops',
            'cupy-cuda12x',
            'torch-summary',
        ],
    },
    keywords=['MIDI', 'tegridy', 'tools'],  # Add your keywords here
    python_requires='>=3.6',  # Specify the Python version
    license='Apache Software License 2.0',  # Specify the license
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',        
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
)
