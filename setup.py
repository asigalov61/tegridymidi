from setuptools import setup, find_packages
import os

# Install fluidsynth using apt
os.system('apt-get update && apt-get install -y fluidsynth')

setup(
    name="tegridymidi",
    version="0.1.0",
    description="Tegridy MIDI",
    author="Alex Lev",
    author_email="alexlev61@proton.me",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
)
