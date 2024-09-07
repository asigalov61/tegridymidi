#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	      Tegridy Plots Python Module (TPLOTS)
#	      Version 1.0
#
#	      Project Los Angeles
#
#	      Tegridy Code 2024
#
#       https://github.com/asigalov61/tegridy-tools
#
#
################################################################################
#
#       Copyright 2024 Project Los Angeles / Tegridy Code
#
#       Licensed under the Apache License, Version 2.0 (the "License");
#       you may not use this file except in compliance with the License.
#       You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
#       Unless required by applicable law or agreed to in writing, software
#       distributed under the License is distributed on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#       See the License for the specific language governing permissions and
#       limitations under the License.
#
################################################################################
################################################################################
#
# Critical dependencies
#
# !pip install numpy
# !pip install scipy
# !pip install matplotlib
# !pip install networkx
# !pip3 install scikit-learn
#
################################################################################
#
# Future critical dependencies
#
# !pip install umap-learn
# !pip install alphashape
#
################################################################################
'''

################################################################################
# Modules imports
################################################################################

import os
from collections import Counter
from itertools import groupby

import numpy as np

import networkx as nx

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.ndimage import zoom
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import zscore

import matplotlib.pyplot as plt
from PIL import Image

################################################################################
# Constants
################################################################################

################################################################################




################################################################################





################################################################################


################################################################################






################################################################################




  
################################################################################
# [WIP] Future dev functions
################################################################################

'''
import umap

def reduce_dimensionality_umap(list_of_values,
                               n_comp=2,
                               n_neighbors=15,
                               ):

  """
  Reduces the dimensionality of the values using UMAP.
  """

  vals = np.array(list_of_values)

  umap_reducer = umap.UMAP(n_components=n_comp,
                           n_neighbors=n_neighbors,
                           n_epochs=5000,
                           verbose=True
                           )

  reduced_vals = umap_reducer.fit_transform(vals)

  return reduced_vals.tolist()
'''

################################################################################

'''
import alphashape
from shapely.geometry import Point
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.stats import zscore

#===============================================================================

coordinates = points

dist_matrix = minkowski_distance_matrix(coordinates, p=3)  # You can change the value of p as needed

# Centering matrix
n = dist_matrix.shape[0]
H = np.eye(n) - np.ones((n, n)) / n

# Apply double centering
B = -0.5 * H @ dist_matrix**2 @ H

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(B)

# Sort eigenvalues and eigenvectors
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Select the top 2 eigenvectors
X_transformed = eigvecs[:, :2] * np.sqrt(eigvals[:2])

#===============================================================================

src_points = X_transformed
src_values = np.array([[p[1]] for p in points]) #np.random.rand(X_transformed.shape[0])

#===============================================================================

# Normalize the points to the range [0, 1]
scaler = MinMaxScaler()
points_normalized = scaler.fit_transform(src_points)

values_normalized = custom_normalize(src_values)

# Remove outliers based on z-score
z_scores = np.abs(zscore(points_normalized, axis=0))
filtered_points = points_normalized[(z_scores < 3).all(axis=1)]
filtered_values = values_normalized[(z_scores < 3).all(axis=1)]

# Compute the concave hull (alpha shape)
alpha = 8  # Adjust alpha as needed
hull = alphashape.alphashape(filtered_points, alpha)

# Create a triangulation
tri = Triangulation(filtered_points[:, 0], filtered_points[:, 1])

# Interpolate the values on the triangulation
interpolator = LinearTriInterpolator(tri, filtered_values[:, 0])
xi, yi = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
zi = interpolator(xi, yi)

# Mask out points outside the concave hull
mask = np.array([hull.contains(Point(x, y)) for x, y in zip(xi.flatten(), yi.flatten())])
zi = np.ma.array(zi, mask=~mask.reshape(zi.shape))

# Plot the filled contour based on the interpolated values
plt.contourf(xi, yi, zi, levels=50, cmap='viridis')

# Plot the original points
#plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c=filtered_values, edgecolors='k')

plt.title('Filled Contour Plot with Original Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Value')
plt.show()
'''

################################################################################

__all__ = [name for name in globals() if not name.startswith('_')]

################################################################################
#
# This is the end of TPLOTS Python modules
#
################################################################################
