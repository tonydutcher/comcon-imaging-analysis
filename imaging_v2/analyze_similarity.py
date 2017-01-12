import os
import numpy as numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

dataFile       = '/Users/amd5226/lewPeaCodeAMD/comcon/analysis/imaging/image_similarity.csv'
data           = pd.read_csv(dataFile, delimiter=',', header=0, index_col=0)
objects_matrix = data.iloc[:191,:191]