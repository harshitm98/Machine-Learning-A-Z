# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 18:46:25 2018

@author: Harshit Maheshwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Takecare of the missing data
