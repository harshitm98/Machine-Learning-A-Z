# -*- coding: utf-8 -*-
"""
Created on Sun May 20 00:33:54 2018

@author: Harshit Maheshwari
"""

# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
# Training Apriori on the dataset
from apyori import apriori
support = 3*7/7500 # An item that is bought 3 times a week
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.20, min_lift = 3, min_length = 2)
results = list(rules)

def display_top_products(results, n_products=5):
    print("Support\tConf.\tLift\tProducts")
    for result in results[:n_products]:
        support = round(100 * result.support, 2)
        confidence = round(result.ordered_statistics[0].confidence, 2)
        lift = round(result.ordered_statistics[0].lift, 2)
        products = " + ".join(list(result.items))
        print("{0}%\t{1}\t{2}\t{3}".format(support, confidence, lift, products))
display_top_products(results, 30)