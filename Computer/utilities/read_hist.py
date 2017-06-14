# Use this script to read a history file.
import pickle
import matplotlib.pyplot as plt

with open("../Self-Driving-RC-Data/training-2017-06-11-training-8/model-history.pkl", "rb") as pfile:
    hist = pickle.load(pfile)

