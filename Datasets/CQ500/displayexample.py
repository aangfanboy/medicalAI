import glob
import random
import pandas as pd
import matplotlib.pyplot as plt

from pydicom import dcmread

all_file_paths = glob.glob("all_data/*/*/*/*/*.dcm")
random_file = random.choice(all_file_paths)
pred_prob_csv = pd.read_csv("all_data/prediction_probabilities.csv")

name, Category_New, ICH, IPH, IVH, SDH, EDH, SAH, CalvarialFracture, MassEffect, MidlineShift = \
    pred_prob_csv[pred_prob_csv["name"] == random_file.split("\\")[1]].values[0]

print(f"Path: {random_file}\nName: {name}\n"
      f"Category: {Category_New}\nIntracranial hemorrhage Prob: {ICH}\n"
      f"Intraparenchymal Prob: {IPH}\nIntraventricular Prob: {IVH}\n"
      f"Subdural Prob: {SDH}\nExtradural Prob: {EDH}\n"
      f"Subarachnoid Prob: {SAH}\nCalvarial Fracture: {CalvarialFracture}\n"
      f"Mass Effect: {MassEffect}\nMidline Shift: {MidlineShift}")

ds = dcmread(random_file)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()
