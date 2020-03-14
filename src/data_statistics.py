import numpy as np
import pandas as pd
from data_extractor import get_data
from operator import itemgetter
import matplotlib.pyplot as plt

dict_ = {}

# kopieret fra data_extractor.py
# np.array for noegletal

    #Hent navne paa kommuner
with open("attrs_min.out", encoding="utf-8") as p:
    attrs = p.read().split("\n")
    attrs = {
        "kommuner": attrs[0].split(";"),
        "aarstal": [str(x) for x in range(2007, 2019)],
        "noegletal": attrs[2].split(";"),
    }
attributes = [
		"anmeldte tyverier/indbrud pr. 1.000 indb.",
		"grundværdier pr. indb.", "beskatningsgrundlag pr. indb.",
		"udg. (brutto) til dagtilbud pr. indb.", "andel 25-64-årige uden erhvervsuddannelse",
		"andel 25-64-årige med videregående uddannelse", "udg. til folkeskoleområdet pr. indb.",
		"statsborgere fra ikke-vestlige lande pr. 10.000 indb.", "udg. til aktivering pr. 17-64/66-årig"
]
# Dictionary with key metrics as keys and dataframes as values

for noegletal in attributes:
    data = get_data(aarstal=attrs["aarstal"], noegletal=[noegletal], use_min=False)[:, :, 0].T
    dict_[noegletal] = pd.DataFrame(data.ravel())

# Attribute dictionary

attr_short = ["AT","GV","BG","DT","EU","VU","FO","IV","ATK"]

attr_dict = (dict(zip(attributes,attr_short)))

"""
def tolatex(d: np.ndarray):
    s = []
    for row in d:
        s.append("&".join([("%.2f"%x).replace(".", ",") for x in row]))
    return "\\\\\n".join(s)
"""

# Summary statistics and boxplots

summary = []

plot_values = []

for attr in attributes:
    print(attr_dict[attr])
    print(dict_[attr].describe())
    summary.append(dict_[attr].describe())
    plot_values.append(dict_[attr].values)

plot_values = np.array([np.reshape(plot_values,(9,1176))]) 

plot_values = np.squeeze(plot_values)

fig, ax = plt.subplots(nrows=3, ncols=3)

a = 0
for row in ax:
    for col in row:
        col.boxplot(plot_values[a])
        col.set_title(attr_dict[attributes[a]])
        a += 1

plt.show()
