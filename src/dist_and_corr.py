from os import chdir
from os.path import dirname, realpath
chdir(realpath(dirname(__file__)))

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from data_extractor import get_data

def qq_plots(attrs: list or tuple):
	# Makes qq plots for given attributes
	plt.figure(figsize=(20, 20))
	aarstal = [str(x) for x in range(2007, 2020)]
	for i, attr in enumerate(attrs):
		data = get_data(aarstal=aarstal, noegletal=[attr], use_min=False)[:, :-1, 0]
		# The municipals are combined by attribute, as all observation are adjusted for population size
		data = np.sort(data.reshape(data.size))
		# Removes nans
		data = data[~np.isnan(data)]
		# Calculates theoretical quantiles
		tq = norm.ppf((np.arange(1, data.size+1) - 1/2) / data.size)
		# Correlation
		kor = np.corrcoef(tq, data, ddof=1)[0, 1]

		plt.subplot(330+i+1)
		plt.title(attr.capitalize() + "\nEmpirical correlation: %.4f" % kor)
		plt.plot(tq, data, ".")
		plt.xlabel("Theorical quantile")
		plt.ylabel("Observed value")

		print("Færdig med %s" % attr)
	plt.savefig("../latex/Billeder/qq-plots.png")

def print_corr_mat(a):
	for b in a:
		print(" ".join(["%.2f" % x for x in b]))

def corrs_and_plots(attrs: list or tuple, *attr_indices):
	aarstal = [str(x) for x in range(2007, 2020)]
	data = get_data(aarstal=aarstal, noegletal=[attrs[i] for i in [0, *attr_indices]])
	
	# shifts the first attribute five years
	newdata = []
	newdata.append(data[:, 5:-1, 0])
	newdata[-1] = newdata[-1]
	for i in range(1, data.shape[2]):
		newdata.append(data[:, :-6, i])
		newdata[-1] = newdata[-1]
	for i in range(len(newdata)):
		newdata[i] = np.ravel(newdata[i])
	data = np.array(newdata)
	
	# nans are set to mean of their respective attributes
	for i in range(data.shape[0]):
		data[i, np.isnan(data[i])] = data[i, ~np.isnan(data[i])].mean()
	corrs = np.corrcoef(data, ddof=1)
	print("Korrelationer")
	print_corr_mat(corrs)
	with open("corr.out", "w", encoding="utf-8") as out:
		lines = [";".join([x for x in attrs])]
		for i in range(corrs.shape[0]):
			lines.append(" ".join(["%.3f" % x for x in corrs[i]]))
		out.write("\n".join(lines))

	# Plotting certain variables against each other
	plt.figure(figsize=(20, 5))
	for j, i in enumerate(attr_indices):
		plt.subplot(1, len(attr_indices), j+1)
		plt.scatter(data[0], data[j+1], 4)
		plt.title(
			attrs[0].capitalize() + " after five years\n"
			+ "and " + attrs[i].capitalize()
			+ "\nEmpircal correlation: %.3f" % corrs[0, j+1]			
		)
		plt.xlabel(attrs[0].capitalize() + " after five years")
		plt.ylabel(attrs[i].capitalize())
	plt.savefig("../latex/Billeder/corrs.png")

if __name__ == "__main__":
	attrs = (
		"anmeldte tyverier/indbrud pr. 1.000 indb.",
		"grundværdier pr. indb.", "beskatningsgrundlag pr. indb.",
		"udg. (brutto) til dagtilbud pr. indb.", "andel 25-64-årige uden erhvervsuddannelse",
		"andel 25-64-årige med videregående uddannelse", "udg. til folkeskoleområdet pr. indb.",
		"statsborgere fra ikke-vestlige lande pr. 10.000 indb.", "udg. til aktivering pr. 17-64/66-årig"
	)
	print("Making qq-plots...")
	qq_plots(attrs)
	print("Calculating correlations...")
	corrs_and_plots(attrs, 3, 4, -2)
	

