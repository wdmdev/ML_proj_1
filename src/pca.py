import os
from os.path import dirname, realpath
os.chdir(realpath(dirname(__file__)))

import json
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib import colors as mcolors
import numpy as np

from data_extractor import get_data

from time import time

def do_pca(data: np.ndarray):

	"""
	Udfører PCA for et givent datasæt og år
	:param data: Observationer nedad, variabel henad
	:return:
	"""
	
	# Laver dataene om til en datamatrix med x i første række og observationer i anden
	dat = data[:, 0, :]
	for i in range(1, data.shape[1]):
		dat = np.concatenate((dat, data[:, i, :]), axis=0)
	dat = dat.T

	# Standardiserer data
	mu = dat.mean(axis=1)
	std = np.std(dat, axis=1, ddof=1)
	x = (dat.T - mu) / std
	
	# Laver eigenvektorer og -værdier
	lambdas, V = np.linalg.eig(x.T @ x)
	sort_arr = lambdas.argsort()[::-1]
	lambdas = lambdas[sort_arr]
	V = V[:, sort_arr]

	# Projicerer ind i egenbasis
	z = V.T @ x.T

	return x.T, z, lambdas, V

def le_plot(z: np.ndarray, m: list or tuple, nyears: int):
    fs = 16
    # Gets first two principal components of z and sorts to make masking easier
    z = z[:2]
    for i in range(z.shape[0]):
    	z[i] = sort_by_municipality(z[i])
    
    # Gets municipalities
    with open("attrs.out", encoding="utf-8") as f:
    	muns = f.readline().strip().split(";")
    
    indices = [muns.index(x.lower()) for x in m if x.lower() in muns]
    # Splits data into municipalities
    mundat = []
    markers = list(MarkerStyle.markers)
    for i in reversed(indices):
    	mundat.append(z[:, i*nyears:(i+1)*nyears])
    	z = np.concatenate((z[:, :i*nyears], z[:, (i+1)*nyears:]), axis=1)
    mundat.reverse()
    
    #Plots stuff
    plt.scatter(*z, 4, c="black")
    for i, md in enumerate(mundat):
    	plt.plot(*md, markers[i+2], markersize=12, label=muns[i].capitalize() + " municipality")
    plt.title('Projection of Municipality Key Figures into PCA Space',fontsize=fs+8)
    plt.xlabel('PC 1', fontsize=fs)
    plt.ylabel('PC 2', fontsize=fs)
    plt.legend(labels=[str.capitalize(mu) + ' kommune' for mu in m], fontsize=fs)
    plt.show()

def sort_by_municipality(x: np.ndarray):
	"""
	Sorts a 1D array x such that (k1-2007, k2-2007, ..., k1-2008, ...)
	becomes (k1-2007, k1-2008, ..., k2-2007, ...)
	"""

	return x.reshape(98, x.size//98).ravel(order="F")

def __plot_pc_coefficients(V, attributeNames):
    attr_short = ["AT","GV","BG","DT","EU","VU","FO","IV","ATK"]
    fs = 16
    pcs = range(5)

    legendStrs = ['PC'+str(e+1) for e in pcs]
    c = ['r','g','b']
    bw = .2
    r = np.arange(1,len(attributeNames)+1) * (1+2*bw)

    with open('pcs_coefficients.out', 'a+') as pcs_file:
        pcs_file.write('---------- Starting new run ----------\n')
        for i in pcs:
            print(r+i*bw)
            plt.bar(r+i*bw, V[:,i], width=bw)
            pcs_file.write('Principal Component {0}: \n'.format(i+1))
            pcs_file.write('{0} \n\n'.format(V[:,i]))

    plt.xticks(r+bw, attr_short, fontsize=fs)
    plt.xlabel('Attributes', fontsize=fs)
    plt.ylabel('Component coefficients', fontsize=fs)
    plt.legend(legendStrs, fontsize=fs)
    plt.grid()
    plt.title('Municipality Key Figures: Component Coefficients', fontsize=fs+8)
    plt.show()


def __plot_attribute_coefficients(V, attributeNames):
    attr_short = ["AT","GV","BG","DT","EU","VU","FO","IV","ATK"]
    fs = 16
    i = 0
    j = 1

    # Plot attribute coefficients in principal component space
    arrow_colors = list(mcolors.BASE_COLORS.keys())[:-1]+list(mcolors.CSS4_COLORS.keys())[11:]
    arrows = []
    for att in range(V.shape[1]):
        a = plt.arrow(0, 0, V[att, i], V[att, j], head_width=0.03, color=arrow_colors[att])
        arrows.append(a)

    
    plt.xlim([-0.1, 0.1])
    plt.ylim([-0.1, 0.1])
    plt.xlabel('PC'+str(i+1), fontsize=fs)
    plt.ylabel('PC'+str(j+1), fontsize=fs)
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)),
        np.sin(np.arange(0, 2*np.pi, 0.01)))
    
    plt.title('Attribute coefficients', fontsize=fs+8)
    plt.axis('equal')
    #Create legends
    legends = []
    for i in range(len(attr_short)):
        legends.append('({0}) {1}'.format(attr_short[i], attributeNames[i]))

    plt.legend(arrows, legends, fontsize=fs, loc='upper right')
    plt.show()

def __plot_pc_variance(lambdas):
    """
    :param lambdas: The eigenvalues from SVD
    """
    fs = 16
    #computing variance explained by principal components
    rho = lambdas / lambdas.sum()

    #Threshold for variance explained by pc
    threshold = 0.9

    #Plot varianc explained
    plt.figure()
    plt.plot(range(1, len(rho)+1), rho, 'x-')
    plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components', fontsize=fs+8)
    plt.xlabel('Principal component', fontsize=fs)
    plt.ylabel('Variance explained', fontsize=fs)
    plt.legend(['Individual', 'Cumulative', 'Threshold {0}%'.format(threshold*100)], fontsize=fs)
    plt.grid()
    plt.show()


if __name__ == "__main__":

	with open("william_pca_settings.json") as f:
		config = json.load(f)

	x, z, l, V = do_pca(get_data(**config))

	#le_plot(z, ("thisted", "lyngby-taarbæk", "albertslund", "kolding"), len(config["aarstal"]))
	#__plot_pc_coefficients(V, config["noegletal"])
	__plot_attribute_coefficients(V, config["noegletal"])
	#__plot_pc_variance(l)
