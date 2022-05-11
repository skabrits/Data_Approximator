import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PMcalcl.main import un, mean_un
from scipyf import DataCalc

w = list()
h1 = list()
h2 = list()

with open(os.path.dirname(os.path.realpath(__file__)) + '/l32.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    for row in spamreader:
        w = np.append(w, float(str(row[0]).replace(",",".")))
        h1 = np.append(h1, float(str(row[1]).replace(",",".")))
        h2 = np.append(h2, float(str(row[2]).replace(",",".")))

linap1 = DataCalc(func="w=a*v+b", v=h1[:14], w=w[:14], koef_names="a b", x_label="H2 volume, мл", y_label="W, вт*с", title="U = 7,5 V", mode=DataCalc.mode.approximation)
linap2 = DataCalc(func="w=a*v+b", v=h1[15:], w=w[15:], koef_names="a b", x_label="H2 volume, мл", y_label="W, вт*с", title="U = 5 V", mode=DataCalc.mode.approximation)
linap3 = DataCalc(func="F=(W*R*T/(2*U*(rho*g*(h1-h2)+P0)*V))", P0 = un(779.5*133.322, 0.5*133.322), V=np.array([un(i/1000000, 0.0005/1000000) for i in h1[1:14]]), W=np.array([un(i, 0.001) for i in w[1:14]]), R = 8.314, T = un(273.15+23.5,0.0025), U = 7.5, rho = 1100, g = 9.81, h2 = np.array([un(i/100, 0.0005/100) for i in h2[1:14]]), h1 = np.array([un(i/100, 0.0005/100) for i in h1[1:14]]), koef_names="", mode=DataCalc.mode.calculator)
linap4 = DataCalc(func="F=(W*R*T/(2*U*(rho*g*(h1-h2)+P0)*V))", P0 = un(779.5*133.322, 0.5*133.322), V=np.array([un(i/1000000, 0.0005/1000000) for i in h1[16:]]), W=np.array([un(i, 0.001) for i in w[16:]]), R = 8.314, T = un(273.15+23.5,0.0025), U = 5, rho = 1100, g = 9.81, h2 = np.array([un(i/100, 0.0005/100) for i in h2[16:]]), h1 = np.array([un(i/100, 0.0005/100) for i in h1[16:]]), koef_names="", mode=DataCalc.mode.calculator)

# print(str(linap.approximate()) + "\n\n")
# print(str(linap.get_error()) + "\n\n")
# print(str(linap.get_koefs()) + "\n\n")

linap1.build_graph()
linap2.build_graph()
print(str(mean_un(np.append(linap3.calculate(), linap4.calculate()))) + "\n\n")
print(str(mean_un(linap3.calculate())) + "\n\n")
print(str(mean_un(linap4.calculate())) + "\n\n")