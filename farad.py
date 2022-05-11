import matplotlib.pyplot as plt
import numpy as np
import csv
import os

x_0 = list()
y_0 = list()
with open(os.path.dirname(os.path.realpath(__file__)) + '/farinput.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    for row in spamreader:
        x_0 = np.append(x_0, float(str(row[0]).replace(",",".")))
        y_0 = np.append(y_0, float(str(row[1]).replace(",",".")))

linap = DataCalc(func="y=-(A)/(T1*2*D)*(C+A/B*D)", T1 = un(273.15+20.1,0.1), A=un(0.006,0.002), B = un(-0.003,0.0009),C=un(6.2,0.2),D=un(-1.2,0.2), mode=DataCalc.mode.calculator)
# print(str(linap.approximate()) + "\n\n")
# print(str(linap.get_error()) + "\n\n")
# print(str(linap.get_koefs()) + "\n\n")
print(str(linap.calculate()) + "\n\n")
# linap.build_graph()