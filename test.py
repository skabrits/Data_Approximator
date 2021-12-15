import matplotlib.pyplot as plt
import numpy as np
import csv
import os

r_r = list()
r_p = list()
v_r = list()
v_p = list()
v_hp = list()
k_r = list()
k_p = list()
with open(os.path.dirname(os.path.realpath(__file__)) + '/input1.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    for row in spamreader:
        r_r = np.append(r_r, float(str(row[0]).replace(",",".")))
        v_r = np.append(v_r, float(str(row[1]).replace(",",".")))
        k_r = np.append(k_r, float(str(row[2]).replace(",", ".")))
        r_p = np.append(r_p, float(str(row[3]).replace(",", ".")))
        v_p = np.append(v_p, float(str(row[4]).replace(",", ".")))
        v_hp = np.append(v_hp, float(str(row[5]).replace(",", ".")))
        k_p = np.append(k_p, float(str(row[6]).replace(",", ".")))

(_, plts) = plt.subplots()
plt1 = plts.twinx()

p1, = plts.plot(r_r, v_r, "r-", label="U реостата")
p2, = plt1.plot(r_r, k_r, "m-", label="кпд реостата")
p3, = plts.plot(r_p, v_p, "b-", label="U потенциометра")
p4, = plts.plot(r_p, v_hp, "g-", label="U холостого хода")
p5, = plt1.plot(r_p, k_p, "c-", label="кпд потенциометра")
plts.set_xlabel("R реостата/потенциометра, ом")
plts.set_ylabel("напряжение, В")
plt1.set_ylabel("кпд, %")

lines = [p1, p2, p3, p4, p5]

plts.legend(lines, [l.get_label()
                    for l in lines])

plt.show()

# linap = DataCalc(func="c1+c2*x*log(x)+c3*e^x", x=x_0, y = y_0, koef_names="c1 c2 c3", mode=DataCalc.mode.approximation)