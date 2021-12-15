import matplotlib.pyplot as plt
import numpy as np
import csv
import os

x_0 = list()
y_0 = list()
with open(os.path.dirname(os.path.realpath(__file__)) + '/input1.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    for row in spamreader:
        x_0 = np.append(x_0, float(str(row[0]).replace(",",".")))
        y_0 = np.append(y_0, float(str(row[1]).replace(",",".")))

(_, plts) = plt.subplots()
plt1 = plts.twinx()

p1, = plts.plot(x_0, y_0, "r-", label="Экспериментальные данные")
p2, = plts.plot(x_0, np.array(x_0)*659.442476448413/5.76, "b-", label="Аппроксимация данные")
plts.set_xlabel("F, н")
plts.set_ylabel("f^2, гц^2")
plts.set_title("Частота колебаний от силы натяжения")

lines = [p1, p2]

plts.legend(lines, [l.get_label()
                    for l in lines])

plt.show()