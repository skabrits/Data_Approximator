import os

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import math
import csv
from PMcalcl.main import un
import re


class DataCalc:

    class error_funcs:
        squared_error= "squared_error"
        linear_error = "linear_error"
        errors = [linear_error,squared_error]

    rep_dict = {"^": "**", "e": "math.e", "pi": "math.pi", "log": "math.log", "cos": "math.cos", "sin": "math.sin",
                "tg": "math.tan", "ln": "math.log", "sqrt": "math.sqrt", "exp": "math.exp", " ": ""}

    # TODO: koeff num, dx and delt should be  specific for each coefficient and weight decays
    def __init__(self, func="", y_name="y", koef_names="", max_iter=10 ** 4, derivative_weights=None, borders=None,
                 generating_borders=None,
                 gen_koef_num=10, dx=10 ** -10, delt=10 ** -10, dx_weight_decay=0., weight_decay=0., rush_num =10 ** -7,
                 optim_func=error_funcs.squared_error, approximation_function_type=None, x_label="x", y_label="y", title="",
                 **kwargs):
        """
        :param func: function you want to approximate with (like in wolfram alpha)
        :param y_name: name of variable that defines function  output, live "" for automatic detection
        :param koef_names: names of function params
        :param max_iter: maximum number of iterations fo gradient descent
        :param derivative_weights: weight for gradient descent derivative for each coefficient
        :param borders: smaller and upper limits of coefficients for each coefficient
        :param generating_borders: smaller and upper limits of coefficients for each coefficient at zero iteration
        :param gen_koef_num: number of starting points for coefficients
        :param dx: function step
        :param delt: zero gap
        :param dx_weight_decay: weight decay for dx for gradient decay
        :param weight_decay: weight decay for gradient decay
        :param rush_num: penalty for slow error decrease
        :param optim_func: optimization function: linear_error or  squared_error
        :param approximation_function_type: set "linear" for precise calculations of uncertainty
        :param x_label: label for x axis of plot
        :param y_label: label for x axis of plot
        :param title: title of plot
        :param kwargs: all function variables including output of function

        This function initializes class that contains data (mesurments: parameters and output) and can execute
        operations on it (at the moment approximate data by custom function and build graph of data). It requires
        you:
        1) to pass data, you want to work with in form of several arrays, where each array is passed as named
        parameter to function
        2) to pass function you want to approximate data with in form of string (whether as in
        wolfram alpha's format or as python code). Note: names of parameters of function must match data's names. It
        should be weather equation on output ("y") variable (better option), or any equation
        3) to pass function coefficients' names (names of all "letters" in equation, which are not parameters (data)).
        They must be separated with space.

        Also if output ("y") variable name is not y you can pass its name to y_name parameter, or "" if you want program
        to detect it automatically. Other parameters are described higher.

        Example of usage:

        linap = DataCalc(func="y=(k*x^2+b)^1/2", koef_names="k b", x=[1, 3, 5], y=[4, 8, 12],
        optim_func=DataCalc.error_funcs.squared_error, max_iter=10 ** 4, weight_decay=0, derivative_weights=0.01)

        print(str(linap.approximate()) + "\n\n") #
        print(str(linap.get_error()) + "\n\n") #
        print(str(linap.get_koefs()) + "\n\n") #
        linap.build_graph() # graph:

        |              .  *
        |             .*
        |           *.
        |        *  .
        |  .  *
        |  *
        |_________________

        *** - original plot
        ... - approximated plot
        """

        self.optim_func = optim_func
        self.approximation_function_type = approximation_function_type
        self.title = title
        self.y_label = y_label
        self.x_label = x_label

        self.koef_names = list(filter(lambda x: x != "", koef_names.strip().split(" ")))
        self.rush_num = rush_num
        self.weight_decay = weight_decay
        self.dx = dx
        self.delt = delt

        if borders is None:
            borders = [[-10000, 10000] for _ in range(len(self.koef_names))]

        if type(borders[0]) in {int,float}:
            borders = [[borders[0], borders[1]] for _ in range(len(self.koef_names))]

        if generating_borders is None:
            generating_borders = [[i[0]/3*2,i[1]/3*2] for i in borders]

        if type(generating_borders[0]) in {int,float}:
            generating_borders = [[generating_borders[0], generating_borders[1]] for _ in range(len(self.koef_names))]

        self.borders = borders
        self.generating_borders = generating_borders
        self.gen_koef_num = gen_koef_num

        if derivative_weights is None:
            derivative_weights = [0.1 for i in range(len(self.koef_names))]

        if type(derivative_weights) in {int, float}:
            derivative_weights = [derivative_weights for _ in range(len(self.koef_names))]

        self.derivative_weights = derivative_weights
        self.vars = kwargs
        self.var_names = list(kwargs.keys())
        self.x = list(self.vars[k] for k in kwargs.keys())
        self.y_name = y_name
        self.max_iter = max_iter
        self.dx_weight_decay = dx_weight_decay
        self.purefunc = ""
        self.approx_val = None
        self.is_graphable = False
        self.func = self.prep_func(func)

    def functionv(self, koefs):
        total_error = 0
        ll = len(self.x[0])
        for j in range(ll):
            for i in range(len(self.var_names)):
                exec(str(self.var_names[i]) + " = " + str(self.x[i][j]))
            for i in range(len(self.koef_names)):
                exec(str(self.koef_names[i]) + " = " + str(koefs[i]))


            total_error += eval("(" + str(self.func) + ")**2") if self.optim_func == self.error_funcs.squared_error else eval("abs(" + str(self.func) + ")")
        total_error /= ll
        return total_error

    def build_graph(self):
        if len(self.var_names) > 2:
            print("Too much variables - working only in 2 d")
            return 0
        else:
            plt.plot(self.x[1 - self.var_names.index(self.y_name)], self.x[self.var_names.index(self.y_name)])
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)
            plt.title(self.title)
            if self.is_graphable and not self.approx_val is None:
                for i in range(len(self.koef_names)):
                    exec(str(self.koef_names[i]) + "=" + (str(self.approx_val[self.koef_names[i]]) if not isinstance(self.approx_val[self.koef_names[i]], un) else str(self.approx_val[self.koef_names[i]].num)))

                gvd = dict(globals().items())
                gvd.update(dict(locals().items()))
                gvd.update({'self': self})
                plt.plot(self.x[1 - self.var_names.index(self.y_name)], eval(str("[eval(self.purefunc) for " + str(self.var_names[1 - self.var_names.index(self.y_name)]) + " in self.x[1 - self.var_names.index(self.y_name)]]"), gvd))
                plt.legend(('Original Data', 'Approximated Data',),shadow=True)

            plt.show()

    def approximate(self):
        if self.func == "":
            print("No function was passed - pass function")
            return 0
        if self.koef_names == [""]:
            print("No coefficient names were passed - pass them")
            return 0

        return self.grad_descent()

    def grad_descent(self):
        func = self.functionv

        def num_deriv(k, i=0):
            dx = self.dx * (1 - self.dx_weight_decay) ** i
            return [(func([k[l] if l != j else k[l] + dx for l in range(len(k))]) - func(k)) / dx for j in
                    range(len(k))]

        def find_local_min(k):
            zero_transitions = 0
            delt = self.delt
            i = 0
            nd = num_deriv(k)
            ndo = np.copy(nd)
            while (max(map(abs, nd)) >= delt and i < self.max_iter):
                res_sum = 0
                was_trans = ((np.array(nd) * np.array(ndo)) < 0).all()
                zero_transitions += 1 if was_trans and not self.optim_func in {self.error_funcs.squared_error} else 0
                for jk in range(len(nd)):
                    ccv = 1
                    nk = k[jk] - self.derivative_weights[jk] ** ccv * nd[jk] * (1-self.weight_decay) ** i * (0.5 ** zero_transitions if zero_transitions > 0 else 1)
                    while (self.borders[jk][0] > nk or nk > self.borders[jk][1]) and ccv < 5:
                        nk = k[jk] - self.derivative_weights[jk] ** ccv * nd[jk] * (1-self.weight_decay) ** i * (0.5 ** zero_transitions if zero_transitions > 0 else 1)
                        ccv += 1
                    if self.borders[jk][0] < nk < self.borders[jk][1]:
                        k[jk] = nk
                    else:
                        res_sum = res_sum + 1

                if res_sum == range(len(nd)):
                    break

                i = i + 1
                ndo = np.copy(nd)
                nd = num_deriv(k, i)

            return (func(k),) + tuple(k)

        arr_lm = np.array([np.linspace(self.generating_borders[i][0], self.generating_borders[i][1], self.gen_koef_num) for i in
                           range(len(self.koef_names))]).transpose()
        m = min(list(map(find_local_min, arr_lm)))
        m = tuple(m)
        if self.approximation_function_type == "linear":
            if len(self.koef_names) == 1:
                kvls = list(self.x[self.var_names.index(self.y_name)][i] / self.x[1 - self.var_names.index(self.y_name)][i] for i in range(len(self.x)))
                dk = math.sqrt(sum([(m[1] - kvls[i])**2 for i in range(len(kvls))])/len(kvls))
                tk = un(m[1], dk)
                m = (m[0], tk)
            elif len(self.koef_names) == 2:
                xp = self.func.find(self.var_names[1 - self.var_names.index(self.y_name)])
                tkn = None
                tbn = None
                if self.func[xp-1] == "*":
                    if self.func[xp-1-len(self.koef_names[0]):xp-1] != self.koef_names[0]:
                        tkn = self.koef_names[1]
                        tbn = self.koef_names[0]
                    else:
                        tkn = self.koef_names[0]
                        tbn = self.koef_names[1]
                elif self.func[xp+1] == "*":
                    if self.func[xp+1:xp+1+len(self.koef_names[0])] != self.koef_names[0]:
                        tkn = self.koef_names[1]
                        tbn = self.koef_names[0]
                    else:
                        tkn = self.koef_names[0]
                        tbn = self.koef_names[1]

                yv = np.array(self.x[self.var_names.index(self.y_name)])
                xv = np.array(self.x[1 - self.var_names.index(self.y_name)])

                i2n = {self.koef_names.index(tkn): tkn, self.koef_names.index(tbn): tbn}
                n2i = {tkn: self.koef_names.index(tkn), tbn: self.koef_names.index(tbn)}

                dk = math.sqrt((1/(len(self.x[0])-2))*(yv.std()**2/xv.std()**2 - m[1+n2i[tkn]]**2))
                db = dk * math.sqrt(np.mean(xv**2))
                tkv = un(m[1+n2i[tkn]],dk)
                tbv = un(m[1 + n2i[tbn]], db)
                n2v = {tkn: tkv, tbn: tbv}
                m = (m[0], n2v[i2n[0]], n2v[i2n[1]])


        self.approx_val = {self.koef_names[i]: m[1 + i] for i in range(len(self.koef_names))}
        self.approx_val.update({"error": m[0]})
        return self.approx_val

    def get_error(self):
        return None if self.approx_val is None else self.approx_val["error"]

    def calc_error(self, koefs):
        return None if self.approx_val is None else self.functionv(koefs)

    def get_koefs(self):
        return None if self.approx_val is None else {k: self.approx_val[k] for k in self.approx_val.keys() if k != "error"}

    def prep_func(self, func):
        for k in self.rep_dict.keys():
            func = func.replace(k, DataCalc.rep_dict[k])

        if "=" in func:
            funcs = func.split("=")

            func = func[:func.find("=")+1] + "(" + func[func.find("=")+1:]
            func = func.replace("=", "-", 1)

            while func.count("=") > 0:
                func = func[:func.find("=")] + ")" + func[func.find("="):]
                func = func[:func.find("=") + 1] + "(" + func[func.find("=") + 1:]
                func = func.replace("=", "-", 1)

            func = func + ")"

            func = func.replace("=", "-")
            funcsf = list(filter(lambda x: len(re.split('[+\-()*+/,.|;]',x)) == 1, funcs))
            self.purefunc = list(filter(lambda x: len(x) >= 1 and x != self.y_name, funcs))[0]
            self.is_graphable = True if len(funcsf) == 1 else False
            if len(funcsf) >= 1 and (self.y_name == "" or self.y_name not in self.var_names):
                self.y_name = str(funcsf[0])

        return func


def runfile():
    x_0 = list()
    y_0 = list()
    with open(os.path.dirname(os.path.realpath(__file__)) + '/input.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        for row in spamreader:
            x_0 = np.append(x_0, float(str(row[0]).replace(",",".")))
            y_0 = np.append(y_0, float(str(row[1]).replace(",",".")))

    linap = DataCalc(func="y=c1+c2*x*log(x)+c3*e^x", koef_names="c1 c2 c3", x=x_0,
                     y=y_0, optim_func=DataCalc.error_funcs.squared_error,
                     borders=[-2, 2], generating_borders=[-11, 11],
                     gen_koef_num=11, approximation_function_type="square", max_iter=10 ** 3, weight_decay=0, derivative_weights=0.01, x_label="f_п, гц", y_label="f_д, гц", title="Частота прецессий от чавстоты диска")
    print(str(linap.approximate()) + "\n\n")
    print(str(linap.get_error()) + "\n\n")
    print(str(linap.get_koefs()) + "\n\n")
    linap.build_graph()


if __name__ == "__main__":
    runfile()
