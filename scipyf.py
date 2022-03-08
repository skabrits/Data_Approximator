import numpy as np
import scipy as sp
import scipy.optimize
from PMcalcl.main import un
import os
import matplotlib.pyplot as plt
import math
import csv
from PMcalcl.main import un
import re


class DataCalc:

    class error_funcs:
        lm= "lm"
        trf = "trf"
        db = "dogbox"
        errors = [lm,trf,db]

    class mode:
        approximation = "approximation"
        calculator = "calculator"
        errors = [approximation, calculator]

    rep_dict = {"^": "**", "e": "math.e", "pi": "math.pi", "log": "math.log", "cos": "math.cos", "sin": "math.sin",
                "tg": "math.tan", "ln": "math.log", "sqrt": "math.sqrt", "exp": "math.exp", " ": ""}

    def __init__(self, func="", mode=mode.approximation, y_name="y", koef_names="", borders=None,
                 uncertain=False,
                 optimization_method=None, x_label="x", y_label="y", title="",
                 **kwargs):
        """
        :param func: function you want to approximate with (like in wolfram alpha)
        :param mode: mode in which you would like to use DataCalc -- whether as approximation software or as calculation;
                                        in calculation mode you have only to pass function and corresponding data
        :param y_name: name of variable that defines function  output, live "" for automatic detection
        :param koef_names: names of function params
        :param borders: smaller and upper limits of coefficients for each coefficient
        :param uncertain: if true output numbers have uncertainties
        :param optimization_method: optimization method:
                                                       ‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
                                                        ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical use case is small problems with bounds. Not recommended for problems with rank-deficient Jacobian.
                                                        ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle bounds and sparse Jacobians. Usually the most efficient method for small unconstrained problems.
        :param x_label: label for x axis of plot
        :param y_label: label for x axis of plot
        :param title: title of plot
        :param kwargs: all function variables including output of function

        This function initializes class that contains data (mesurments: parameters and output) and can execute
        operations on it (at the moment approximate data by custom function and build graph of data). It requires
        you:
        1) to pass data, you want to work with in form of several arrays, where each array is passed as named
        parameter to function
        2) to pass function you want to approximate or calculate data with in form of string (whether as in
        wolfram alpha's format or as python code). Note: names of parameters of function must match data's names. It
        should be weather equation on output ("y") variable (better option), or any equation
        3) to pass function coefficients' names (names of all "letters" in equation, which are not parameters (data)).
        They must be separated with space.

        Also if output ("y") variable name is not y you can pass its name to y_name parameter, or "" if you want program
        to detect it automatically. Other parameters are described higher.

        Example of usage:

        linap = DataCalc(func="y=(k*x^2+b)^1/2", koef_names="k b", x=[1, 3, 5], y=[4, 8, 12],
        optimization_method=DataCalc.error_funcs.lm)

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

        self.vars = kwargs
        self.var_names = list(kwargs.keys())
        self.x = list(self.vars[k] for k in kwargs.keys())
        self.y_name = y_name
        self.purefunc = ""
        self.approx_val = None
        self.is_graphable = False
        self.func = self.prep_func(func)

        self.calc_result = None
        self.optim_func = None
        self.title = None
        self.y_label = None
        self.x_label = None
        self.koef_names = []
        self.entered_borders = None
        self.borders = []
        self.uncertain = None

        if mode == self.mode.approximation:
            self.optim_func = optimization_method
            self.title = title
            self.y_label = y_label
            self.x_label = x_label

            self.koef_names = list(filter(lambda x: x != "", koef_names.strip().split(" ")))

            self.entered_borders = True

            if borders is None:
                borders = [[-10000, 10000] for _ in range(len(self.koef_names))]
                self.entered_borders = False

            if type(borders[0]) in {int,float}:
                borders = [[-borders[0], borders[1]] for _ in range(len(self.koef_names))]

            self.borders = borders
            self.uncertain = uncertain

        else:
            ml = max(map(lambda x: len(x) if hasattr(x, "__getitem__") else 0, self.vars.values()))
            for k in self.var_names:
                if k != self.y_name:
                    if not hasattr(self.vars[k], "__getitem__"):
                        self.vars[k] = np.ones(ml) * self.vars[k]

            self.calc_result = self.functionv(np.array([self.vars[k] for k in self.var_names if k != self.y_name]))

    def functionv(self, *params):
        koefs = [i for i in params[1:]]
        for i in range(len(self.var_names)):
            if str(self.var_names[i]) != self.y_name:
                exec(str(self.var_names[i]) + " = " + "params[0][i]")
        for i in range(len(self.koef_names)):
            exec(str(self.koef_names[i]) + " = " + str(koefs[i]))
        try:
            return eval(str(self.func))
        except Exception:
            evalar = np.zeros((len(params[0][0]),))
            for j in range(len(params[0][0])):
                for i in range(len(self.var_names)):
                    if str(self.var_names[i]) != self.y_name:
                        exec(str(self.var_names[i]) + " = " + "params[0][i][j]")
                evalar[j] = eval(str(self.func))
            return np.array(evalar)

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
        # bounds = np.array(self.borders).transpose(),
        [popt, pcov] = sp.optimize.curve_fit(func, np.array([self.vars[k] for k in self.var_names if k != self.y_name]), np.array(self.vars[self.y_name]), p0 = [(self.borders[i][0]+self.borders[i][1])/2 for i in range(len(self.borders))], bounds = np.array(self.borders).transpose(), absolute_sigma = True) if self.entered_borders else sp.optimize.curve_fit(func, np.array([self.vars[k] for k in self.var_names if k != self.y_name]), np.array(self.vars[self.y_name]), p0 = [(self.borders[i][0]+self.borders[i][1])/2 for i in range(len(self.borders))], absolute_sigma = True)
        perr = np.sqrt(np.diag(pcov))
        m = (((np.array(self.vars[self.y_name])-func(np.array([self.vars[k] for k in self.var_names if k != self.y_name]), *popt))**2).mean(),) + tuple(popt) + tuple(perr)

        if self.uncertain:
            self.approx_val = {self.koef_names[i]: un(m[1 + i], m[len(self.koef_names) + i]) for i in range(len(self.koef_names))}
        else:
            self.approx_val = {self.koef_names[i]: m[1 + i] for i in range(len(self.koef_names))}
        self.approx_val.update({"error": m[0]})
        return self.approx_val

    def get_error(self):
        return None if self.approx_val is None else self.approx_val["error"]

    def calc_error(self, koefs):
        return None if self.approx_val is None else ((np.array(self.vars[self.y_name])-self.functionv(np.array([self.vars[k] for k in self.var_names if k != self.y_name]), *[self.approx_val[i] for i in self.approx_val.keys() if i != "error"]))**2).mean()

    def calculate(self):
        return self.calc_result

    def get_koefs(self):
        return None if self.approx_val is None else {k: self.approx_val[k] for k in self.approx_val.keys() if k != "error"}

    def prep_func(self, func):
        for k in self.rep_dict.keys():
            func = func.replace(k, DataCalc.rep_dict[k])

        if "=" in func:
            funcs = func.split("=")

            func = "(" + func[func.find("=")+1:]

            while func.count("=") > 0:
                func = func[:func.find("=")] + ")" + func[func.find("="):]
                func = func[:func.find("=") + 1] + "(" + func[func.find("=") + 1:]
                func = func.replace("=", "-", 1)

            func = func + ")"

            func = func.replace("=", "-")
            funcsf = list(filter(lambda x: len(re.split('[+\-()*+/,.|;]',x)) == 1, funcs))
            self.is_graphable = True if len(funcsf) == 1 else False
            if len(funcsf) >= 1 and (self.y_name == "" or self.y_name not in self.var_names):
                self.y_name = str(funcsf[0])
            self.purefunc = list(filter(lambda x: len(x) >= 1 and x != self.y_name, funcs))[0]

        return func


def runfile():
    x_0 = list()
    y_0 = list()
    with open(os.path.dirname(os.path.realpath(__file__)) + '/input.csv') as csvfile:
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


if __name__ == "__main__":
    runfile()