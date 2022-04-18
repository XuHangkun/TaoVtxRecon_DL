import pandas as pd
import matplotlib.pyplot as plt
from opt import config_parser
import sys
import numpy as np
from tqdm import tqdm
plt.style.use("tao")

def cal_bias(df, key, kinetic = 1, draw = False):
    sel_condition = (df["fGdLSEdep"] > (kinetic + 1.01) - 0.1) & (df["fGdLSEdep"] < (kinetic + 1.01) + 0.1)
    edep_r = df["fGdLSEdepR"][sel_condition]
    bias = df["fRec%s"%(key)] - df["fGdLSEdep%s"%(key)]
    bias = bias[sel_condition]
    if key == "Phi":
        bias[bias > 180] = bias[bias > 180] - 360
        bias[bias < -180] = bias[bias < -180] + 360

    if draw:
        fig, ax = plt.subplots()
        hist = ax.hist2d(np.power(edep_r,3), bias, bins = [200, 200], cmin = 1)
        fig.colorbar(hist[3], ax = ax)
        plt.ylabel("%s Bias"%(key))
        plt.xlabel("$r_{edep}^{3}$ [$mm^{3}$]")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots()
        hist = ax.hist(bias, bins = 100, label = "$\mu$=%.2f; $\sigma$=%.2f"%(np.mean(bias), np.std(bias)))
        plt.xlabel("%s Bias"%(key))
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return len(bias), np.mean(bias), np.std(bias)

if __name__ == "__main__":
    # get config
    args = config_parser()
    sys.stdout.write(str(args) + "\n")
    df = pd.read_csv(args.eval_output)
    biases = []
    stds    = []
    key = "Z"
    for i in tqdm(range(10)):
        _, bias, std = cal_bias(df, key, i)
        biases.append(bias)
        stds.append(std)
    plt.plot(np.array(range(10)) + 1.01, biases)
    plt.xlabel("$E_{dep}$ [MeV]")
    plt.ylabel("Bias [mm]")
    plt.tight_layout()
    plt.show()
    plt.plot(np.array(range(10)) + 1.01, stds)
    plt.xlabel("$E_{dep}$ [MeV]")
    plt.ylabel("Vertex %s Resolution [mm]"%(key))
    plt.tight_layout()
    plt.show()
