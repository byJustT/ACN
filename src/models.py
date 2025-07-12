import numpy as np
from src import parameters  # 导入全局参数
import math

def calculate_transmission_metrics(wi, bi, device_params, task_params):
    I = task_params["I"]
    v = 1 / (task_params["w_over_v"] + 1)
    vi = v / I
    alpha_i = wi + vi

    d_m = device_params["distance"].values.astype(float)
    d_km = np.maximum(d_m / 1000, 0.001)
    PL_dB = 148 + 40 * np.log10(d_km)
    h = 10 ** (-PL_dB / 10)

    fi = device_params["fi"].values
    kappa = device_params["kappa"].values
    Piup = device_params["Piup"].values

    pi = (bi / (task_params["B"] + 1e-8)) * task_params["Pdown"]
    gamma_down = (pi * h) / (task_params["N0"] * bi + 1e-8)
    R_down = bi * np.log2(1 + gamma_down + 1e-8)
    T_down = (alpha_i * task_params["D"]) / (R_down + 1e-8)
    E_down = pi * T_down

    T_comp = (alpha_i * task_params["C"]) / (fi + 1e-8)
    E_comp = kappa * (alpha_i * task_params["C"]) * (fi ** 2)

    gamma_up = (Piup * h) / (task_params["N0"] * bi + 1e-8)
    R_up = bi * np.log2(1 + gamma_up + 1e-8)
    T_up = (alpha_i * task_params["O"]) / (R_up + 1e-8)
    E_up = Piup * T_up

    return {
        "T_down": T_down,
        "E_down": E_down,
        "T_comp": T_comp,
        "E_comp": E_comp,
        "T_up": T_up,
        "E_up": E_up,
        "pi": pi,
    }

def calculate_total_metrics(transmission_metrics):
    T_total = (
        np.max(transmission_metrics["T_down"])
        + np.max(transmission_metrics["T_comp"])
        + np.max(transmission_metrics["T_up"])
    )
    E_total = (
        np.sum(transmission_metrics["E_down"])
        + np.sum(transmission_metrics["E_comp"])
        + np.sum(transmission_metrics["E_up"])
    )
    return T_total, E_total
