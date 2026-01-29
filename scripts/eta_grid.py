from pathlib import Path

import numpy as np
from aptapy.plotting import plt
from eta import hxeta

from hexsample import pipeline
from hexsample.hexagon import HexagonalLayout


def eta_grid():
    """Run a grid of simulations to study the eta function calibration parameters as functions
    of the readout noise (ENC).
    """
    # Number of simulations to run
    N = 5
    enc_ar = np.linspace(0, 60, N, dtype=int)
    # Arrays to store the fit parameters
    pars_2pix = np.zeros((N, 2))
    pars_3pix_r = np.zeros((N, 2))
    pars_3pix_theta = np.zeros((N, 2))
    # Loop over the different ENC values
    for i, enc in enumerate(enc_ar):
        output_file = Path.home() / "hexsampledata" / f"eta_grid_simulation_{enc}.h5"
        # Run simulation if the output file does not already exist
        if not output_file.exists():
            pipeline.simulate(
                num_events=100000,
                output_file=str(output_file),
                beam="hexagonal",
                enc=enc,
                zero_sup_threshold=30,
                readout_mode="circular",
                pitch=0.005,
                layout=HexagonalLayout.ODD_R,
                num_cols=304,
                num_rows=352,
                gain=1.

            )
        model_2pix, model_3pix_r, model_3pix_theta = hxeta(
            input_file=str(output_file)
        )
        pars_2pix[i, :] = model_2pix.parameter_values()
        pars_3pix_r[i, :] = model_3pix_r.parameter_values()
        pars_3pix_theta[i, :] = model_3pix_theta.parameter_values()
    # Plot the parameters as functions of ENC
    plt.figure("2pix")
    plt.plot(enc_ar, pars_2pix[:,0], ".", label="param 0")
    plt.plot(enc_ar, pars_2pix[:,1], ".", label="param 1")
    plt.xlabel("ENC [e]")
    plt.ylabel("2-pixel eta parameters")
    plt.legend()
    plt.figure("3pix dr")
    plt.plot(enc_ar, pars_3pix_r[:,0], ".", label="param 0")
    plt.plot(enc_ar, pars_3pix_r[:,1], ".", label="param 1")
    plt.xlabel("ENC [e]")
    plt.ylabel("3-pixel dr eta parameters")
    plt.legend()
    plt.figure("3pix theta")
    plt.plot(enc_ar, pars_3pix_theta[:,0], ".", label="param 0")
    plt.plot(enc_ar, pars_3pix_theta[:,1], ".", label="param 1")
    plt.xlabel("ENC [e]")
    plt.ylabel("3-pixel theta eta parameters")
    plt.legend()
    plt.show()
             

if __name__ == "__main__":
    eta_grid()
