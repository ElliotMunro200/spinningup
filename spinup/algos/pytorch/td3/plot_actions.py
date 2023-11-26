# Updated script to include the predicted sine function parameters in the legend
import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sine_function(x, amplitude, frequency, phase, offset):
    """Sine function for curve fitting."""
    return amplitude * np.sin(frequency * x + phase) + offset

def r_squared(y_true, y_pred):
    """Calculate the R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    # Read and parse the data
    with open('data.txt', 'r') as file:
        lines = file.readlines()
    actions = [list(map(float, line.strip().strip('[]').split())) for line in lines]
    action1, action2, action3 = zip(*actions)

    NsI = 150
    action1 = action1[0:NsI]
    action2 = action2[0:NsI]
    action3 = action3[0:NsI]

    dt = 0.008
    numtimesteps = len(action1)
    time = numtimesteps * dt
    timesteps = np.linspace(0, time, num=numtimesteps)
    #timesteps = np.arange(len(action1) * dt)

    # Prepare data for curve fitting

    # Fit sine functions with reasonable bounds
    bounds = ([0.1, 0.1, -np.pi, -2], [2, 2, np.pi, 2])
    params1, _ = curve_fit(sine_function, timesteps, action1, bounds=bounds)
    params2, _ = curve_fit(sine_function, timesteps, action2, bounds=bounds)
    params3, _ = curve_fit(sine_function, timesteps, action3, bounds=bounds)
    params_me = (1.0, 25, m.pi, 0.0)

    # Generate sine fits for plotting
    sine_fit1 = sine_function(timesteps, *params1)
    sine_fit2 = sine_function(timesteps, *params2)
    sine_fit3 = sine_function(timesteps, *params3)
    sine_fit_me = sine_function(timesteps, *params_me)

    # Calculate R-squared scores
    r2_score1 = r_squared(action1, sine_fit1)
    r2_score2 = r_squared(action2, sine_fit2)
    r2_score3 = r_squared(action3, sine_fit3)
    r2_score_me = r_squared(action1, sine_fit_me)

    # Plot the results with predicted parameters in legend
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(timesteps, action1, 'b-', label='Original Action 1')
    axs[0].plot(timesteps, sine_fit1, 'r--', label=f'Sine Fit: A={params1[0]:.2f}, f={params1[1]:.2f}, ϕ={params1[2]:.2f}, O={params1[3]:.2f} (R²={r2_score1:.2f})')
    axs[0].plot(timesteps, sine_fit_me, 'g-', label=f'My Sine Fit: A={params_me[0]:.2f}, f={params_me[1]:.2f}, ϕ={params_me[2]:.2f}, O={params_me[3]:.2f} (R²={r2_score_me:.2f})')
    axs[0].set_title('Action 1 and Sine Fit')
    axs[0].legend()

    axs[1].plot(timesteps, action2, 'b-', label='Original Action 2')
    axs[1].plot(timesteps, sine_fit2, 'r--', label=f'Sine Fit: A={params2[0]:.2f}, f={params2[1]:.2f}, ϕ={params2[2]:.2f}, O={params2[3]:.2f} (R²={r2_score2:.2f})')
    axs[1].set_title('Action 2 and Sine Fit')
    axs[1].legend()

    axs[2].plot(timesteps, action3, 'b-', label='Original Action 3')
    axs[2].plot(timesteps, sine_fit3, 'r--', label=f'Sine Fit: A={params3[0]:.2f}, f={params3[1]:.2f}, ϕ={params3[2]:.2f}, O={params3[3]:.2f} (R²={r2_score3:.2f})')
    axs[2].set_title('Action 3 and Sine Fit')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
