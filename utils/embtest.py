#!/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def logarithmic_pitch_to_bin(pitch, pitch_min, pitch_max, num_bins):

  y = np.log(pitch - pitch_min + 1.0) / np.log(pitch_max - pitch_min + 1.0)

  y *= num_bins

  return y

def linear_pitch_to_bin(pitch, pitch_min, pitch_max, num_bins):

  y = (pitch - pitch_min) / (pitch_max - pitch_min)

  y *= num_bins

  return y

pitch_min = -5.42
pitch_max = 567.23

x = np.linspace(pitch_min, pitch_max, 500)

#y = [logarithmic_pitch_to_bin(xi, -5, 567, 16) for xi in x]
y = [linear_pitch_to_bin(xi, -5, 567, 16) for xi in x]

y_rounded = np.round(y)

#Alternative with scatter for rounded values (often better for discrete values)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
fig.suptitle("Logarithmic Function and Rounded Values (Scatter)")

ax1.plot(x, y, label="ln(x)", color='blue')
ax1.set_ylabel("y = ln(x)")
ax1.set_title("Logarithmic Function")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

ax2.scatter(x, y_rounded, label="Rounded ln(x)", color='green', marker='o', s=20) #Scatter plot
ax2.set_xlabel("x")
ax2.set_ylabel("Rounded y")
ax2.set_title("Rounded Integer Values (Scatter)")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
