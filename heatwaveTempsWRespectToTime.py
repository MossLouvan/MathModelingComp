import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Temp outdoor over the24 hour period
time_hours = np.arange(0, 24)  # 24-hour period
outdoor_temp = [85, 85, 84, 83, 83, 83, 84, 88, 91, 94, 96, 97, 
                100, 100, 102, 102, 100, 99, 97, 94, 92, 91, 90, 89]  # °F

# Convert Fahrenheit to Celsius
outdoor_temp_C = [(t - 32) * 5/9 for t in outdoor_temp]

# Making the variables constant to the home in each
homes = {
    "Home 1": {"size_m2": 88, "stories": 1, "shade_factor": 0.8, "occupants": 3},
    "Home 2": {"size_m2": 63, "stories": 2, "shade_factor": 0.5, "occupants": 3},
    "Home 3": {"size_m2": 74, "stories": 25, "shade_factor": 0.2, "occupants": 2},
    "Home 4": {"size_m2": 278, "stories": 2, "shade_factor": 0.2, "occupants": 6},
}

# Heat const
heat_per_person = 100  # Watts per person ~ converted to °C effect
heat_transfer_coefficient = 0.1  # Represents insulation and material heat retention


def indoor_temp_model(t, T_in, T_out_func, shade_factor, occupants, size_m2):
    T_out = np.interp(t, time_hours, T_out_func)  # Interpolating outdoor temp over time
    heat_gain_people = (heat_per_person * occupants) / (size_m2 * 50)  # Adjusted for home size
    dTdt = heat_transfer_coefficient * (T_out - T_in) * shade_factor + heat_gain_people
    return dTdt

#create list to solve for each home
indoor_temps = {}

for home, params in homes.items():
    sol = solve_ivp(
        indoor_temp_model, [0, 24], [outdoor_temp_C[0]], 
        args=(outdoor_temp_C, params["shade_factor"], params["occupants"], params["size_m2"]),
        t_eval=time_hours
    )
    indoor_temps[home] = sol.y[0]

# Plot data using matplotlib
plt.figure(figsize=(10, 6))
for home, temps in indoor_temps.items():
    plt.plot(time_hours, temps, label=home)

plt.plot(time_hours, outdoor_temp_C, 'k--', label="Outdoor Temperature", linewidth=2)
plt.xlabel("Time (Hours)")
plt.ylabel("Temperature (°C)")
plt.title("Predicted Indoor Temperatures Over 24 Hours (No A/C)")
plt.legend()
plt.grid(True)
plt.show()