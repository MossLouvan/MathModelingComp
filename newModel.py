import sympy as sp
from sympy import symbols, integrate, lambdify
import numpy as np 
import matplotlib.pyplot as plt

# Define variables
x, k = sp.symbols('x k')

# Define the function
expr = (
    (0.010662 * sp.exp(k*x) / k**7) - (0.010662 * x * sp.exp(k*x) / k**6) +
    (0.093264 * sp.exp(k*x) / k**6) + (0.00533099 * x**2 * sp.exp(k*x) / k**5) -
    (0.093264 * x * sp.exp(k*x) / k**5) + (0.21918 * sp.exp(k*x) / k**5) +
    (0.001777 * x**3 * sp.exp(k*x) / k**4) - (0.046632 * x**2 * sp.exp(k*x) / k**4) -
    (0.21918 * x * sp.exp(k*x) / k**4) + (0.456 * sp.exp(k*x) / k**4) +
    (0.000444249 * x**4 * sp.exp(k*x) / k**3) - (0.015544 * x**3 * sp.exp(k*x) / k**3) +
    (0.10959 * x**2 * sp.exp(k*x) / k**3) + (0.456 * x * sp.exp(k*x) / k**3) -
    (1.985 * sp.exp(k*x) / k**3) + (0.000088498 * x**5 * sp.exp(k*x) / k**2) +
    (0.003886 * x**4 * sp.exp(k*x) / k**2) - (0.03653 * x**3 * sp.exp(k*x) / k**2) +
    (0.228 * x**2 * sp.exp(k*x) / k**2) + (1.985 * x * sp.exp(k*x) / k**2) -
    (85.9 * sp.exp(k*x) / k**2) + (0.000148083 * x**6 * sp.exp(k*x) / k) -
    (0.00007772 * x**5 * sp.exp(k*x) / k) + (0.0091325 * x**4 * sp.exp(k*x) / k) +
    (0.076 * x**3 * sp.exp(k*x) / k) - (0.9925 * x**2 * sp.exp(k*x) / k) +
    (85.9 * x * sp.exp(k*x) / k)
)

# Calculate the integral of the expression with respect to x
integral = sp.integrate(expr, x)

# Define the function for indoor temperature over time (modeled in Fahrenheit)
def Tint(d, c, k, A, t, s, Ti):
    # Ensure x and k are treated correctly in NumPy operations
    integral_value = sp.lambdify((x, k), integral, modules=["numpy"])
    
    # Convert t to NumPy array for compatibility
    t = np.asarray(t, dtype=np.float64)
    
    term1 = ((d * c) / (k * A)) * integral_value(t, k)
    term2 = Ti * np.exp((-k * A * t) / (d * c))
    term3 = ((k * A * t) / (d * c)) * 17.5 * s * np.exp((-k * A * t) / (d * c))

    return term1 + term2 + term3  # Temperature in Fahrenheit

# Time labels for 24-hour period
time_labels = [
    "12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM", "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM",
    "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"
]

# Outdoor temperatures (in Fahrenheit) for each hour
outdoor_temps = np.array([85, 85, 84, 83, 83, 83, 84, 88, 91, 94, 96, 97, 100, 100, 102, 102, 100, 99, 97, 94, 92, 91, 90, 89])

# Define parameters based on Memphis, TN average values
d = 0.1524  # Thickness of walls (meters)
c = 900     # Specific heat capacity (J/kg*K)
k = 0.04    # Thermal conductivity (W/m*K)
A = 100     # Surface area of walls (m^2)
s = 0.7     # Solar heat gain factor
Ti = 71.6   # Initial indoor temperature (Fahrenheit)

# Time range in hours (converted to seconds for the function)
t_hours = np.arange(0, 24, 1)  # Every hour from 0 to 24
T_indoor = [Tint(d, c, k, A, t * 3600, s, Ti) for t in t_hours]  # Compute indoor temps

# Plot both indoor and outdoor temperatures on the same graph
time_range = np.arange(24)
plt.figure(figsize=(10, 5))
plt.plot(time_range, outdoor_temps, marker='o', linestyle='-', color='blue', label='Outdoor Temperature')
plt.plot(time_range, T_indoor, marker='o', linestyle='-', color='red', label='Indoor Temperature')
plt.xticks(time_range, time_labels, rotation=45)
plt.xlabel('Time of Day')
plt.ylabel('Temperature (°F)')
plt.title('Indoor and Outdoor Temperature Over 24 Hours (No AC)')
plt.legend()
plt.grid()
plt.show()
