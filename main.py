import numpy as np
from scipy.stats import weibull_min as wbl
import matplotlib.pyplot as plt
from escenario1 import revenue_scenario1 as r1

lambda_val = 50
k_shape = 10
lb = 0.5
ls = 0.5
a = 53
b = 48
p0 = 51

weibull_dist = wbl(c=k_shape, scale=lambda_val)

x = np.linspace(0, 100, 1000)

pdf_values = weibull_dist.pdf(x)

plt.figure(figsize=(8, 5))            # Define el tamaño de la figura
plt.plot(x, pdf_values, lw=2, color='blue')  # Grafica la pdf
plt.title("Distribución Weibull (λ=50, K=10)")
plt.xlabel("Valor")
plt.ylabel("Densidad de probabilidad")
plt.grid(True)
plt.show()  # Muestra la gráfica



revenue = r1(a, b, p0, lb, ls)

print("Ganancia del market maker: " , revenue)