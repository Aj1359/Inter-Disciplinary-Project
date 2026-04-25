import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("efficiency.csv")

plt.figure(figsize=(8,6))
plt.plot(data["T"], data["Efficiency"], marker='o')
plt.xlabel("Number of Samples (T)")
plt.ylabel("Efficiency")
plt.title("Betweenness Ordering Efficiency vs T (Wiki-Vote)")
plt.grid(True)
plt.show()
