import numpy as np

path = r"D:\GitHub\DynaAlign\experiments\results\chaos_scan\chaos_scan_data.npz"

data = np.load(path)

L = data["L_values"]
lam = data["lam_values"]
H = data["heatmap"]

print("L axis:", L)
print("lam axis:", lam)
print("Heatmap shape:", H.shape)

print("\nHeatmap matrix (1 = Alive, 0 = Dead):\n")
print(H)

# count alive/dead
print("\nAlive count =", np.sum(H == 1))
print("Dead count  =", np.sum(H == 0))
