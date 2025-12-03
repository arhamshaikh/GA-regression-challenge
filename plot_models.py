import matplotlib.pyplot as plt

# Epoch numbers
epochs_32 = list(range(1, 51))  # 50 epochs
epochs_16 = list(range(1, 51))

# 32-frame model validation MAE
val_mae_32 = [
    63.8456, 64.7149, 64.6690, 60.3441, 61.2244, 60.1517, 58.4588, 61.6185, 49.1919, 48.1026,
    43.0152, 39.5898, 49.1411, 50.0828, 40.0794, 72.9441, 37.6983, 41.2419, 32.7106, 36.9167,
    43.9593, 34.5707, 36.6540, 38.4114, 32.6140, 32.1083, 38.8454, 53.7185, 39.1937, 36.6606,
    30.8745, 37.9664, 34.5198, 33.2530, 35.3499, 33.4104, 37.8153, 33.8820, 31.4576, 33.0819,
    35.2337, 30.0224, 46.1790, 33.6474, 30.4849, 32.5070, 38.1546, 33.5809, 35.2999, 34.3740
]

# 16-frame model validation MAE
val_mae_16 = [
    127.5233, 54.0947, 52.5603, 39.6303, 29.2728, 33.6857, 30.1373, 36.2717, 20.0579, 23.5916,
    19.6572, 45.3435, 22.8042, 17.5100, 16.6647, 25.0240, 22.4303, 15.1991, 24.3680, 40.8663,
    14.7243, 15.6189, 15.6969, 22.2580, 16.0264, 16.5590, 20.1356, 18.5327, 17.8508, 16.2436,
    16.6798, 21.6462, 17.5116, 17.2698, 17.3483, 14.6439, 16.2888, 18.3891, 19.7771, 12.5431,
    14.7108, 10.9508, 19.5633, 18.6782, 16.9280, 10.7789, 12.7502, 17.5870, 13.0678, 14.8557
]

# Plotting
plt.figure(figsize=(12,6))
plt.plot(epochs_16, val_mae_16, color='blue', linestyle='-', marker='o', label='Model 16 Frames')
plt.plot(epochs_32, val_mae_32, color='red', linestyle='--', marker='s', label='Model 32 Frames')

# Labels and title
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.title('Validation MAE: 16-frame vs 32-frame Model')
plt.legend()
plt.grid(True)

# Save and show
plt.savefig("model_comparison_real.png")
plt.show()
