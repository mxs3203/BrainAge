import requests
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix(matrix, labels, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(title+".png")

# Simulate input EEG data (replace with actual)
x = torch.randn(1, 512, 14)
data_flat = x.view(-1).tolist()

response = requests.post("http://127.0.0.1:8000/predict", json={"data": data_flat})
print(response.status_code)
print(response.text)
result = response.json()

print("Predicted age:", result["predicted_age"])

# Visualize summarized channel-level attention
plot_matrix(result["channel_attention"], result["channel_labels"], "Channel Attention")