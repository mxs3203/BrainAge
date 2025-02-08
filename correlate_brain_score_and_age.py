import json
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Specify the parent folder that contains the subfolders (e.g., BA1, BA2, BA3, etc.)
parent_folder = "results"  # Modify this path accordingly

# Initialize a list to store the DataFrames from each subfolder
df_list = []

# Iterate through each item in the parent folder
for folder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, folder_name)
    if os.path.isdir(subfolder_path):
        # Build the full path for "4_brain_score.csv" inside the subfolder
        csv_file = os.path.join(subfolder_path, "4_brain_score.csv")
        if os.path.exists(csv_file):
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Add a new column 'ID' with the subfolder name (e.g., BA1, BA2, etc.)
            df["ID"] = folder_name
            # Append the DataFrame to the list
            df_list.append(df)
        else:
            print(f"Warning: {csv_file} does not exist.")

# Concatenate all DataFrames into a single DataFrame
merged_brain_score_df = pd.concat(df_list, ignore_index=True)

# Read the Excel file with subject information
pilot_df = pd.read_excel("data/PilotDatasetSubjects.xlsx",skiprows=1)
final_df = pd.merge(merged_brain_score_df, pilot_df, on="ID", how="right")

print("Final merged DataFrame saved as 'merged_final.csv'.")
plt.figure(figsize=(10, 6))
sns.lmplot(
    x="Age",
    y="BrainScore",
    hue="Gender",
    data=final_df,
    markers=["o", "s"],  # Customize markers if you want different symbols for each gender
    height=6,
    aspect=1.5,
    ci=95  # Confidence interval for the regression line
)
plt.title("Correlation between Age and BrainScore")
plt.xlabel("Age")
plt.ylabel("BrainScore")
plt.tight_layout()
plt.show()