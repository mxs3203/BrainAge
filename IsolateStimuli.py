import sys
import pandas as pd

data_file = sys.argv[1]
markers_file = sys.argv[2]
output_file = sys.argv[3]

# Load data
data = pd.read_csv(data_file)
markers = pd.read_csv(markers_file)
