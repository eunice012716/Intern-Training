import pandas as pd
import os
from exercise1 import process_data
import torch

if __name__ == "__main__":
    data_file = os.path.join("example", "example_df.csv")
    data = pd.read_csv(data_file)
    processed_data = process_data(data)
    inputs, outputs = processed_data.iloc[:, 0:4], processed_data.iloc[:, -1]
    inputs = inputs.fillna(inputs.mean())
    outputs = outputs.fillna(outputs.mean())
    print(inputs)
    print(outputs)
    print()

    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X)
    print(y)
