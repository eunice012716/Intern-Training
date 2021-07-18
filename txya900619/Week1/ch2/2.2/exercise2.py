import pandas as pd
import torch

from exercise1 import get_random_na_dataframe, process_data

if __name__ == "__main__":
    with open("example_dataset.csv", "r") as f:
        dataFrame = pd.read_csv(f)
        inputs, outputs = (
            process_data(get_random_na_dataframe(dataFrame.iloc[:, :-1])),
            dataFrame.iloc[:, -1],
        )

        X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

        print("X:\n", X)
        print()

        print("y:\n", y)
