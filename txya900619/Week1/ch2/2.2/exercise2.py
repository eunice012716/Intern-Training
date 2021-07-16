from exercise1 import getRandomNaDataFrame, processData
import pandas as pd
import torch

if __name__ == "__main__":
    with open("example_dataset.csv", "r") as f:
        dataFrame = pd.read_csv(f)
        inputs, outputs = (
            processData(getRandomNaDataFrame(dataFrame.iloc[:, :-1])),
            dataFrame.iloc[:, -1],
        )

        X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

        print("X:\n", X)
        print()

        print("y:\n", y)
