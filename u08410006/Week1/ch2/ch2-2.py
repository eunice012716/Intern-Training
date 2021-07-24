import os
import torch
import pandas as pd

if __name__ == "__main__":
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    data_path = os.path.join("..", "data", "house_tiny.csv")
    with open(data_path, "w", encoding="UTF-8") as data_file:
        data_file.write("NumRooms,Alley,Price\n")  # Column names
        data_file.write(
            "NA,Pave,127500\n"
        )  # Each row represents a data example
        data_file.write("2,NA,106000\n")
        data_file.write("4,NA,178100\n")
        data_file.write("NA,NA,140000\n")

    data = pd.read_csv(data_file, encoding="UTF-8")
    print(data)

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    print(inputs)

    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
    print(X, y)
