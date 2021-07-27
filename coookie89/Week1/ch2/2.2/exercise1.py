import os
import torch
import pandas as pd

if __name__ == "__main__":
    os.makedirs(os.path.join(".", "data"), exist_ok=True)
    data_path = os.path.join(".", "data", "exercise.csv")

    with open(data_path, "w", encoding="utf-8") as data_file:
        data_file.write("name,gender,age,married\n")  # Column names
        data_file.write("Chako,M,20,N\n")  # Each row represents a data example
        data_file.write("Chocolate,NaN,21,N\n")
        data_file.write("Chicken,NaN,13,N\n")
        data_file.write("Kitchen,M,NaN,Y\n")
        data_file.write("Mary,F,31,N\n")
        data_file.write("Kerry,NaN,NaN,N\n")

    data = pd.read_csv(data_path)
    print("data ==>\n", data, "\n")

    m = data.isnull().sum(axis=0)  # 計算NaN每個col出現的次數
    print("NaN在col的出現次數 ==>\n", m, "\n")

    nan_max_num = m.max()  # 計算NaN出現最多次的數量

    # 找出最多NaN的col index,並刪除該column
    for index in m.index:
        if m[index] == nan_max_num:
            print("要刪掉的column ==>\n", index, "\n")
            temp = data.drop([index], axis=1)
            print("刪除後的data ==>\n", temp, "\n")

    # 將series轉換成tensor format
    temp = pd.get_dummies(data, dummy_na=True)
    tensor_format = torch.tensor(temp.values)
    print("轉換成tensor format ==>\n", tensor_format, "\n")
