import random
import pandas as pd
import numpy as np
import os


def get_random_id(low: int, high: int, miss_rate=0) -> int:
    assert 0 <= miss_rate <= 1, "probability must in [0,1]"
    miss = random.choices(range(0, 2), weights=[1 - miss_rate, miss_rate])[0]
    if miss is not None:
        return np.nan
    else:
        return random.randint(low, high)


def get_random_age(low: int, high: int, miss_rate=0) -> int:
    assert 0 <= miss_rate <= 1, "probability must in [0,1]"
    miss = random.choices(range(0, 2), weights=[1 - miss_rate, miss_rate])[0]
    if miss is not None:
        return np.nan
    else:
        return random.randint(low, high)


def get_process_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    na_cnt = dataframe.isnull().sum(axis=0).tolist()
    most_na_col = na_cnt.index(max(na_cnt))
    result = dataframe.drop(dataframe.columns[most_na_col], axis=1)
    return result


if __name__ == "__main__":
    department = ["0", "1", "2", "3", np.nan]
    gender = ["0", "1", np.nan]
    os.makedirs(os.path.join("example"), exist_ok=True)
    data_path = os.path.join("example", "example_df.csv")
    with open(data_path, "w", encoding="utf-8") as data_file:
        data_file.write("id,gender,age,department\n")
        for _ in range(0, 40):
            data_id = str(get_random_id(10000, 20000, 0.3)) + ","
            data_gender = str(random.choice(gender)) + ","
            data_age = str(get_random_age(24, 55, 0.3)) + ","
            data_department = str(random.choice(department)) + "\n"
            data_file.write(data_id + data_gender + data_age + data_department)
    data = pd.read_csv(data_path)
    print(data, end="\n\n")

    processed_data = get_process_data(data)
    print(processed_data)
