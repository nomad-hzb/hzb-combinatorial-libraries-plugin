import numpy as np
import pandas as pd


def read_file_pl_unold(file_path: str):
    with open(file_path, 'r+') as file_handle:
        header = {}
        line_split = file_handle.readline().split(";")
        while len(line_split) == 2:
            key = line_split[0].strip().strip('"')
            value = line_split[1].strip().strip('"')
            header[key] = value
            line_split = file_handle.readline().split(";")
        line_split = file_handle.readline().split(";")
        wavelengths = np.array(line_split[-1].strip().split(","))
        columns = ["x", "y", "z", "neutral_density", "power_transmitted", "int_time_PL_sample"]
        columns.extend(wavelengths)
        df = pd.read_csv(file_handle, names=columns, delimiter=';|,', engine='python')
        df = df.round(3)
    cut_off_wavelength = 420
    columns = ["x", "y", "z", "neutral_density", "power_transmitted", "int_time_PL_sample"]
    columns.extend(df.columns[6:][np.array(df.columns[6:], dtype=np.float64) > cut_off_wavelength])
    df = df[columns]
    return header, df.dropna(axis=1)


header, df = read_file_pl_unold("tests/data/4025-12__PL2024_03_12_1120.csv")
print(header)
