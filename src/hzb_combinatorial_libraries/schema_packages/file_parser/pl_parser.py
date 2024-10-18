#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import numpy as np


def read_file_pl_unold(file_handle):
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
