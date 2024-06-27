import pandas as pd


def conductivity_data_processing(conductivity_data: dict):
    conductivities = []
    conductivity_measurements = conductivity_data["measurements"]
    for measurement in conductivity_measurements:
        conductivities.append(measurement)

    conductivity_df = pd.DataFrame(conductivities)

    return conductivity_df
