import pandas as pd


def xrf_data_processing(xrf_data: dict):
    layer1_data = []
    for measurement in xrf_data["measurements"]:
        layer1 = measurement['layer'][1]
        amt = []
        for comp in layer1['composition']:
            amt.append(comp["amount"])
            # todo check len(amt) == len(elements)
            layer_dict = {
                "position_x": measurement['position_x'],
                "position_y": measurement['position_y'],
                "thickness": layer1['thickness'],
                "amount": amt,
            }

            layer1_data.append(layer_dict)

    xrf_df = pd.DataFrame(layer1_data)
    elements = get_elements_from_str(xrf_data['material_names'])
    xrf_df["elements"] = [elements] * len(xrf_df)

    return xrf_df


def get_elements_from_str(ele_str):
    elements = []
    ele_strs =  ele_str.split('[%]')
    for s in ele_strs[:-1]:
        ele = s.split(':')[1].strip()
        elements.append(ele)
    return elements
