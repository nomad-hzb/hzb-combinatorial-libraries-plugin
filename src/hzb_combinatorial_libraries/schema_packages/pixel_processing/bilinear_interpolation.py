import pandas as pd


def bilinear_interpolation(x, y, value, x_new, y_new):

    """
    Perform bilinear interpolation to estimate the value at (x_new, y_new)
    within the grid defined by the points (x, y) and corresponding values.

    Args:
    - x: List of x-coordinates in ascending order.
    - y: List of y-coordinates in descending order.
    - values (list): List of corresponding values.
    - x_new (float): New x-coordinate for interpolation.
    - y_new (float): New y-coordinate for interpolation.

    Returns:
    - float: Interpolated value at (x_new, y_new).
    """
    # Find the indices of the points surrounding the target point
    i = 0

    while x[i] < x_new:

        i += 1
        if i == len(x) - 1:
            break

    j = 0
    while y[j] > y_new:
        j += 1
        if j == len(y) - 1:
            break

    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[j - 1], y[j]


    """
    
    (x0, y0)   (x1, y0)
     (Q11)----(Q21)
       |       |
       |    Q  |
       |       |
    (Q12)----(Q22)
    (x0, y1)   (x1, y1)


    Q (x_new, y_new)
    """

    data = pd.DataFrame({
        'x': x,
        'y': y,
        'value': value,
    })

    Q12 = data.loc[(data['x'] == x0) & (data['y'] == y1)]["value"].values
    if len(Q12) == 0:
        return None
    else:
        Q12 = Q12[0]
    Q11 = data.loc[(data['x'] == x0) & (data['y'] == y0)]["value"].values
    if len(Q11) == 0:
        return None
    else:
        Q11 = Q11[0]
    Q21 = data.loc[(data['x'] == x1) & (data['y'] == y0)]["value"].values
    if len(Q21) == 0:
        return None
    else:
        Q21 = Q21[0]
    Q22 = data.loc[(data['x'] == x1) & (data['y'] == y1)]["value"].values
    if len(Q22) == 0:
        return None
    else:
        Q22 = Q22[0]

    x_factor = (x1 - x_new) / (x1 - x0)
    y_factor = (y1 - y_new) / (y1 - y0)

    interpolated_value = Q11 * x_factor * y_factor + Q21 * (1 - x_factor) * y_factor + \
                         Q12 * x_factor * (1 - y_factor) + Q22 * (1 - x_factor) * (1 - y_factor)


    # check if the interpolated value is within the range of the data
    if i == len(x) - 1 and x[i] < x_new:
        interpolated_value = None
    if j == len(y) - 1 and y[j] > y_new:
        interpolated_value = None

    return interpolated_value
