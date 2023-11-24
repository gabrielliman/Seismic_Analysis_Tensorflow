def closest_odd_number(float_number):
    # Convert the float to the nearest integer
    nearest_integer = round(float_number)

    # If the nearest integer is even, adjust to the closest odd integer
    if nearest_integer % 2 == 0:
        return nearest_integer + 1
    else:
        return nearest_integer