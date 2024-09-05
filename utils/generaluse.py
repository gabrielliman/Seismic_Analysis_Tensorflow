def closest_odd_number(float_number):
    # Convert the float to the nearest integer
    nearest_integer = round(float_number)

    # If the nearest integer is even, adjust to the closest odd integer
    if nearest_integer % 2 == 0:
        return nearest_integer + 1
    else:
        return nearest_integer
    
def get_equal_limits(start,end,num_limits):
    step_size = (start - end) / (num_limits + 1)

    limits = []
    for i in range(1, num_limits + 1):
        limit = end + i * step_size
        limits.append(limit)

    return limits