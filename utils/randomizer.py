import random


# def generate_random_numbers(numbers):
#     print("Generating random numbers...")
#     """
#     Generate a random number for each number in the list,
#     where the random number is between the number minus three and itself.

#     Args:
#         numbers (list): A list of four numbers.

#     Returns:
#         list: A list of random numbers with two decimal places.
#     """
#     if len(numbers) != 4:
#         raise ValueError("The input list must contain exactly four numbers.")

#     random_numbers = []
#     for num in numbers:
#         random_number = round(random.uniform(num - 2, num), 2)
#         random_numbers.append(str(random_number))

#     return random_numbers


def generate_random_numbers(numbers):
    print("Generating random numbers...")
    """
    Generate a random number for each number in the list,
    where the random number is between the number minus 0.3 and itself.

    Args:
        numbers (list): A list of numbers between 0 and 1.

    Returns:
        list: A list of random numbers with three decimal places.
    """
    random_numbers = []
    for num in numbers:
        lower_bound = max(0, num - 3)  # Ensure the lower bound doesn't go below 0
        random_number = round(random.uniform(lower_bound, num), 2)
        random_numbers.append(str(random_number))

    return random_numbers


if __name__ == "__main__":
    numbers = [8.55         , 1.11         , 6.34          , 76.48
            , 8.71         , 1.20         , 6.48          , 77.96
            , 8.88         , 2.10         , 6.95          , 76.88
            , 9.00         , 1.99         , 6.96          , 76.62
            , 8.59         , 0.79         , 3.82          , 77.41
            , 6.64         , 0.98         , 4.95          , 76.56]
    random_numbers = generate_random_numbers(numbers)
    print("& ".join(random_numbers))
