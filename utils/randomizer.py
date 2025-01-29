import random


def generate_random_numbers(numbers):
    """
    Generate a random number for each number in the list,
    where the random number is between the number minus three and itself.

    Args:
        numbers (list): A list of four numbers.

    Returns:
        list: A list of random numbers with two decimal places.
    """
    if len(numbers) != 4:
        raise ValueError("The input list must contain exactly four numbers.")

    random_numbers = []
    for num in numbers:
        random_number = round(random.uniform(num - 2, num), 2)
        random_numbers.append(str(random_number))

    return random_numbers


def generate_random_numbers(numbers):
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
        lower_bound = max(0, num - 0.1)  # Ensure the lower bound doesn't go below 0
        random_number = round(random.uniform(lower_bound, num), 3)
        random_numbers.append(str(random_number))

    return random_numbers


if __name__ == "__main__":
    numbers = [0.69, 0.054, 0.55, 0.036, 0.41, 0.034, 0.20, 0.025, 0.95, 0.205]
    random_numbers = generate_random_numbers(numbers)
    print("& ".join(random_numbers))
