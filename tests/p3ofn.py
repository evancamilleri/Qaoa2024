from itertools import combinations

def generate_combinations(n):
    # Create a range from 0 to n (inclusive)
    numbers = range(n + 1)
    # Generate all combinations of 3 numbers from the range
    result = list(combinations(numbers, 3))
    return result

# Example usage: get all combinations of 3 numbers from 0 to 5
n = 5
combinations_result = generate_combinations(n)
print(combinations_result)
