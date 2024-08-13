import re


# Define the cost_function string
cost_function = "3.625*Z[0]*Z[1]*Z[2] - 3.625*Z[0]*Z[1] - 3.625*Z[0]*Z[2] + 3.625*Z[0] - 5.25*Z[1]*Z[2]*Z[3] + 1.625*Z[1]*Z[2] + 5.25*Z[1]*Z[3] - 1.625*Z[1] + 1.5*Z[2]*Z[3]*Z[4] + 3.75*Z[2]*Z[3] - 1.5*Z[2]*Z[4] - 0.125*Z[2] - 8.875*Z[3]*Z[4]*Z[5] + 7.375*Z[3]*Z[4] + 8.875*Z[3]*Z[5] - 12.625*Z[3] + 8.875*Z[4]*Z[5] - 7.375*Z[4] - 8.875*Z[5] + 9.0"
cost_function = cost_function.replace(" ", "")

# Correctly use regex to split the cost_function by + and -, keeping the sign with the term
# The adjusted pattern looks for any + or - that is not immediately after an opening bracket [
terms = re.split(r'(?<!\[)(?=[+-])', cost_function)

# The resulting terms should be split correctly, with each sign retained
print(terms)

###########################


edge_orbits = [
    {(0, 1, 2), (3, 4, 5)}, {(0, 1), (4, 5)}, {(0, 2), (3, 5)}, {(0,), (5,)},
    {(1, 2, 3), (2, 3, 4)}, {(1, 2), (3, 4)}, {(2, 4), (1, 3)}, {(1,), (4,)},
    {(2, 3)}, {(2,), (3,)}
]

# Prepare to store the new structure similar to edge_orbits but with coefficients
coef_orbits = []

# Iterate over each set in edge_orbits
for orbit_set in edge_orbits:
    coef_set = set()

    for orbit_tuple in orbit_set:
        # For each term in terms, find matching z indices and extract coefficient
        for term in terms:
            # Extract all Z indices from the term
            z_indices = [int(match.group(1)) for match in re.finditer(r'Z\[(\d+)\]', term)]

            # Attempt to extract a coefficient, defaulting to 1.0 if not found or empty
            coefficient_match = re.match(r'^([+-]?\d*\.?\d*)(?=\*)', term)
            coefficient = float(coefficient_match.group(1)) if coefficient_match and coefficient_match.group(1) else 1.0

            # Check if current orbit tuple matches z indices from term
            if set(orbit_tuple) == set(z_indices):
                coef_set.add(coefficient)
                break  # Break if match is found to avoid duplicate additions

    coef_orbits.append(coef_set)

print(coef_orbits)

# Initialize a list to store the results (averages or the single value)
averaged_coef_orbits = []

# Loop through each set in coef_orbits
for coef_set in coef_orbits:
    # Check if there is more than one value in the set
    if len(coef_set) > 1:
        # Calculate the average if there are multiple coefficients
        #average = sum(coef_set) / len(coef_set)
        average = sum(abs(coef) for coef in coef_set) / len(coef_set)
        averaged_coef_orbits.append({average})  # Add the average as a single-item set
    else:
        # If there's only one coefficient, just add it as is
        averaged_coef_orbits.append(coef_set)

# Display the resulting list
print(averaged_coef_orbits)

