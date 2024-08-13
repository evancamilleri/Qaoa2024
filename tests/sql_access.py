import pyodbc

# Define your connection string
# Adjust the DRIVER, SERVER, DATABASE, UID, and PWD as necessary
connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=Quantum_MaxCut_20240808;"
    "Trusted_Connection=yes;"
    # "UID=your_username;"  # Uncomment if you use SQL Server authentication
    # "PWD=your_password;"  # Uncomment if you use SQL Server authentication
)

# Establish a connection to the database
try:
    connection = pyodbc.connect(connection_string)
    print("Connection successful!")
except pyodbc.Error as e:
    print("Error connecting to the database:", e)
    exit(1)

# Create a cursor object
cursor = connection.cursor()

# Define your query to fetch data from tb_C_AngleStudy
query = "SELECT * FROM tb_C_AngleStudy"

# Execute the query
cursor.execute(query)

# Fetch and print the results
columns = [column[0] for column in cursor.description]
rows = cursor.fetchall()

# Print column headers
print("\t".join(columns))

# Print each row
for row in rows:
    print("\t".join(str(value) for value in row))

# Close the connection
cursor.close()
connection.close()
