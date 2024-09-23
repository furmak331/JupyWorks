import pandas as pd

# Create a simple DataFrame
data = {
    'Name': ['Asm', 'Lamma', 'Faiz', 'Danish'],
    'Age': [25, 30, 35, 40],
    'City': ['Nil', 'London', 'india', 'Srg']
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("DataFrame:")
print(df)

# Access specific columns
print("\nNames column:")
print(df['Name'])

# Filter rows where Age > 30
filtered_df = df[df['Age'] > 30]
print("\nFiltered DataFrame (Age > 30):")
print(filtered_df)

# Basic statistics on the 'Age' column
print("\nBasic statistics on Age:")
print(df['Age'].describe())
