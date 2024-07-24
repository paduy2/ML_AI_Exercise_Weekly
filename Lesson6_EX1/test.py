import pandas as pd

# Step 1: Create a DataFrame with the date string
data = {'date': ['13/7/20']}
df = pd.DataFrame(data)

# Step 2: Convert the date string to a pandas datetime object, specifying the format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

# Display the DataFrame
print(df)
