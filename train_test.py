import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Datasett/mushrooms.csv")

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and testing sets
train_set.to_csv("Datasett/train_set.csv", index=False)
test_set.to_csv("Datasett/test_set.csv", index=False)
