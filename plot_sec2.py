import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file, skip the header row
df = pd.read_csv('Datasett/secretary.csv', header=None)

# Extract values from the first row and convert to integers
values = df.iloc[0].astype(int)

# Plotting
for i, value in enumerate(values):
    color = 'red' if value == 100 else 'green'
    plt.scatter(i + 1, value, color=color)

# Setting y-axis and x-axis limits
plt.ylim(0, 101)
plt.xlim(0, 1001)

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customized Plot')

# Display the plot
plt.show()
