import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file, skip the header row
df = pd.read_csv('Datasett/secretary.csv', header=None)

    # Extract values from the first row
values = df.iloc[0].astype(int)
truth = df.iloc[1]

    # Count the number of red points (where the value in the second row is "p")
num_red_points = sum(tru == "p" for tru in truth)

    # Plotting
plt.scatter(range(1, len(values) + 1), values, c=['red' if tru == "p" else 'green' for tru in truth])
    #plt.scatter(500, 50, c='red')

    # Setting x-axis and y-axis limits
plt.xlim(0, 1001)
plt.ylim(20, 111)
#plt.scatter(500, values[1])

    # Adding labels and title
plt.ylabel('The selected mushroom')
plt.xlabel('Trials')
plt.title(f'Secretary plot for mushroom\nNumber of red points: {num_red_points}')

    # Display the plot
plt.show()
