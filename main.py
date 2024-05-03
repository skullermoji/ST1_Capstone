import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('medical_insurance.csv')

# Calculates the Interquartile Range for charges
Q1 = data['charges'].quantile(0.25)
Q3 = data['charges'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers
filtered_data = data[(data['charges'] >= lower_bound) & (data['charges'] <= upper_bound)]

# Check the size of the filtered dataset compared to the original dataset
print("Original dataset size:", len(data))
print("Filtered dataset size:", len(filtered_data))

# Display first few rows of the dataset in consoel
print(data.head())

# Display general information and basic statistics in console
print(data.info())
print(data.describe())

# Correlation matrix
# corr_matrix = data.corr()
# plt.figure(figsize=(10, 8))
# plt.title("Correlation Matrix")
# plt.show()

# Pair plot to examine relationships among numerical features
sns.pairplot(data)
plt.show()

# Analyse categorical variables using boxplots
for column in ['sex', 'smoker', 'region']:
    sns.boxplot(x=column, y='charges', data=data)
    plt.title(f"Charges vs {column}")
    plt.show()

# Convert categorical variables to one-hot encoded format
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Display the updated DataFrame
print(data.head())

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop('charges', axis=1)
y = data['charges']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the Ridge regression model
model = Ridge(alpha=1.0) # Default alpha 1.0 as per Ridge documentation
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Scatter plot of actual vs predicted charges
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()

# Root window for the Tkinter GUI
root = tk.Tk()
root.title("Insurance Analysis")

# Generic file loading for flexibility
def load_data():
    # Open a file dialog to select the CSV file
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a CSV file.")
        return None
    
    # Loads data
    data = pd.read_csv(file_path)
    return data

# Displays data information
def display_data_info():
    data = load_data()
    if data is not None:
        # Display basic info
        info = data.info()
        print(info)  # For console output
        # Show basic stats in a new window
        new_window = tk.Toplevel(root)
        new_window.title("Data Information")
        info_label = tk.Label(new_window, text=str(data.describe()), justify=tk.LEFT)
        info_label.pack()

# Create scatter plots
def create_scatter_plot():
    data = load_data()
    if data is not None:
        plt.scatter(data['age'], data['charges']) # Can change as needed
        plt.xlabel("Age")
        plt.ylabel("Charges")
        plt.title("Age vs Charges")
        plt.show()

# Define a function to create box plots for categorical data
def create_box_plots():
    data = load_data()
    if data is not None:
        for column in ['sex', 'smoker', 'region']:
            sns.boxplot(x=column, y='charges', data=data)
            plt.title(f"Charges vs {column}")
            plt.show()

# Create buttons
info_button = tk.Button(root, text="Display Data Information", command=display_data_info)
scatter_button = tk.Button(root, text="Create Scatter Plot", command=create_scatter_plot)
box_plot_button = tk.Button(root, text="Create Box Plots", command=create_box_plots)

# Place the buttons in the GUI
info_button.pack(pady=10, padx=10)
scatter_button.pack(pady=10, padx=10)
box_plot_button.pack(pady=10, padx=10)

# Start the Tkinter event loop
root.mainloop()
