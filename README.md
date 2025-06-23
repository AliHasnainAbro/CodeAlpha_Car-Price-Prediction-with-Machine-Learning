This project implements a machine learning model to predict the selling price of used cars. It leverages various car attributes such as kilometers driven, fuel type, selling type, transmission, and the age of the car to build a robust prediction model using the Random Forest Regressor.

Table of Contents
Features

Installation

Dataset

Project Structure

Methodology

Results

Visualizations

Usage

Contributing

License

Features
Data Loading and Initial Exploration: Loads the car data.csv dataset and displays its first few rows and summary information.

Data Preprocessing and Feature Engineering:

Checks for missing values.

Calculates Car_Age based on the Year of manufacturing and a defined current_year.

Removes irrelevant features like Year and Car_Name.

Applies One-Hot Encoding to categorical features (Fuel_Type, Selling_type, Transmission) to convert them into a numerical format suitable for machine learning algorithms, avoiding multicollinearity by dropping the first category.

Exploratory Data Analysis (EDA):

Provides a statistical summary of numerical features.

Generates a correlation heatmap to visualize the relationships between different features in the dataset.

Model Training:

Splits the preprocessed data into training (80%) and testing (20%) sets.

Initializes and trains a Random Forest Regressor model with 100 estimators.

Model Evaluation:

Makes predictions on the test set.

Calculates and displays standard regression evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).

Results Visualization:

Generates a scatter plot comparing Actual vs. Predicted Selling Prices, including a reference line for ideal predictions.

Computes and visualizes Feature Importances to show which features contribute most significantly to the price prediction.

Installation
To set up and run this project, follow these steps:

Ensure Python is installed: This project requires Python 3.x.

Clone the repository (or download the project files):

git clone https://github.com/your-username/CodeAlpha_Car-Price-Prediction-with-Machine-Learning.git
cd car-price-prediction

(Replace https://github.com/your-username/CodeAlpha_Car-Price-Prediction-with-Machine-Learning.git with the actual repository URL if hosted online, or simply navigate to your project directory.)

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

Install the required Python libraries:

pip install pandas scikit-learn matplotlib numpy seaborn

Dataset
The project relies on a dataset named car data.csv. This CSV file should contain columns relevant to car features and a Selling_Price column as the target variable. Ensure this file is present in the root directory of the project alongside the Python script.

Example of expected columns (though the code handles additional columns via one-hot encoding):

Year: Manufacturing year of the car.

Selling_Price: The target variable, car's selling price (in lakhs, typically).

Present_Price: The ex-showroom price of the car.

Driven_kms: Kilometers driven by the car.

Fuel_Type: Type of fuel used (e.g., Petrol, Diesel, CNG).

Selling_type: Type of seller (e.g., Dealer, Individual).

Transmission: Transmission type (e.g., Manual, Automatic).

Owner: Number of previous owners.

Project Structure
.
├── car data.csv                              # The input dataset
├── car_price_prediction.py                   # The main Python script
├── README.md                                 # This README file
├── correlation_matrix.png                    # Generated: Correlation heatmap
├── actual_vs_predicted_selling_price.png     # Generated: Actual vs. Predicted values plot
└── feature_importances.png                   # Generated: Feature importances plot

Methodology
The project follows these main steps:

Data Loading and Inspection: The car data.csv is loaded. Basic information like the first 5 rows, data types, and non-null counts are displayed to get an initial understanding of the dataset.

Feature Engineering:

A new feature Car_Age is calculated by subtracting the Year of manufacture from the current_year (set to 2025 in the code).

The original Year column and Car_Name (which is too specific) are dropped as they are not directly used in the model.

Categorical Data Handling:

Unique values for Fuel_Type, Selling_type, and Transmission are printed to inspect categories.

One-hot encoding is applied to these categorical columns using pd.get_dummies(). drop_first=True is used to prevent multicollinearity, ensuring that n-1 binary columns are created for n categories.

Exploratory Data Analysis (EDA):

A statistical summary of the numerical features is generated using .describe().

A correlation heatmap is plotted using seaborn to visualize the linear relationships between all numerical features. This plot is saved as correlation_matrix.png.

Model Training:

The dataset is split into features (X, all columns except Selling_Price) and the target variable (y, Selling_Price).

The data is then divided into training (80%) and testing (20%) sets using train_test_split with random_state=42 for reproducibility.

A RandomForestRegressor model is initialized with n_estimators=100 and random_state=42, and then trained (.fit()) on the training data.

Model Evaluation:

Predictions (y_pred) are made on the test set (X_test).

The model's performance is quantitatively assessed using mean_absolute_error, mean_squared_error, np.sqrt(mse) for RMSE, and r2_score.

Results Visualization:

A scatter plot comparing y_test (actual prices) and y_pred (predicted prices) is generated. An ideal red dashed line is also plotted for reference. This plot is saved as actual_vs_predicted_selling_price.png.

Feature importances from the trained RandomForestRegressor model are extracted and visualized using a bar plot. This helps in understanding which features are most impactful in the prediction. This plot is saved as feature_importances.png.

Results
Upon successful execution of the script, the console output will display the various stages of data processing, model training, and evaluation metrics. Additionally, three image files will be generated in your project directory:

Console Output Example:

--- 1. Loading the Dataset ---
First 5 rows of the dataset:
   Car_Name  Year  Selling_Price  Present_Price  Driven_kms Fuel_Type  Selling_type Transmission  Owner
0     ritz  2014           3.35           5.59        27000    Petrol        Dealer       Manual      0
1      sx4  2013           4.75           9.54        43000    Diesel        Dealer       Manual      0
...
Mean Absolute Error (MAE): X.XX
Mean Squared Error (MSE): X.XX
Root Mean Squared Error (RMSE): X.XX
R-squared (R2): X.XX

--- 7. Feature Importance ---
Feature Importances:
Present_Price       0.79xxxx
Driven_kms          0.10xxxx
Car_Age             0.07xxxx
...

(Note: 'X.XX' and 'XXXX' will be replaced by actual numerical values when you run the script.)

Visualizations
The following plots will be generated in your project directory:

correlation_matrix.png
A heatmap visualizing the correlation coefficients between all numerical features. This helps in identifying multicollinearity and understanding feature relationships.

actual_vs_predicted_selling_price.png
A scatter plot showing the relationship between the actual car selling prices and the prices predicted by the Random Forest model. The red dashed line represents perfect predictions where actual equals predicted.

feature_importances.png
A bar chart indicating the relative importance of each feature in predicting the car's selling price, as determined by the Random Forest model. This provides insights into which factors are most influential.

Usage
Make sure you have all the Installation requirements met and the car data.csv file in the correct directory.

Open your terminal or command prompt.

Navigate to the project's root directory:

cd /path/to/your/car-price-prediction

Activate your virtual environment:

source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate   # Windows

Run the main Python script:

python car_price_prediction.py

The script will execute, display progress and results in the console, and save the generated plots to the current directory.

Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeatureName).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeatureName).

Open a Pull Request.
