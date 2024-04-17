# Import necessary libraries (unchanged)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Set the working directory to the folder containing the CSV file
import os
os.chdir("Path of file/")

# Load the dataset into a Pandas DataFrame and remove 'condition' column
vegetable_df = pd.read_csv("vegetable_data.csv")


# Initialize a Flask application.
app = Flask(__name__)

# Define preprocessing steps for numerical features including 
# imputation for missing values (removed standard scaling)
numeric_features = ['Deasaster Happen in last 3month']
numeric_transformer = SimpleImputer(strategy='mean')

# Define preprocessing steps for categorical features including imputation and one-hot encoding.
categorical_features = ['Vegetable', 'Season', 'Month', 'State']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numerical and categorical transformers using ColumnTransformer.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest Regressor model.
model = RandomForestRegressor()

# Create a pipeline including preprocessing and modeling steps.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)])

# Train the model
# Define the features (X) and target variable (y).
X = vegetable_df.drop('Price per kg', axis=1)
y = vegetable_df['Price per kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        vegetable = request.form['vegetable']
        season = request.form['season']
        month = request.form['month']
        state = request.form['state']
        year = int(request.form['year'])  # Retrieve the year from the form

        # Set the value of disaster to 0
        disaster = 0

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Vegetable': [vegetable],
            'Season': [season],
            'Month': [month],
            'Deasaster Happen in last 3month': [disaster],
            'State': [state],
            'Year': [year]  # Include the 'Year' column
        })

        # Make predictions with the trained pipeline
        user_prediction = pipeline.predict(user_data)

        # Convert the prediction value to float
        user_prediction = float(user_prediction[0])

        # Generate graph
        # Assuming your dataset is stored in vegetable_df
        data = vegetable_df[vegetable_df['Vegetable'] == vegetable]
        plt.plot(data['Price per kg'], data['Year'], marker='o')
        plt.xlabel('Price per kg')
        plt.ylabel('Year')
        plt.title(f'Price per kg of {vegetable} over the years')
        
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert the plot to base64 encoding
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Pass the base64 encoded plot to the result template
        return render_template('result.html', prediction=user_prediction, vegetable=vegetable, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
