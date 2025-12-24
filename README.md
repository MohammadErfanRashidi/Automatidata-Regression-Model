# NYC Yellow Taxi Fare Prediction

## Overview
This project aims to predict the fare amount for New York City yellow taxi trips using a linear regression model. The analysis involves loading the taxi trip data, performing extensive data cleaning and feature engineering, exploratory data analysis (EDA), and finally building and evaluating a predictive model.

## Dataset
The dataset used is the **2017 Yellow Taxi Trip Data**, loaded from `/content/2017_Yellow_Taxi_Trip_Data.csv`. It contains detailed information about individual taxi trips, including:
-   `VendorID`: A code indicating the TPEP provider that provided the record.
-   `tpep_pickup_datetime`: The date and time when the meter was engaged.
-   `tpep_dropoff_datetime`: The date and time when the meter was disengaged.
-   `passenger_count`: The number of passengers in the vehicle.
-   `trip_distance`: The elapsed trip distance in miles.
-   `RatecodeID`: The final rate code in effect at the end of the trip.
-   `PULocationID`: Taxi Zone ID where the taximeter was engaged.
-   `DOLocationID`: Taxi Zone ID where the taximeter was disengaged.
-   `payment_type`: Numeric code indicating how the passenger paid for the trip.
-   `fare_amount`: The time-and-distance fare calculated by the meter.
-   Additional charges such as `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, and `improvement_surcharge`.
-   `total_amount`: The total amount charged to passengers.

Initial inspection revealed 22,699 entries and 18 columns. There were no missing values or duplicate rows.

## Methodology

### 1. Data Loading and Initial Inspection
-   Loaded the CSV file into a Pandas DataFrame.
-   Checked the shape, info, and descriptive statistics of the dataset.

### 2. Data Cleaning and Preprocessing
-   Converted `tpep_pickup_datetime` and `tpep_dropoff_datetime` columns to datetime objects.
-   Created a `duration` column (trip duration in seconds).
-   Handled outliers in `fare_amount` and `duration` by setting negative values to zero and capping extreme values using an IQR-based imputer function.
    -   `fare_amount` was capped between 0 and 26.5.
    -   `duration` was capped between 0 and 2159 seconds.

### 3. Feature Engineering
-   **`pickup_dropoff`**: Combined `PULocationID` and `DOLocationID` to identify unique routes.
-   **`mean_distance`**: Calculated the average `trip_distance` for each unique `pickup_dropoff` pair and mapped it back to the DataFrame.
-   **`mean_duration`**: Calculated the average `duration` for each unique `pickup_dropoff` pair and mapped it back to the DataFrame.
-   **Time-based features**: Extracted `day_of_week`, `month`, and `hour` from `tpep_pickup_datetime`.
-   **`rush_hour`**: Created a binary feature indicating whether a trip occurred during weekday rush hours (6-10 AM or 4-8 PM).

### 4. Exploratory Data Analysis (EDA)
-   Visualized distributions and outliers for `trip_distance`, `fare_amount`, and `duration` using box plots.
-   Analyzed descriptive statistics for key columns.
-   Created a scatter plot between `mean_duration` and `fare_amount`.
-   Generated a pairplot for `fare_amount`, `mean_duration`, and `mean_distance` to observe relationships.
-   Displayed a correlation heatmap to understand feature interdependencies.

### 5. Model Building
-   Selected relevant features for modeling: `VendorID`, `passenger_count`, `mean_distance`, `mean_duration`, and `rush_hour`.
-   Converted `VendorID` to a string type and applied one-hot encoding using `pd.get_dummies`.
-   Split the data into training (80%) and testing (20%) sets.
-   Standardized the features using `StandardScaler`.
-   Trained a `LinearRegression` model on the prepared data.

### 6. Model Evaluation
-   Predicted `fare_amount` on the test set.
-   Evaluated the model using common regression metrics:
    -   **R-squared (R²)**: 0.7904
    -   **Mean Absolute Error (MAE)**: 2.1267
    -   **Mean Squared Error (MSE)**: 9.3429
    -   **Root Mean Squared Error (RMSE)**: 3.0566
-   Visualized the model's performance with a scatter plot comparing actual vs. predicted `fare_amount` and a histogram of residuals.
-   Displayed the model's coefficients and intercept:
    -   Coefficients: `[ 0.0059311 (passenger_count), 1.62005303 (mean_distance), 4.63246821 (mean_duration), 0.02792427 (rush_hour), -0.02401904 (VendorID_2)]`
    -   Intercept: `11.650163555261853`

## Results
The linear regression model achieved an R-squared value of approximately 0.79, indicating that it explains about 79% of the variance in `fare_amount`. The RMSE of around 3.06 suggests that, on average, the model's predictions are off by about $3.06. Key insights from the coefficients:
-   `mean_duration` has the strongest positive impact on `fare_amount`, followed by `mean_distance`.
-   `rush_hour` has a small positive impact.
-   `passenger_count` has a very small positive impact.
-   `VendorID_2` (being Vendor 2) has a very small negative impact compared to Vendor 1.

## Technologies Used
-   Python
-   Pandas (for data manipulation and analysis)
-   NumPy (for numerical operations)
-   Matplotlib (for plotting and visualization)
-   Seaborn (for enhanced statistical data visualization)
-   Scikit-learn (for machine learning models and preprocessing: `train_test_split`, `LinearRegression`, `StandardScaler`, `metrics`)

## How to Run
1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Ensure Dataset Availability**: Make sure the `2017_Yellow_Taxi_Trip_Data.csv` file is accessible in the specified path (e.g., `/content/` if running in Google Colab).
3.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Run the Jupyter Notebook**:
    Open `your_notebook_name.ipynb` in a Jupyter environment (e.g., JupyterLab, Google Colab) and run all cells sequentially.
