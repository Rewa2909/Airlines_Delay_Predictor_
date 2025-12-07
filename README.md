# Airline Delay Predictor

## Project Overview

This project focuses on **predicting the total arrival delay (`arr_delay`)** of commercial airline flights using historical data on flight volumes, cancellation rates, and detailed delay causes (carrier, weather, NAS, etc.). The goal is to build a robust regression model capable of accurately forecasting aggregated delay minutes for specific carriers at specific airports.

The project involves critical steps in data preprocessing for machine learning, including handling highly skewed target variables and using advanced encoding techniques for high-cardinality categorical features.

## Data Source

The analysis is based on the **`Airline_Delay_Cause.csv`** dataset, which contains monthly-level statistics for U.S. air carriers operating at various airports.

| Column | Description |
| :--- | :--- |
| `year`, `month` | Temporal features. |
| `carrier`, `airport` | Categorical identifiers for carrier and airport. |
| `arr_flights` | Total number of arriving flights. |
| `arr_del15` | Total number of delayed flights (by $\ge 15$ minutes). |
| **`arr_delay`** | **(Target)** Total arrival delay in minutes. |
| `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay` | Minutes of delay attributed to specific causes. |

## Key Methodologies

### 1\. Target Variable Transformation (Critical Step)

The target variable, **`arr_delay`**, was found to be highly **right-skewed** (Skewness $\approx 9.12$), which violates the assumptions of many linear models and can lead to unstable training.

  * **Action:** The target was transformed using the **$\mathbf{\text{log1p}}$ function** ($\ln(1 + x)$).
  * **Result:** This dramatically reduced the skewness ($\approx -1.13$), resulting in a distribution much closer to normal, which improves model performance and stability.

$$\mathbf{y}_{\text{log}} = \text{np.log1p}(\mathbf{arr\_delay})$$

### 2\. Feature Engineering & Encoding

High-cardinality features like `carrier`, `carrier_name`, `airport`, and `airport_name` were handled using the **Target Encoding** technique.

  * **Target Encoding:** Replaced each categorical value with the **mean of the log-transformed target ($\mathbf{y}_{\text{log}}$)** for that category.
  * **Advantage:** This method captures the relationship between the category and the target while creating fewer features than One-Hot Encoding and is necessary because the target is transformed.

### 3\. Machine Learning Model

*(You will fill in this section after selecting and training a model)*

  * **Model Used:** \[e.g., Gradient Boosting Regressor, Random Forest, or Ridge Regression]
  * **Evaluation Metric:** \[e.g., Root Mean Squared Error (RMSE) on the original scale, or Mean Absolute Error (MAE)]

### 4\. Prediction Reversion

After the model trained on $\mathbf{y}_{\text{log}}$ provides a prediction, the result is converted back to the original minutes scale using the $\mathbf{\text{expm1}}$ function:

$$\mathbf{\text{Predicted Delay (minutes)}} = \text{np.expm1}(\mathbf{\text{Predicted Log Delay}})$$

## Project Structure

  * `Airline_Delay_Predictor.ipynb`: The main Jupyter Notebook containing all data loading, cleaning, transformation, encoding, model training, and evaluation steps.
  * `Airline_Delay_Cause.csv`: The primary dataset used for this project.
  * `README.md`: This file.

## Technologies Used

  * **Python**
  * **Pandas** (for data manipulation)
  * **NumPy** (for mathematical transformations, specifically $\text{log1p}$ and $\text{expm1}$)
  * **Scikit-learn** (for modeling and splitting data)
  * **Category Encoders** (for Target Encoding)
  * **Matplotlib / Seaborn** (for data visualization and EDA)

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd Airline-Delay-Predictor
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn category_encoders
    ```
3.  **Run the Notebook:** Open and execute the cells in `Airline_Delay_Predictor.ipynb` using Jupyter Lab or VS Code.
