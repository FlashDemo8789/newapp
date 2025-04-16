# Startup Success Prediction Model

This directory contains scripts for training and serving a machine learning model that predicts startup success based on financial and operational metrics.

## Training the Model

To train the startup success prediction model, run the following command:

```bash
python -m ml_infrastructure.training.train_startup_model
```

This will:
1. Load the startup data from `data/big_startups.json`
2. Preprocess the data (convert categorical variables to numeric, handle missing values)
3. Train a Random Forest classifier to predict funding outcomes
4. Register the trained model with the model registry

The model will be saved to the `model_store` directory.

## Serving the Model

To serve the trained model via a REST API, run:

```bash
python -m ml_infrastructure.training.serve_startup_model
```

This will:
1. Start a model server on port 5000
2. Load the trained startup success prediction model
3. Set up preprocessing and postprocessing functions
4. Provide a prediction endpoint at `/models/startup_success_predictor/predict`

## Testing the Model

To test the model with a sample prediction, run:

```bash
python -m ml_infrastructure.training.serve_startup_model --test
```

## API Usage

Once the model server is running, you can make predictions using HTTP requests:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "name": "TechVenture",
  "stage": "seed",
  "sector": "saas",
  "monthly_revenue": 50000,
  "annual_recurring_revenue": 600000,
  "lifetime_value_ltv": 8000,
  "gross_margin_percent": 70,
  "operating_margin_percent": 15,
  "burn_rate": 1.2,
  "runway_months": 18,
  "cash_on_hand_million": 2.5,
  "debt_ratio": 0.1,
  "financing_round_count": 1,
  "monthly_active_users": 12000
}' http://localhost:5000/models/startup_success_predictor/predict
```

## Required Fields

The following fields are required for prediction:
- `stage`: Funding stage ('seed', 'series_a', 'series_b', 'series_c', 'growth')
- `sector`: Industry sector (e.g., 'saas', 'fintech', 'healthtech')
- `monthly_revenue`: Monthly revenue in dollars
- `annual_recurring_revenue`: Annual recurring revenue in dollars
- `gross_margin_percent`: Gross margin percentage
- `burn_rate`: Burn rate (ratio of monthly expenses to revenue)
- `runway_months`: Months of runway remaining 