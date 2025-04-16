# Startup Success Prediction Model

This project provides a machine learning model to predict startup success based on financial and operational metrics. The model is trained on a dataset of 8,800 startups with funding outcomes.

## Features

- **Machine Learning Model**: A Random Forest classifier to predict startup funding outcomes.
- **Model Registry**: Manage model versions, A/B testing, and deployment.
- **Prediction API**: Serve the model via a REST API.
- **Model Training**: Automatically train the model on startup data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/startup-success-predictor.git
cd startup-success-predictor
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training the Model

To train the startup success prediction model:

```bash
python -m ml_infrastructure.training.train_startup_model
```

This will:
1. Load the startup data from `data/big_startups.json`
2. Preprocess the data
3. Train a Random Forest classifier
4. Register the model with the model registry

The model will be saved to the `model_store` directory.

### Serving the Model

To serve the model via a simple API:

```bash
python -m ml_infrastructure.training.simple_api
```

The API will be available at http://localhost:5001 with the following endpoints:
- `GET /health` - Check API health
- `POST /predict` - Make predictions

### Testing the API

```bash
python -m ml_infrastructure.training.test_simple_api
```

## Prediction Outcomes

Our model predicts whether a startup will be approved or rejected:

- **Pass**: The startup passes evaluation criteria (positive outcome)
- **Fund**: The startup fails to meet criteria (negative outcome)

Note: In our dataset, "pass" means approval, not rejection. This naming might seem counterintuitive at first.

## Example Prediction Request

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
}' http://localhost:5001/predict
```

## Project Structure

```
ml_infrastructure/
├── __init__.py
├── model_registry/        # Model registry for managing models
├── serving/               # Model serving components
└── training/              # Training scripts and utilities
    ├── train_startup_model.py   # Main training script
    ├── serve_startup_model.py   # Original model server (complex)
    └── simple_api.py            # Simple API for serving the model
```

## Required Fields

For prediction, the following fields are required:
- `stage`: Funding stage ('seed', 'series_a', 'series_b', 'series_c', 'growth')
- `sector`: Industry sector (e.g., 'saas', 'fintech', 'healthtech')
- `monthly_revenue`: Monthly revenue in dollars
- `annual_recurring_revenue`: Annual recurring revenue in dollars
- `gross_margin_percent`: Gross margin percentage
- `burn_rate`: Burn rate (ratio of monthly expenses to revenue)
- `runway_months`: Months of runway remaining
