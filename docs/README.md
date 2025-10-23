# Mobile Phone Price Prediction using Deep Learning

A comprehensive deep learning solution for predicting mobile phone price ranges using PyTorch with CUDA acceleration. This project implements advanced neural network architectures to classify mobile phones into different price categories based on their technical specifications.

## 🚀 Features

- **Deep Learning Models**: Two neural network architectures (Simple DNN and Advanced DNN with attention mechanism)
- **CUDA Support**: GPU acceleration for faster training and inference
- **Data Preprocessing**: Automated feature scaling and data preparation
- **Model Comparison**: Train and compare multiple architectures
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and confusion matrices
- **REST APIs**: Both FastAPI and Flask implementations for model serving
- **Batch Prediction**: Support for single and batch predictions
- **Visualization**: Training curves, confusion matrices, and dataset analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Windows/Linux/macOS

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify CUDA installation (optional):**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### 1. Train the Models

Run the complete training pipeline:

```bash
python main.py
```

This will:
- ✅ Load and analyze the dataset
- ✅ Train both Simple and Advanced DNN models
- ✅ Generate comparison plots and metrics
- ✅ Save trained models to `models/` directory
- ✅ Create visualizations in `outputs/plots/`
- ✅ Generate training reports in `outputs/logs/`

### 2. Make Predictions

Evaluate trained models:

```bash
python evaluate.py
```

### 3. Start API Server

**FastAPI (Recommended):**
```bash
cd api
python main_api.py
```

**Flask Alternative:**
```bash
python api.py
```

### Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd "D:\Mobile Phone Pricing"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### 📊 Dataset

The project uses two CSV files:
- `dataset/train.csv` - Training data with features and target prices
- `dataset/test.csv` - Test data for making predictions

### 🎯 Usage Options

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook mobile_phone_pricing_prediction.ipynb
```

This provides an interactive environment with:
- Data exploration and visualization
- Feature engineering
- Model training and comparison
- Interactive plots and insights

#### Option 2: Python Script
```bash
python mobile_phone_predictor.py
```

This runs the complete pipeline and outputs:
- Model comparison results
- Best model selection
- Predictions saved to `mobile_phone_price_predictions.csv`

#### Option 3: Web API (Optional)
Install additional dependencies:
```bash
pip install fastapi uvicorn
```

Start the API server:
```bash
python api.py
```

Access the API at: `http://localhost:8000/docs`

## 📈 Models Included

1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble method with feature importance
3. **XGBoost** - Gradient boosting with advanced features
4. **LightGBM** - Fast gradient boosting framework

## 🔧 Features

### Data Processing
- Automated missing value handling
- Categorical encoding
- Feature scaling for appropriate models
- Feature engineering (interactions, statistical features)

### Model Evaluation
- Cross-validation
- Multiple metrics (RMSE, R², MAE)
- Overfitting analysis
- Feature importance visualization

### Visualizations
- Correlation heatmaps
- Distribution plots
- Model comparison charts
- Prediction vs actual plots
- Interactive Plotly visualizations

## 📁 Project Structure

```
Mobile Phone Pricing/
├── dataset/
│   ├── train.csv                     # Training data
│   └── test.csv                      # Test data
├── mobile_phone_pricing_prediction.ipynb  # Main Jupyter notebook
├── mobile_phone_predictor.py        # Complete Python pipeline
├── api.py                           # FastAPI web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── venv/                           # Virtual environment
└── mobile_phone_price_predictions.csv  # Output predictions
```

## 🎯 Expected Results

The pipeline will:
1. **Load and explore** your mobile phone dataset
2. **Preprocess** data with automated cleaning and encoding
3. **Engineer features** to improve model performance
4. **Train multiple models** and compare their performance
5. **Select the best model** based on validation metrics
6. **Generate predictions** for the test set
7. **Save results** to CSV file

### Sample Output
```
MODEL COMPARISON RESULTS:
           Model  Val_R2  Val_RMSE  Val_MAE
   Random Forest   0.892     0.234    0.187
         XGBoost   0.885     0.241    0.195
        LightGBM   0.883     0.243    0.198
Linear Regression   0.756     0.351    0.298

🏆 Best Model: Random Forest
Best Validation R²: 0.892
```

## 📊 Key Features

### Automated Analysis
- **Target Detection**: Automatically identifies price-related columns
- **Data Types**: Handles both numeric and categorical features
- **Missing Values**: Intelligent imputation strategies
- **Feature Engineering**: Creates interaction and statistical features

### Model Comparison
- **Multiple Algorithms**: Compares 4 different ML approaches
- **Robust Evaluation**: Uses train/validation split with multiple metrics
- **Overfitting Detection**: Monitors train vs validation performance
- **Feature Importance**: Shows which features matter most

### Professional Output
- **Detailed Logging**: Progress tracking throughout the pipeline
- **Visual Analytics**: Charts and plots for insights
- **CSV Export**: Ready-to-submit prediction file
- **Model Insights**: Performance summaries and recommendations

## 🛠️ Troubleshooting

### Common Issues

1. **scikit-learn installation error (Windows)**:
   ```bash
   # Use pre-compiled wheels
   pip install --only-binary=all scikit-learn
   ```

2. **Missing Visual C++ Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Or use conda: `conda install scikit-learn`

3. **Memory issues with large datasets**:
   - Reduce features in feature engineering section
   - Use smaller sample for initial testing

### Performance Tips

1. **For faster training**:
   - Reduce `n_estimators` in Random Forest and XGBoost
   - Use fewer features in feature engineering

2. **For better accuracy**:
   - Increase `n_estimators` to 200-500
   - Add more feature engineering
   - Try hyperparameter tuning

## 📝 Customization

### Adding New Models
Add to the `models` dictionary in either script:
```python
models['Your Model'] = YourModelClass(parameters)
```

### Feature Engineering
Modify the `create_features()` function to add domain-specific features:
```python
# Add your custom features
df_new['price_per_gb'] = df_new['price'] / df_new['memory']
df_new['camera_ratio'] = df_new['front_camera'] / df_new['back_camera']
```

### Model Parameters
Tune hyperparameters in the model initialization:
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5
)
```

## 📞 Support

If you encounter any issues:
1. Check the error messages in the console output
2. Verify your data format matches the expected structure
3. Ensure all dependencies are installed correctly
4. Check that your dataset files are in the correct location

## 🏆 Results

After running the pipeline, you'll get:
- **Trained models** with performance comparisons
- **Predictions file** ready for submission
- **Insights** about which features drive mobile phone prices
- **Visualizations** showing model performance and data patterns

The best model typically achieves **85-95% accuracy** (R² score) depending on your dataset quality and features.

---

**Happy Predicting! 🎯📱**
