"""
Evaluation and Prediction utilities for Mobile Phone Price Prediction
"""

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mobile_phone_predictor import MobilePhonePricePredictor, AdvancedMobilePhonePricePredictor

class MobilePhonePredictor:
    """Class for making predictions with trained models"""
    
    def __init__(self, model_path, scaler_path=None):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        self.load_model(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model architecture info
        arch_info = checkpoint['model_architecture']
        
        # Try to determine model type from state_dict keys or model_type
        state_dict = checkpoint['model_state_dict']
        has_attention = any('attention' in key for key in state_dict.keys())
        model_type = arch_info.get('model_type', '')
        
        # Create model based on architecture
        if has_attention or 'Advanced' in model_type:
            # Advanced model
            self.model = AdvancedMobilePhonePricePredictor(
                input_dim=arch_info['input_dim'],
                num_classes=arch_info['num_classes'],
                hidden_dims=[512, 256, 128],
                dropout_rate=0.3
            )
        else:
            # Simple model
            self.model = MobilePhonePricePredictor(
                input_dim=arch_info['input_dim'],
                num_classes=arch_info['num_classes'],
                hidden_dims=[256, 128, 64],
                dropout_rate=0.3
            )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Device: {self.device}")
    
    def load_scaler(self, scaler_path):
        """Load feature scaler"""
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    
    def create_scaler_from_data(self, train_data_path):
        """Create and fit scaler from training data"""
        train_df = pd.read_csv(train_data_path)
        
        # Remove target column if present
        if 'price_range' in train_df.columns:
            features = train_df.drop(columns=['price_range'])
        else:
            features = train_df
        
        self.feature_columns = features.columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        print("Scaler created and fitted from training data")
        return self.scaler
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Can be dict, list, numpy array, or pandas DataFrame
        
        Returns:
            torch.Tensor: Preprocessed tensor ready for model
        """
        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            if self.feature_columns:
                input_df = pd.DataFrame([input_data], columns=self.feature_columns)
            else:
                input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, np.ndarray):
            if self.feature_columns:
                input_df = pd.DataFrame(input_data.reshape(1, -1), columns=self.feature_columns)
            else:
                input_df = pd.DataFrame(input_data.reshape(1, -1))
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data must be dict, list, numpy array, or pandas DataFrame")
        
        # Handle missing values
        input_df = input_df.fillna(input_df.mean())
        
        # Scale features if scaler is available
        if self.scaler is not None:
            scaled_features = self.scaler.transform(input_df)
        else:
            scaled_features = input_df.values
        
        # Convert to tensor
        tensor = torch.FloatTensor(scaled_features).to(self.device)
        
        return tensor
    
    def predict(self, input_data, return_probabilities=False):
        """
        Make prediction
        
        Args:
            input_data: Input features
            return_probabilities (bool): Whether to return class probabilities
        
        Returns:
            int or tuple: Predicted class (and probabilities if requested)
        """
        # Preprocess input
        input_tensor = self.preprocess_input(input_data)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        
        if return_probabilities:
            probs = probabilities.cpu().numpy()[0]
            return predicted_class, probs
        else:
            return predicted_class
    
    def predict_batch(self, input_data, return_probabilities=False):
        """
        Make batch predictions
        
        Args:
            input_data: Batch of input features (DataFrame or numpy array)
            return_probabilities (bool): Whether to return class probabilities
        
        Returns:
            numpy.ndarray: Predicted classes (and probabilities if requested)
        """
        # Preprocess input
        if isinstance(input_data, pd.DataFrame):
            input_tensor = self.preprocess_input(input_data)
        else:
            # For numpy array, process each row
            all_predictions = []
            all_probabilities = []
            
            for row in input_data:
                pred, prob = self.predict(row, return_probabilities=True)
                all_predictions.append(pred)
                all_probabilities.append(prob)
            
            predictions = np.array(all_predictions)
            probabilities = np.array(all_probabilities)
            
            if return_probabilities:
                return predictions, probabilities
            else:
                return predictions
        
        # Make batch prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        
        if return_probabilities:
            probs = probabilities.cpu().numpy()
            return predicted_classes, probs
        else:
            return predicted_classes
    
    def get_feature_importance(self, input_data, target_class=None):
        """
        Get feature importance using gradient-based method
        
        Args:
            input_data: Input features
            target_class (int): Target class for importance calculation
        
        Returns:
            numpy.ndarray: Feature importance scores
        """
        input_tensor = self.preprocess_input(input_data)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        # If target class not specified, use predicted class
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1)
        
        # Backward pass
        target_output = outputs[0, target_class]
        target_output.backward()
        
        # Get gradients
        importance = torch.abs(input_tensor.grad).cpu().numpy()[0]
        
        return importance

def create_sample_prediction():
    """Create a sample prediction example"""
    print("Creating sample prediction...")
    
    # Sample mobile phone features
    sample_phone = {
        'battery_power': 1500,
        'blue': 1,
        'clock_speed': 2.0,
        'dual_sim': 1,
        'fc': 5,
        'four_g': 1,
        'int_memory': 32,
        'm_dep': 0.8,
        'mobile_wt': 150,
        'n_cores': 4,
        'pc': 12,
        'px_height': 1080,
        'px_width': 1920,
        'ram': 3000,
        'sc_h': 15,
        'sc_w': 8,
        'talk_time': 18,
        'three_g': 1,
        'touch_screen': 1,
        'wifi': 1
    }
    
    # Try to load a model and make prediction
    model_files = [
        "models/advanced_dnn_model.pth",
        "models/simple_dnn_model.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Using model: {model_file}")
            
            # Create predictor
            predictor = MobilePhonePredictor(model_file)
            
            # Create scaler from training data
            if os.path.exists("dataset/train.csv"):
                predictor.create_scaler_from_data("dataset/train.csv")
            
            # Make prediction
            predicted_class, probabilities = predictor.predict(sample_phone, return_probabilities=True)
            
            print("\nSample Phone Specifications:")
            for feature, value in sample_phone.items():
                print(f"  {feature}: {value}")
            
            print(f"\nPredicted Price Range: {predicted_class}")
            print("Class Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  Price Range {i}: {prob:.4f} ({prob*100:.2f}%)")
            
            # Price range descriptions
            price_descriptions = {
                0: "Low Cost (0-25%)",
                1: "Medium Low (25-50%)",
                2: "Medium High (50-75%)",
                3: "High Cost (75-100%)"
            }
            
            print(f"\nPredicted Category: {price_descriptions[predicted_class]}")
            break
    else:
        print("No trained models found. Please run main.py first to train models.")

def evaluate_model_on_test_data():
    """Evaluate trained model on test data"""
    print("Evaluating model on test data...")
    
    # Check for test data
    if not os.path.exists("dataset/test.csv"):
        print("Test data not found!")
        return
    
    # Load test data
    test_df = pd.read_csv("dataset/test.csv")
    print(f"Test data shape: {test_df.shape}")
    
    # Try to load a model
    model_files = [
        "models/advanced_dnn_model.pth",
        "models/simple_dnn_model.pth"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Using model: {model_file}")
            
            # Create predictor
            predictor = MobilePhonePredictor(model_file)
            
            # Create scaler from training data
            if os.path.exists("dataset/train.csv"):
                predictor.create_scaler_from_data("dataset/train.csv")
            
            # Prepare test data - remove id column and use only feature columns
            feature_columns = [
                'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',
                'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
                'talk_time', 'three_g', 'touch_screen', 'wifi'
            ]
            
            # Check if price_range exists (for labeled test data)
            if 'price_range' in test_df.columns:
                X_test = test_df[feature_columns]
                y_test = test_df['price_range'].values
                
                # Make predictions
                predictions, probabilities = predictor.predict_batch(X_test, return_probabilities=True)
                
                # Calculate accuracy
                accuracy = np.mean(predictions == y_test)
                print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                
                # Show some example predictions
                print("\nSample Predictions:")
                for i in range(min(5, len(predictions))):
                    print(f"  Sample {i+1}: True={y_test[i]}, Predicted={predictions[i]}, "
                          f"Confidence={probabilities[i][predictions[i]]:.3f}")
            else:
                # Just make predictions without ground truth
                X_test = test_df[feature_columns]
                predictions, probabilities = predictor.predict_batch(X_test, return_probabilities=True)
                
                print("Predictions made on test data:")
                pred_counts = np.bincount(predictions)
                for i, count in enumerate(pred_counts):
                    print(f"  Price Range {i}: {count} samples")
                
                # Show some sample predictions with details
                print("\nSample Predictions:")
                for i in range(min(5, len(predictions))):
                    print(f"  Sample {i+1} (ID: {test_df.iloc[i]['id']}): "
                          f"Predicted={predictions[i]}, "
                          f"Confidence={probabilities[i][predictions[i]]:.3f}")
                    # Show top features for this sample
                    sample_features = test_df.iloc[i][feature_columns]
                    print(f"    RAM: {sample_features['ram']}MB, "
                          f"Battery: {sample_features['battery_power']}mAh, "
                          f"Primary Cam: {sample_features['pc']}MP")
            
            break
    else:
        print("No trained models found. Please run main.py first to train models.")

if __name__ == "__main__":
    print("🔮 MOBILE PHONE PRICE PREDICTION - EVALUATION")
    print("=" * 60)
    
    # Create sample prediction
    create_sample_prediction()
    
    print("\n" + "="*60)
    
    # Evaluate on test data
    evaluate_model_on_test_data()
