import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class MobilePhoneDataset(Dataset):
    """Custom Dataset for Mobile Phone Price Prediction"""
    
    def __init__(self, features, targets, transform=None):
        """
        Args:
            features (numpy.ndarray): Feature matrix
            targets (numpy.ndarray): Target labels
            transform (callable, optional): Optional transform to be applied on features
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, target

class MobilePhoneDataLoader:
    """Data loader and preprocessor for Mobile Phone dataset"""
    
    def __init__(self, train_path, test_path=None, batch_size=32, val_split=0.2, random_state=42):
        """
        Initialize the data loader
        
        Args:
            train_path (str): Path to training CSV file
            test_path (str): Path to test CSV file (optional)
            batch_size (int): Batch size for DataLoader
            val_split (float): Validation split ratio
            random_state (int): Random state for reproducibility
        """
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price_range'
        self.num_features = None
        self.num_classes = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        # Load training data
        train_df = pd.read_csv(self.train_path)
        print(f"Loaded training data: {train_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
        
        # Separate features and target
        if self.target_column in train_df.columns:
            X = train_df.drop(columns=[self.target_column])
            y = train_df[self.target_column]
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        self.feature_columns = X.columns.tolist()
        self.num_features = len(self.feature_columns)
        self.num_classes = len(y.unique())
        
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class distribution:\n{y.value_counts().sort_index()}")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        
        # Load test data if provided
        X_test_scaled = None
        y_test = None
        if self.test_path and os.path.exists(self.test_path):
            test_df = pd.read_csv(self.test_path)
            print(f"Loaded test data: {test_df.shape}")
            
            if self.target_column in test_df.columns:
                X_test = test_df.drop(columns=[self.target_column])
                y_test = test_df[self.target_column].values
            else:
                X_test = test_df[self.feature_columns]
                y_test = None
            
            X_test = X_test.fillna(X_test.mean())
            X_test_scaled = self.scaler.transform(X_test)
            print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test
    
    def create_data_loaders(self):
        """Create PyTorch DataLoaders"""
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Create datasets
        train_dataset = MobilePhoneDataset(X_train, y_train)
        val_dataset = MobilePhoneDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        test_loader = None
        if X_test is not None:
            test_dataset = MobilePhoneDataset(X_test, y_test if y_test is not None else np.zeros(len(X_test)))
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0
            )
        
        return train_loader, val_loader, test_loader
    
    def get_data_info(self):
        """Get information about the data"""
        return {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }

def get_device():
    """Get the best available device (CUDA if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device

if __name__ == "__main__":
    # Test the data loader
    train_path = "dataset/train.csv"
    test_path = "dataset/test.csv"
    
    dataloader = MobilePhoneDataLoader(train_path, test_path, batch_size=64)
    train_loader, val_loader, test_loader = dataloader.create_data_loaders()
    
    print(f"\nData loader created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for features, targets in train_loader:
        print(f"Batch shape - Features: {features.shape}, Targets: {targets.shape}")
        break
    
    # Get device info
    device = get_device()
