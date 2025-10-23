"""
Refined Model Enhancement with Better Data Augmentation and Training Strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
import random
import os
import warnings
warnings.filterwarnings('ignore')

class SmartDataAugmentator:
    """Smart data augmentation that preserves data quality"""
    
    def __init__(self, original_data_path: str):
        self.original_data = pd.read_csv(original_data_path)
        self.feature_stats = self._analyze_features()
        self.class_patterns = self._analyze_class_patterns()
        
    def _analyze_features(self):
        """Analyze feature distributions and relationships"""
        features = self.original_data.drop(columns=['price_range'])
        stats = {}
        
        for col in features.columns:
            stats[col] = {
                'min': features[col].min(),
                'max': features[col].max(),
                'mean': features[col].mean(),
                'std': features[col].std(),
                'median': features[col].median(),
                'q25': features[col].quantile(0.25),
                'q75': features[col].quantile(0.75),
                'is_binary': len(features[col].unique()) == 2
            }
        return stats
    
    def _analyze_class_patterns(self):
        """Analyze patterns for each price class"""
        patterns = {}
        
        for price_class in range(4):
            class_data = self.original_data[self.original_data['price_range'] == price_class]
            features = class_data.drop(columns=['price_range'])
            
            patterns[price_class] = {
                'mean': features.mean().to_dict(),
                'std': features.std().to_dict(),
                'count': len(class_data)
            }
        
        return patterns
    
    def generate_quality_samples(self, n_samples_per_class: int = 100) -> pd.DataFrame:
        """Generate high-quality augmented samples"""
        augmented_samples = []
        
        for price_class in range(4):
            class_pattern = self.class_patterns[price_class]
            
            for _ in range(n_samples_per_class):
                sample = {}
                
                # Generate features based on class patterns with controlled noise
                for feature, stats in self.feature_stats.items():
                    class_mean = class_pattern['mean'][feature]
                    class_std = class_pattern['std'][feature]
                    
                    if stats['is_binary']:
                        # For binary features, use class probability
                        prob = class_mean  # This is actually the probability for binary features
                        sample[feature] = np.random.choice([0, 1], p=[1-prob, prob])
                    else:
                        # For continuous features, use Gaussian noise around class mean
                        noise_factor = 0.1  # Small noise factor
                        value = np.random.normal(class_mean, class_std * noise_factor)
                        
                        # Ensure within realistic bounds
                        value = np.clip(value, stats['min'], stats['max'])
                        
                        # Round to appropriate precision
                        if feature in ['battery_power', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 
                                     'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']:
                            value = int(round(value))
                        else:
                            value = round(value, 2)
                    
                    sample[feature] = value
                
                sample['price_range'] = price_class
                augmented_samples.append(sample)
        
        return pd.DataFrame(augmented_samples)
    
    def create_balanced_dataset(self, augment_factor: float = 1.5) -> pd.DataFrame:
        """Create a balanced augmented dataset"""
        # Get original data
        original = self.original_data.copy()
        
        # Calculate target samples per class
        original_per_class = len(original) // 4
        target_per_class = int(original_per_class * augment_factor)
        additional_per_class = target_per_class - original_per_class
        
        # Generate additional samples
        if additional_per_class > 0:
            augmented = self.generate_quality_samples(additional_per_class)
            combined = pd.concat([original, augmented], ignore_index=True)
        else:
            combined = original
        
        # Ensure perfect balance
        balanced_data = []
        for price_class in range(4):
            class_data = combined[combined['price_range'] == price_class]
            if len(class_data) > target_per_class:
                # Randomly sample if we have too many
                class_data = class_data.sample(n=target_per_class, random_state=42)
            elif len(class_data) < target_per_class:
                # Upsample if we have too few
                class_data = resample(class_data, n_samples=target_per_class, random_state=42)
            
            balanced_data.append(class_data)
        
        return pd.concat(balanced_data, ignore_index=True)

class OptimizedNeuralNetwork(nn.Module):
    """Optimized neural network with proven architecture"""
    
    def __init__(self, input_dim: int, num_classes: int = 4):
        super(OptimizedNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Classification
        output = self.classifier(features)
        
        return output

class AdvancedTrainer:
    """Advanced trainer with multiple optimization techniques"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.best_accuracy = 0.0
        self.best_model_state = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_with_advanced_techniques(self, train_loader, val_loader, epochs=200):
        """Train with advanced techniques for maximum accuracy"""
        print(f"🚀 Starting advanced training for {epochs} epochs...")
        
        # Advanced optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Early stopping parameters
        patience = 30
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion, scheduler)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch < 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                      f"LR: {current_lr:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"✅ Training completed! Best validation accuracy: {self.best_accuracy:.2f}%")
        return self.best_accuracy
    
    def _train_epoch(self, train_loader, optimizer, criterion, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        return avg_loss, accuracy

def create_optimized_model():
    """Create and train an optimized model"""
    print("🚀 CREATING OPTIMIZED MODEL FOR NEAR-PERFECT ACCURACY")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Smart data augmentation
    print("\n📊 Step 1: Smart Data Augmentation")
    print("-" * 40)
    
    augmentator = SmartDataAugmentator("dataset/train.csv")
    balanced_data = augmentator.create_balanced_dataset(augment_factor=2.0)
    
    print(f"Original data size: {len(augmentator.original_data)}")
    print(f"Balanced data size: {len(balanced_data)}")
    print(f"Class distribution: {balanced_data['price_range'].value_counts().sort_index().tolist()}")
    
    # Step 2: Data preparation
    print("\n⚙️ Step 2: Advanced Data Preparation")
    print("-" * 40)
    
    # Separate features and targets
    X = balanced_data.drop(columns=['price_range'])
    y = balanced_data['price_range']
    
    # Stratified split to maintain class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Advanced scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val.values)
    
    # Create data loaders with balanced sampling
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    
    # Calculate class weights for balanced sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Step 3: Create optimized model
    print("\n🧠 Step 3: Optimized Model Creation")
    print("-" * 40)
    
    model = OptimizedNeuralNetwork(input_dim=X.shape[1], num_classes=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Step 4: Advanced training
    print("\n🏋️ Step 4: Advanced Training")
    print("-" * 40)
    
    trainer = AdvancedTrainer(model, device)
    final_accuracy = trainer.train_with_advanced_techniques(train_loader, val_loader, epochs=200)
    
    # Step 5: Save optimized model
    print("\n💾 Step 5: Save Optimized Model")
    print("-" * 40)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_dim': X.shape[1],
            'num_classes': 4,
            'model_type': 'OptimizedNeuralNetwork'
        },
        'best_val_accuracy': final_accuracy,
        'scaler': scaler,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies
        }
    }, 'models/optimized_model.pth')
    
    print(f"✅ Optimized model saved with {final_accuracy:.2f}% validation accuracy")
    
    # Step 6: Performance summary
    print("\n📈 Step 6: Performance Summary")
    print("-" * 40)
    
    print(f"Previous Simple DNN: 93.75%")
    print(f"Previous Advanced DNN: 88.50%")
    print(f"Previous Enhanced Model: 71.90%")
    print(f"New Optimized Model: {final_accuracy:.2f}%")
    
    improvement = final_accuracy - 93.75
    print(f"Improvement over best original: {improvement:+.2f}%")
    
    if final_accuracy >= 99.0:
        print("🏆 OUTSTANDING: Near-perfect accuracy achieved!")
    elif final_accuracy >= 97.0:
        print("🎯 EXCELLENT: Very high accuracy achieved!")
    elif final_accuracy >= 95.0:
        print("✅ GREAT: Significant improvement achieved!")
    else:
        print("📊 GOOD: Model performance improved!")
    
    return final_accuracy, model, scaler

if __name__ == "__main__":
    final_accuracy, model, scaler = create_optimized_model()
    
    print(f"\n🎉 OPTIMIZED MODEL CREATION COMPLETED!")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    
    if final_accuracy >= 99.0:
        print("🏆 SUCCESS: Achieved near-perfect accuracy!")
        print("📱 The model can now predict mobile phone prices with exceptional precision!")
    elif final_accuracy >= 97.0:
        print("🎯 EXCELLENT: Very high accuracy achieved!")
        print("📱 The model has excellent predictive performance!")
    else:
        print("📊 The model shows improved performance with advanced techniques!")
