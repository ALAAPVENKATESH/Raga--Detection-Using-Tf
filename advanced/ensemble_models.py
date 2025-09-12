# advanced/ensemble_models.py
"""
Ensemble learning models for superior raga recognition accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class EnsembleRagaClassifier:
    """
    Advanced ensemble classifier combining multiple ML models for optimal accuracy
    """
    
    def __init__(self, tradition='Carnatic', famous_ragas=None):
        self.tradition = tradition
        self.famous_ragas = famous_ragas or []
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize individual models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all individual models in the ensemble"""
        
        # 1. XGBoost - Excellent for structured/tabular data
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # 2. Random Forest - Robust, handles overfitting well
        self.models['rf'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 3. Gradient Boosting - Sequential learning
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        )
        
        # 4. Support Vector Machine - Good for high-dimensional data
        self.models['svm'] = SVC(
            kernel='rbf',
            gamma='scale',
            C=1.0,
            probability=True,
            random_state=42
        )
        
        # 5. Neural Network - Can learn complex patterns
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=300,
            random_state=42
        )
        
        print(f"✅ Initialized {len(self.models)} models for {self.tradition} tradition")
    
    def train(self, X, y, validation_split=0.2):
        """Train all models and create ensemble"""
        
        print(f"🏋️ Training ensemble on {len(X)} samples with {X.shape[1]} features...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        n_val = int(len(X_scaled) * validation_split)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        # Train individual models in parallel
        trained_models = {}
        model_scores = {}
        
        def train_single_model(name_model_pair):
            name, model = name_model_pair
            print(f"  Training {name}...")
            
            try:
                # Train model
                if name == 'mlp':
                    # Neural network may need more iterations
                    model.set_params(max_iter=500)
                
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_score = model.score(X_val, y_val)
                print(f"  ✅ {name}: validation accuracy = {val_score:.3f}")
                
                # Calibrate probabilities
                calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                calibrated_model.fit(X_train, y_train)
                
                return name, calibrated_model, val_score
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                return name, None, 0.0
        
        # Parallel training
        with ThreadPoolExecutor(max_workers=min(5, len(self.models))) as executor:
            results = list(executor.map(train_single_model, self.models.items()))
        
        # Collect results
        for name, model, score in results:
            if model is not None:
                trained_models[name] = model
                model_scores[name] = score
        
        self.models = trained_models
        print(f"✅ Successfully trained {len(trained_models)} models")
        print(f"📊 Model scores: {model_scores}")
        
        # Create weighted ensemble based on validation performance
        self._create_weighted_ensemble(model_scores)
        
        # Final ensemble training
        if self.ensemble:
            print("🎯 Training final ensemble...")
            self.ensemble.fit(X_train, y_train)
            ensemble_score = self.ensemble.score(X_val, y_val)
            print(f"✅ Ensemble validation accuracy: {ensemble_score:.3f}")
        
        self.is_trained = True
        return model_scores
    
    def _create_weighted_ensemble(self, model_scores):
        """Create weighted voting ensemble based on individual model performance"""
        
        if len(self.models) < 2:
            print("⚠️  Not enough models for ensemble, using single model")
            return
        
        # Create estimators list with models that performed reasonably well
        estimators = []
        weights = []
        
        for name, model in self.models.items():
            score = model_scores.get(name, 0.0)
            if score > 0.3:  # Only include models with reasonable performance
                estimators.append((name, model))
                weights.append(score)
        
        if len(estimators) >= 2:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Create weighted voting classifier
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability-based voting
            )
            
            print(f"🎯 Created ensemble with {len(estimators)} models")
            print(f"📊 Model weights: {dict(zip([name for name, _ in estimators], weights))}")
        else:
            print("⚠️  Not enough good models for ensemble")
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.ensemble:
            return self.ensemble.predict(X_scaled)
        elif self.models:
            # Fallback to best single model
            best_model = list(self.models.values())[0]
            return best_model.predict(X_scaled)
        else:
            raise ValueError("No trained models available")
    
    def predict_proba(self, X):
        """Get prediction probabilities from ensemble"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.ensemble:
            return self.ensemble.predict_proba(X_scaled)
        elif self.models:
            # Average probabilities from all models
            all_probas = []
            for model in self.models.values():
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_scaled)
                    all_probas.append(probas)
            
            if all_probas:
                return np.mean(all_probas, axis=0)
            else:
                raise ValueError("No models with probability prediction available")
        else:
            raise ValueError("No trained models available")
    
    def get_prediction_details(self, X):
        """Get detailed predictions from all models"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        results = {
            'ensemble_prediction': None,
            'ensemble_probabilities': None,
            'individual_predictions': {},
            'individual_probabilities': {},
            'model_agreement': 0.0
        }
        
        # Get predictions from individual models
        individual_preds = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
                
                results['individual_predictions'][name] = pred
                results['individual_probabilities'][name] = proba
                individual_preds.append(pred)
                
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
        
        # Calculate model agreement
        if len(individual_preds) > 1:
            # Agreement is percentage of models that agree with majority vote
            pred_array = np.array(individual_preds)
            majority_votes = []
            
            for i in range(pred_array.shape[1]):  # For each sample
                sample_preds = pred_array[:, i]
                unique, counts = np.unique(sample_preds, return_counts=True)
                majority_vote = unique[np.argmax(counts)]
                majority_votes.append(majority_vote)
                
            agreement_scores = []
            for i in range(pred_array.shape[1]):
                sample_preds = pred_array[:, i]
                agreement = np.mean(sample_preds == majority_votes[i])
                agreement_scores.append(agreement)
                
            results['model_agreement'] = np.mean(agreement_scores)
        
        # Get ensemble predictions
        if self.ensemble:
            results['ensemble_prediction'] = self.ensemble.predict(X_scaled)
            results['ensemble_probabilities'] = self.ensemble.predict_proba(X_scaled)
        
        return results
    
    def evaluate_models(self, X, y, cv_folds=5):
        """Evaluate individual models using cross-validation"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        X_scaled = self.scaler.transform(X)
        results = {}
        
        print(f"🔍 Evaluating models with {cv_folds}-fold cross-validation...")
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
                results[name] = {
                    'mean_accuracy': np.mean(scores),
                    'std_accuracy': np.std(scores),
                    'all_scores': scores
                }
                print(f"  {name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                
            except Exception as e:
                print(f"  {name}: evaluation failed - {e}")
                results[name] = {'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'error': str(e)}
        
        # Evaluate ensemble if available
        if self.ensemble:
            try:
                ensemble_scores = cross_val_score(self.ensemble, X_scaled, y, cv=cv_folds, scoring='accuracy')
                results['ensemble'] = {
                    'mean_accuracy': np.mean(ensemble_scores),
                    'std_accuracy': np.std(ensemble_scores),
                    'all_scores': ensemble_scores
                }
                print(f"  ensemble: {np.mean(ensemble_scores):.3f} ± {np.std(ensemble_scores):.3f}")
                
            except Exception as e:
                print(f"  ensemble: evaluation failed - {e}")
        
        return results
    
    def save_models(self, model_dir):
        """Save all trained models"""
        
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{self.tradition}_{name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"💾 Saved {name} model to {model_path}")
        
        # Save ensemble
        if self.ensemble:
            ensemble_path = os.path.join(model_dir, f"{self.tradition}_ensemble_model.pkl")
            joblib.dump(self.ensemble, ensemble_path)
            print(f"💾 Saved ensemble model to {ensemble_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.tradition}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")
        
        print(f"✅ All models saved to {model_dir}")
    
    def load_models(self, model_dir):
        """Load previously trained models"""
        
        print(f"📂 Loading models from {model_dir}...")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{self.tradition}_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✅ Loaded scaler")
        
        # Load individual models
        loaded_models = {}
        for name in ['xgb', 'rf', 'gb', 'svm', 'mlp']:
            model_path = os.path.join(model_dir, f"{self.tradition}_{name}_model.pkl")
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    loaded_models[name] = model
                    print(f"✅ Loaded {name} model")
                except Exception as e:
                    print(f"❌ Failed to load {name} model: {e}")
        
        self.models = loaded_models
        
        # Load ensemble
        ensemble_path = os.path.join(model_dir, f"{self.tradition}_ensemble_model.pkl")
        if os.path.exists(ensemble_path):
            try:
                self.ensemble = joblib.load(ensemble_path)
                print("✅ Loaded ensemble model")
            except Exception as e:
                print(f"❌ Failed to load ensemble model: {e}")
        
        if self.models or self.ensemble:
            self.is_trained = True
            print(f"🎯 Successfully loaded {len(self.models)} individual models")
        else:
            print("⚠️  No models loaded")
    
    def feature_importance_analysis(self, feature_names=None):
        """Analyze feature importance across models"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before analyzing feature importance")
        
        importance_results = {}
        
        for name, model in self.models.items():
            try:
                # Get feature importance based on model type
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models (RF, XGB, GB)
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models (SVM, MLPClassifier might have this)
                    importance = np.mean(np.abs(model.coef_), axis=0)
                else:
                    # For models without direct feature importance
                    importance = None
                
                if importance is not None:
                    importance_results[name] = importance
                    
            except Exception as e:
                print(f"Could not get feature importance for {name}: {e}")
        
        # Combine importance scores
        if importance_results:
            # Average importance across models
            all_importances = list(importance_results.values())
            avg_importance = np.mean(all_importances, axis=0)
            
            # Create results dictionary
            results = {
                'average_importance': avg_importance,
                'individual_importances': importance_results
            }
            
            # Add feature names if provided
            if feature_names and len(feature_names) == len(avg_importance):
                feature_ranking = sorted(zip(feature_names, avg_importance), 
                                       key=lambda x: x[1], reverse=True)
                results['feature_ranking'] = feature_ranking
                
                print("🎯 Top 10 Most Important Features:")
                for i, (feature, importance) in enumerate(feature_ranking[:10]):
                    print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
            
            return results
        
        else:
            print("⚠️  No feature importance information available")
            return None

# Convenience functions for easy integration
def create_ensemble_classifier(tradition='Carnatic', famous_ragas=None):
    """Create and return ensemble classifier"""
    return EnsembleRagaClassifier(tradition, famous_ragas)

def train_ensemble_from_data(X, y, tradition='Carnatic', famous_ragas=None, validation_split=0.2):
    """Train ensemble classifier from data"""
    classifier = EnsembleRagaClassifier(tradition, famous_ragas)
    model_scores = classifier.train(X, y, validation_split)
    return classifier, model_scores

if __name__ == "__main__":
    # Demo the ensemble classifier
    print("🚀 Testing Ensemble Raga Classifier")
    
    # Generate dummy data for testing
    n_samples = 1000
    n_features = 50
    n_classes = 15  # Number of famous ragas
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create and train ensemble
    ensemble = EnsembleRagaClassifier('Carnatic')
    model_scores = ensemble.train(X, y)
    
    # Test predictions
    test_X = np.random.randn(10, n_features)
    predictions = ensemble.predict(test_X)
    probabilities = ensemble.predict_proba(test_X)
    
    print(f"\n✅ Test complete!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test detailed predictions
    details = ensemble.get_prediction_details(test_X[:1])
    print(f"Model agreement: {details['model_agreement']:.3f}")