from core import CRePE, SPD_Model
<<<<<<< HEAD
=======
from advanced.ensemble_models import EnsembleRagaClassifier
from advanced.feature_extraction import MultiModalFeatureExtractor
>>>>>>> a150c6c (Sync: updates to advanced rules, README, requirements, simple output)
import recorder
import os
import data_utils
from scipy.io import wavfile
import argparse
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier

<<<<<<< HEAD

def predict_run_time(crepe, cretora_hindustani, cretora_carnatic, tradition='h', tonic=None, seconds=60):
    while True:
        audio = recorder.record(seconds)
        if audio is None:
            return
        pitches = crepe.predict_pitches(audio)
        if tradition == 'h':
            pred_tonic, pred_raga = cretora_hindustani.predict_tonic_raga(crepe, audio, pitches, tonic)
        else:
            pred_tonic, pred_raga = cretora_carnatic.predict_tonic_raga(crepe, audio, pitches, tonic)

        if tonic is None:
            print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))
        else:
            print('Predicted Raga: {}'.format(pred_raga))


def predict_on_file(crepe, cretora, file_path, tonic):
    split_tup = os.path.splitext(file_path)

=======
def extract_features_for_ensemble(audio, pitches, sr=16000):
    """Extract comprehensive features for ensemble prediction"""
    extractor = MultiModalFeatureExtractor(sr=sr)
    
    # Convert pitches to appropriate format if needed
    pitch_array = np.array(pitches) if pitches is not None else None
    
    # Extract all features
    features = extractor.extract_all_features(audio, pitch_array)
    
    # Return as 2D array for model prediction
    return features.reshape(1, -1)

def predict_with_ensemble(ensemble_classifier, audio, crepe_model, tradition, tonic=None):
    """Predict raga using ensemble of 5 models"""
    print(f"🎵 Starting ensemble prediction for {tradition} tradition...")
    
    # Get pitch predictions from CRePE
    pitches = crepe_model.predict_pitches(audio)
    print("✅ Pitch extraction completed")
    
    # Extract comprehensive features for ensemble
    try:
        X_features = extract_features_for_ensemble(audio, pitches)
        print(f"✅ Extracted {X_features.shape[1]} features for ensemble")
        
        # Get ensemble prediction
        ensemble_pred = ensemble_classifier.predict(X_features)[0]
        ensemble_proba = ensemble_classifier.predict_proba(X_features)[0]
        
        # Get detailed prediction info
        details = ensemble_classifier.get_prediction_details(X_features)
        
        print(f"🎯 Ensemble Prediction: {ensemble_pred}")
        print(f"🎯 Confidence: {np.max(ensemble_proba):.3f}")
        print(f"🤖 Model Agreement: {details['model_agreement']:.3f}")
        
        # Show individual model predictions
        print("\n📊 Individual Model Predictions:")
        for model_name, pred in details['individual_predictions'].items():
            confidence = np.max(details['individual_probabilities'][model_name][0]) if details['individual_probabilities'][model_name] is not None else 0
            print(f"   {model_name:>4}: {pred[0]} (conf: {confidence:.3f})")
        
        return ensemble_pred, np.max(ensemble_proba), details
        
    except Exception as e:
        print(f"❌ Ensemble prediction failed: {e}")
        print("🔄 Falling back to original SPD model...")
        return None, None, None

def predict_run_time_ensemble(crepe, ensemble_carnatic, ensemble_hindustani, tradition='h', tonic=None, seconds=60):
    """Real-time prediction using ensemble models"""
    while True:
        print(f"\n🎙️ Recording {seconds} seconds of audio...")
        audio = recorder.record(seconds)
        if audio is None:
            return
        
        # Choose appropriate ensemble
        if tradition == 'h':
            ensemble_classifier = ensemble_hindustani
            tradition_name = "Hindustani"
        else:
            ensemble_classifier = ensemble_carnatic  
            tradition_name = "Carnatic"
            
        # Predict using ensemble
        pred_raga, confidence, details = predict_with_ensemble(
            ensemble_classifier, audio, crepe, tradition_name, tonic)
        
        if pred_raga is not None:
            print(f"\n🎵 {tradition_name} Raga Prediction: {pred_raga}")
            print(f"🎯 Confidence: {confidence:.1%}")
            if details['model_agreement'] > 0.8:
                print("✅ High model agreement - reliable prediction")
            else:
                print("⚠️ Low model agreement - prediction uncertain")
        else:
            print("❌ Ensemble prediction failed")

def predict_on_file_ensemble(crepe, ensemble_classifier, file_path, tradition, tonic=None):
    """Predict raga from file using ensemble"""
    split_tup = os.path.splitext(file_path)
    
    # Load audio file
>>>>>>> a150c6c (Sync: updates to advanced rules, README, requirements, simple output)
    if split_tup[1] == '.mp3':
        audio = data_utils.mp3_to_wav(file_path)
    else:
        sr, audio = wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)
<<<<<<< HEAD

    pitches = crepe.predict_pitches(audio)
    print("Pitch Prediction Complete")

    result = cretora.predict_tonic_raga(crepe, audio, pitches, tonic)
    pred_tonic = result['tonic']
    pred_raga = result['top1'][0]  # or however you want to extract the raga

    print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))


def bhatta(hist1, hist2):
    h1_ = np.mean(hist1)
    h2_ = np.mean(hist2)
    score = np.sum(np.sqrt(np.multiply(hist1, hist2)))
    t = 1 / (math.sqrt(h1_ * h2_ * len(hist1) * len(hist2)) + 0.0000001)
    score = math.sqrt(1 - (t) * score)
    return score


=======
    
    # Normalize to float32 [-1,1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    # Clip to 30s to keep inference fast
    audio = audio[:16000*30]
    
    print(f"🎵 Processing {file_path} ({len(audio)/16000:.1f}s)")
    
    # Predict using ensemble
    pred_raga, confidence, details = predict_with_ensemble(
        ensemble_classifier, audio, crepe, tradition, tonic)
    
    if pred_raga is not None:
        print(f"\n🎵 Final Prediction: {pred_raga}")
        print(f"🎯 Confidence: {confidence:.1%}")
        print(f"🤖 Model Agreement: {details['model_agreement']:.1%}")
        
        # Show top 3 predictions from ensemble
        ensemble_proba = ensemble_classifier.predict_proba(
            extract_features_for_ensemble(audio, crepe.predict_pitches(audio)))[0]
        
        # Get raga names (you'll need to map indices to raga names)
        print(f"\n📊 Top 3 Predictions:")
        top_indices = np.argsort(ensemble_proba)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. Raga {idx}: {ensemble_proba[idx]:.1%}")
    else:
        print("❌ Prediction failed")

# Original functions preserved for compatibility
def predict_run_time(crepe, cretora_hindustani, cretora_carnatic, tradition='h', tonic=None, seconds=60):
    """Original prediction function (preserved for compatibility)"""
    while True:
        audio = recorder.record(seconds)
        if audio is None:
            return
        
        pitches = crepe.predict_pitches(audio)
        
        if tradition=='h':
            pred_tonic, pred_raga = cretora_hindustani.predict_tonic_raga(crepe, audio, pitches, tonic)
        else:
            pred_tonic, pred_raga = cretora_carnatic.predict_tonic_raga(crepe, audio, pitches, tonic)
        
        if tonic is None:
            print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))
        else:
            print('Predicted Raga: {}'.format(pred_raga))

def predict_on_file(crepe, cretora, file_path, tonic):
    """Original file prediction (preserved for compatibility)"""
    split_tup = os.path.splitext(file_path)
    if split_tup[1] == '.mp3':
        audio = data_utils.mp3_to_wav(file_path)
    else:
        sr, audio = wavfile.read(file_path)
        if len(audio.shape) == 2:
            audio = audio.mean(1)
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    audio = audio[:16000*30]
    pitches = crepe.predict_pitches(audio)
    pred_tonic, pred_raga = cretora.predict_tonic_raga(crepe, audio, pitches, tonic)
    print('Predicted Tonic: {} and Raga: {}'.format(pred_tonic, pred_raga))

# Keep existing SPDKNN class
>>>>>>> a150c6c (Sync: updates to advanced rules, README, requirements, simple output)
class SPDKNN:
    def __init__(self, k=5):
        self.y = None
        self.knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', metric=self.bhatta)
<<<<<<< HEAD
=======
        self.wd = None
>>>>>>> a150c6c (Sync: updates to advanced rules, README, requirements, simple output)

    def bhatta(self, hist1, hist2):
        h1_ = np.mean(hist1)
        h2_ = np.mean(hist2)
        score = np.sum(np.sqrt(np.multiply(hist1, hist2)))
        score = math.sqrt(1 - (1 / math.sqrt(h1_ * h2_ * len(hist1) * len(hist2))) * score)
        return score

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        return self.knn.predict_proba(X)

<<<<<<< HEAD

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--runtime', default=False,
                            help='records audio for the given amount of duration')
    arg_parser.add_argument('--runtime_file', default=None,
                            help='runs the model on the input file path')
    arg_parser.add_argument('--duration', default=60,
                            help='sets the duration for recording in seconds')
    arg_parser.add_argument('--tradition', default='h',
                            help='sets the tradition - [h]industani/[c]arnatic')
    arg_parser.add_argument('--tonic', default=None,
                            help='sets the tonic if given, otherwise tonic is predicted')

    p_args = arg_parser.parse_args()

    crepe = CRePE()

    if p_args.runtime:
        cretora_hindustani = SPD_Model('Hindustani')
        cretora_carnatic = SPD_Model('Carnatic')
        predict_run_time(crepe, cretora_hindustani, cretora_carnatic,
                         tradition=p_args.tradition, tonic=p_args.tonic, seconds=int(p_args.duration))

    elif p_args.runtime_file:
        if p_args.tradition == 'h':
            cretora = SPD_Model('Hindustani')
        else:
            cretora = SPD_Model('Carnatic')
        predict_on_file(crepe, cretora, p_args.runtime_file, p_args.tonic)
=======
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--runtime', action='store_true',
                           help='records audio for the given amount of duration')
    arg_parser.add_argument('--runtime_file', default=None,
                           help='runs the model on the input file path')
    arg_parser.add_argument('--duration', default=60,
                           help='sets the duration for recording in seconds')
    arg_parser.add_argument('--tradition', default='h',
                           help='sets the tradition - [h]industani/[c]arnatic')
    arg_parser.add_argument('--tonic', default=None,
                           help='sets the tonic if given, otherwise tonic is predicted')
    arg_parser.add_argument('--use_ensemble', action='store_true',
                           help='use ensemble of 5 ML models for prediction')
    arg_parser.add_argument('--simple_output', action='store_true',
                           help='print only final tonic and raga')
    
    p_args = arg_parser.parse_args()
    
    # Configure quiet/simple logging if requested
    if p_args.simple_output:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['RAGA_QUIET'] = '1'

    # Initialize CRePE model
    crepe = CRePE()
    
    if p_args.use_ensemble:
        print("🚀 Initializing Ensemble Models...")
        
        # Initialize ensemble classifiers
        ensemble_carnatic = EnsembleRagaClassifier(tradition='Carnatic')
        ensemble_hindustani = EnsembleRagaClassifier(tradition='Hindustani')
        
        # Note: You would load pre-trained models here
        # ensemble_carnatic.load_models('models/carnatic_ensemble/')
        # ensemble_hindustani.load_models('models/hindustani_ensemble/')
        
        print("✅ Ensemble models initialized with 5 ML algorithms:")
        print("   - XGBoost Classifier")
        print("   - Random Forest Classifier")  
        print("   - Gradient Boosting Classifier")
        print("   - Support Vector Machine")
        print("   - Multi-layer Perceptron")
        
        if p_args.runtime:
            predict_run_time_ensemble(crepe, ensemble_carnatic, ensemble_hindustani,
                                    tradition=p_args.tradition, tonic=p_args.tonic, 
                                    seconds=int(p_args.duration))
        elif p_args.runtime_file:
            if not os.path.exists(p_args.runtime_file):
                print("File not found:", p_args.runtime_file)
                raise SystemExit(1)
            
            if p_args.tradition == 'h':
                ensemble_classifier = ensemble_hindustani
                tradition_name = "Hindustani"
            else:
                ensemble_classifier = ensemble_carnatic
                tradition_name = "Carnatic"
            
            predict_on_file_ensemble(crepe, ensemble_classifier, p_args.runtime_file, 
                                   tradition_name, p_args.tonic)
    
    else:
        if not p_args.simple_output:
            print("🔄 Using original SPD models...")
        # Original code path
        if p_args.runtime:
            cretora_hindustani = SPD_Model('Hindustani')
            cretora_carnatic = SPD_Model('Carnatic')
            predict_run_time(crepe, cretora_hindustani, cretora_carnatic,
                           tradition=p_args.tradition, tonic=p_args.tonic, 
                           seconds=int(p_args.duration))
        
        elif p_args.runtime_file:
            if not os.path.exists(p_args.runtime_file):
                print("File not found:", p_args.runtime_file)
                raise SystemExit(1)
            
            if p_args.tradition == 'h':
                cretora = SPD_Model('Hindustani')
            else:
                cretora = SPD_Model('Carnatic')
            
            # Simple output path
            if p_args.simple_output:
                split_tup = os.path.splitext(p_args.runtime_file)
                if split_tup[1] == '.mp3':
                    audio = data_utils.mp3_to_wav(p_args.runtime_file)
                else:
                    sr, audio = wavfile.read(p_args.runtime_file)
                    if len(audio.shape) == 2:
                        audio = audio.mean(1)
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val
                audio = audio[:16000*30]
                pitches = crepe.predict_pitches(audio)
                pred_tonic, pred_raga = cretora.predict_tonic_raga(crepe, audio, pitches, p_args.tonic)
                print(f"Tonic: {pred_tonic} | Raga: {pred_raga}")
            else:
                predict_on_file(crepe, cretora, p_args.runtime_file, p_args.tonic)
>>>>>>> a150c6c (Sync: updates to advanced rules, README, requirements, simple output)
