# advanced/feature_extraction.py
"""
Advanced multi-modal feature extraction for next-level raga recognition
"""

import numpy as np
import librosa
from scipy import signal
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class MultiModalFeatureExtractor:
    """Extract comprehensive features from audio and pitch data"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def extract_all_features(self, audio, pitches=None, tonic_idx=0):
        """Extract all types of features"""
        
        features = {}
        
        # 1. Spectral Features
        features.update(self._extract_spectral_features(audio))
        
        # 2. Pitch-based Features
        if pitches is not None:
            features.update(self._extract_pitch_features(pitches, tonic_idx))
        
        # 3. Temporal Features
        features.update(self._extract_temporal_features(audio))
        
        # 4. Harmonic Features
        features.update(self._extract_harmonic_features(audio))
        
        return self._flatten_features(features)
    
    def _extract_spectral_features(self, audio):
        """Extract spectral features"""
        features = {}
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, bins_per_octave=60)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        features['chroma_max'] = np.max(chroma, axis=1)
        
        # Spectral characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def _extract_pitch_features(self, pitches, tonic_idx=0):
        """Extract pitch-based features"""
        features = {}
        
        if len(pitches) == 0:
            return features
            
        # Convert to cents relative to tonic
        pitch_cents = self._to_cents_from_tonic(pitches, tonic_idx)
        
        # Basic pitch statistics
        features['pitch_mean'] = np.mean(pitch_cents)
        features['pitch_std'] = np.std(pitch_cents)
        features['pitch_range'] = np.max(pitch_cents) - np.min(pitch_cents)
        features['pitch_median'] = np.median(pitch_cents)
        
        # Interval analysis
        intervals = np.diff(pitch_cents)
        if len(intervals) > 0:
            features['interval_mean'] = np.mean(intervals)
            features['interval_std'] = np.std(intervals)
            features['ascending_ratio'] = np.sum(intervals > 0) / len(intervals)
            features['large_jumps_ratio'] = np.sum(np.abs(intervals) > 200) / len(intervals)
        
        # N-gram patterns (melodic motifs)
        if len(pitch_cents) >= 3:
            bigrams = self._extract_ngrams(pitch_cents, n=2)
            trigrams = self._extract_ngrams(pitch_cents, n=3)
            
            features['bigram_diversity'] = len(bigrams) / max(1, len(pitch_cents) - 1)
            features['trigram_diversity'] = len(trigrams) / max(1, len(pitch_cents) - 2)
            
            # Most common intervals
            if bigrams:
                common_intervals = [interval for interval, count in bigrams.most_common(3)]
                for i, interval in enumerate(common_intervals):
                    features[f'common_interval_{i}'] = interval
        
        # Phrase complexity
        features['phrase_complexity'] = self._calculate_phrase_complexity(pitch_cents)
        
        # Gamaka (ornament) intensity
        features['gamaka_intensity'] = self._estimate_gamaka_intensity(pitch_cents)
        
        return features
    
    def _extract_temporal_features(self, audio):
        """Extract temporal and rhythmic features"""
        features = {}
        
        # Tempo and beat analysis
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            features['tempo'] = tempo
            features['beat_strength'] = np.mean(librosa.onset.onset_strength(y=audio, sr=self.sr))
        except:
            features['tempo'] = 100.0  # Default
            features['beat_strength'] = 0.1
        
        # Energy dynamics
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['dynamic_range'] = np.max(rms) - np.min(rms)
        
        # Onset detection
        onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, units='time')
        if len(onsets) > 1:
            onset_intervals = np.diff(onsets)
            features['avg_onset_interval'] = np.mean(onset_intervals)
            features['onset_regularity'] = 1.0 / (np.std(onset_intervals) + 0.001)
        else:
            features['avg_onset_interval'] = 1.0
            features['onset_regularity'] = 1.0
        
        return features
    
    def _extract_harmonic_features(self, audio):
        """Extract harmonic content features"""
        features = {}
        
        # Harmonic vs percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        features['harmonic_ratio'] = np.sum(harmonic**2) / (np.sum(audio**2) + 1e-10)
        features['percussive_ratio'] = np.sum(percussive**2) / (np.sum(audio**2) + 1e-10)
        
        # Tonal centroid features
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sr)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        
        return features
    
    def _to_cents_from_tonic(self, pitches, tonic_idx):
        """Convert pitch sequence to cents relative to tonic"""
        if len(pitches) == 0:
            return np.array([])
            
        # Use median as robust tonic estimate
        if tonic_idx < len(pitches):
            tonic_freq = pitches[tonic_idx]
        else:
            tonic_freq = np.median(pitches)
        
        # Convert to cents
        pitch_array = np.array(pitches)
        pitch_array = pitch_array[pitch_array > 0]  # Remove zeros
        
        if len(pitch_array) == 0:
            return np.array([])
            
        return 1200 * np.log2(pitch_array / tonic_freq)
    
    def _extract_ngrams(self, sequence, n=2):
        """Extract n-gram patterns from pitch sequence"""
        if len(sequence) < n:
            return Counter()
            
        # Quantize to 50-cent bins for pattern extraction
        quantized = np.round(np.array(sequence) / 50) * 50
        
        # Extract n-grams
        ngrams = []
        for i in range(len(quantized) - n + 1):
            ngram = tuple(np.diff(quantized[i:i+n]))  # Use intervals, not absolute pitches
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def _calculate_phrase_complexity(self, pitch_cents):
        """Calculate melodic complexity measure"""
        if len(pitch_cents) == 0:
            return 0.0
            
        unique_pitches = len(np.unique(np.round(pitch_cents / 50)))  # 50-cent resolution
        pitch_range = (np.max(pitch_cents) - np.min(pitch_cents)) / 1200  # In octaves
        
        return unique_pitches * pitch_range
    
    def _estimate_gamaka_intensity(self, pitch_cents):
        """Estimate ornamental intensity"""
        if len(pitch_cents) < 2:
            return 0.0
            
        # Measure pitch velocity (rate of change)
        pitch_velocity = np.abs(np.diff(pitch_cents))
        
        # Consider only significant movements (> 20 cents)
        significant_movements = pitch_velocity[pitch_velocity > 20]
        
        if len(significant_movements) == 0:
            return 0.0
            
        return np.mean(significant_movements) / 100  # Normalize
    
    def _flatten_features(self, features_dict):
        """Convert nested feature dict to flat array"""
        flat_features = []
        
        def flatten_recursive(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_recursive(value)
                elif isinstance(value, (list, np.ndarray)):
                    flat_features.extend(value.flatten())
                else:
                    if np.isfinite(value):  # Only add finite values
                        flat_features.append(float(value))
                    else:
                        flat_features.append(0.0)  # Replace inf/nan with 0
        
        flatten_recursive(features_dict)
        return np.array(flat_features)

# Usage example
def extract_enhanced_features(audio, pitches=None, tonic_idx=0):
    """Convenience function for easy feature extraction"""
    
    extractor = MultiModalFeatureExtractor()
    return extractor.extract_all_features(audio, pitches, tonic_idx)

if __name__ == "__main__":
    # Test the feature extractor
    dummy_audio = np.random.randn(16000)  # 1 second
    dummy_pitches = np.random.uniform(200, 800, 100)  # 100 pitch points
    
    features = extract_enhanced_features(dummy_audio, dummy_pitches)
    print(f"Extracted {len(features)} features")
    print(f"Feature sample: {features[:10]}")