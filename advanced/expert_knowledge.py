# advanced/expert_knowledge.py
"""
Musicological expert knowledge for enhanced raga recognition
"""

import numpy as np
from collections import defaultdict

class MusicologicalExpert:
    """Apply expert knowledge for better raga recognition"""
    
    def __init__(self, tradition='Carnatic'):
        self.tradition = tradition
        self.alias_map = self._build_alias_map()
        self.raga_knowledge = self._load_expert_knowledge()
        self.confusion_patterns = self._load_confusion_patterns()
    
    def _build_alias_map(self):
        """Map common alias spellings to canonical raga names"""
        # Canonical names use lowercase
        return {
            # Carnatic aliases
            'hamsadhvani': 'hamsadhvani',
            'hamsadhwani': 'hamsadhvani',
            'hamsdhwani': 'hamsadhvani',
            'mohanam': 'mohanam',
            'shankarabharanam': 'sankarabharanam',
            'sankarabharanam': 'sankarabharanam',
            'kedaragowla': 'kedaragaula',
            'kedaragowlam': 'kedaragaula',
            'todi': 'todi',
            'kalyani': 'kalyani',
            # Hindustani aliases
            'yaman': 'yaman',
            'bhairav': 'bhairav',
            'malkauns': 'malkauns',
            'bhupali': 'bhupali',
            'bhup': 'bhupali',
            'des': 'desh',
            'desh': 'desh',
            'rageshri': 'rageshree',
            'rageshree': 'rageshree',
            'ragesri': 'rageshree',
            'ragesree': 'rageshree',
            'bilawal': 'bilaval',
            'bilaval': 'bilaval',
        }
    
    def _normalize_name(self, name):
        if not name:
            return name
        key = str(name).strip().lower()
        return self.alias_map.get(key, key)
        
    def _load_expert_knowledge(self):
        """Load comprehensive musicological knowledge"""
        
        if self.tradition == 'Carnatic':
            return {
                'hamsadhvani': {
                    'scale_pattern': [0, 200, 400, 700, 1100],  # S R2 G3 P N2
                    'forbidden_notes': [500, 900],  # Ma, Dha completely absent
                    'characteristic_phrases': [
                        [0, 200, 400, 700, 1100],      # S R G P N (ascent)
                        [1100, 700, 400, 200, 0],      # N P G R S (descent)
                        [400, 700, 1100, 1200],        # G P N S (jump to upper Sa)
                        [0, 400, 700],                 # S G P (common phrase)
                    ],
                    'emphasis_notes': [0, 400, 700, 1100],  # Sa, Ga, Pa, Ni important
                    'weak_notes': [200],  # Re is usually quick/passing
                    'typical_movements': 'stepwise_with_jumps',
                    'ornament_style': 'simple_kampita',
                    'time_signature': 'evening',
                    'emotional_content': 'devotional_joyful',
                    'tempo_range': (80, 140),
                    'phrase_endings': [0, 700, 1100],  # Usually end on Sa, Pa, or Ni
                    'vakra_phrases': [],  # No characteristic zigzag patterns
                },
                
                'mohanam': {
                    'scale_pattern': [0, 200, 400, 700, 900],  # S R2 G3 P D2
                    'forbidden_notes': [500, 1100],  # Ma, Ni absent
                    'characteristic_phrases': [
                        [0, 200, 400, 700, 900],       # S R G P D
                        [900, 700, 400, 200, 0],       # D P G R S
                        [0, 400, 700, 900],            # S G P D
                        [700, 900, 1200],              # P D S (upper)
                    ],
                    'emphasis_notes': [0, 400, 700, 900],
                    'weak_notes': [200],
                    'time_signature': 'evening',
                    'emotional_content': 'peaceful_devotional',
                    'tempo_range': (70, 130),
                    'phrase_endings': [0, 700, 900],
                },
                
                'sankarabharanam': {
                    'scale_pattern': [0, 200, 400, 500, 700, 900, 1100],  # Full scale
                    'forbidden_notes': [],  # All notes allowed
                    'characteristic_phrases': [
                        [0, 200, 400, 500, 700, 900, 1100, 1200],  # Full ascent
                        [0, 400, 700, 1200],                       # S G P S (important)
                        [500, 700, 500, 400],                      # M P M G (vakra)
                        [700, 500, 400, 200, 0],                   # P M G R S
                    ],
                    'emphasis_notes': [0, 500, 700],  # Sa, Ma, Pa are very important
                    'vakra_phrases': [
                        [400, 200, 400, 500],  # G R G M (characteristic zigzag)
                        [900, 700, 900, 1100], # D P D N
                    ],
                    'time_signature': 'any',
                    'emotional_content': 'devotional_majestic',
                    'tempo_range': (60, 160),
                },
                
                'todi': {
                    'scale_pattern': [0, 100, 300, 600, 700, 800, 1100],  # S r1 g2 m2 P d1 N2
                    'forbidden_notes': [200, 400, 500, 900],  # R2, G3, M1, D2 forbidden
                    'characteristic_phrases': [
                        [0, 100, 300, 600],    # S r g m (ascent with komal notes)
                        [1100, 800, 700],      # N d P (descent)
                        [600, 300, 100, 0],    # m g r S
                    ],
                    'emphasis_notes': [0, 100, 300, 600, 800],
                    'time_signature': 'morning',
                    'emotional_content': 'serious_contemplative',
                    'tempo_range': (50, 100),
                    'ornament_style': 'heavy_gamaka',
                },
                
                'kalyani': {
                    'scale_pattern': [0, 200, 400, 600, 700, 900, 1100],  # S R2 G3 M2 P D2 N2
                    'forbidden_notes': [500],  # M1 (shuddha Ma) forbidden
                    'characteristic_phrases': [
                        [600, 900, 1100, 1200],  # M D N S (tivra Ma emphasis)
                        [0, 200, 400, 600, 700], # S R G M P
                        [1100, 900, 600, 400],   # N D M G
                    ],
                    'emphasis_notes': [0, 600, 900, 1100],  # Sa, tivra Ma, Dha, Ni
                    'time_signature': 'evening',
                    'emotional_content': 'auspicious_bright',
                    'tempo_range': (80, 140),
                },
            }
        
        else:  # Hindustani
            return {
                'yaman': {
                    'scale_pattern': [0, 200, 400, 600, 700, 900, 1100],  # S R G M(tivra) P D N
                    'forbidden_notes': [500],  # Shuddha Ma completely avoided
                    'characteristic_phrases': [
                        [900, 1100, 1200, 1100, 900],  # D N S N D (signature phrase)
                        [0, 200, 400, 600, 700],       # S R G M P
                        [600, 900, 1100],              # M D N (tivra Ma prominence)
                    ],
                    'emphasis_notes': [600, 900, 1100],  # Tivra Ma, Dha, Ni very important
                    'time_signature': 'evening',
                    'emotional_content': 'romantic_serene',
                    'tempo_range': (60, 120),
                    'phrase_endings': [0, 900, 1100],
                },
                
                'bhairav': {
                    'scale_pattern': [0, 100, 400, 500, 700, 800, 1100],  # S r G M P d N
                    'forbidden_notes': [200, 300, 600, 900, 1000],  # R, g, m2, D, n forbidden
                    'characteristic_phrases': [
                        [0, 100, 400, 500, 700],   # S r G M P
                        [1100, 800, 700],          # N d P
                        [100, 400, 100, 0],        # r G r S (emphasis on komal Re)
                    ],
                    'emphasis_notes': [0, 100, 800, 1100],  # Sa, komal Re, komal Dha, Ni
                    'time_signature': 'early_morning',
                    'emotional_content': 'devotional_serious',
                    'tempo_range': (40, 90),
                    'ornament_style': 'heavy_meend',
                },
                
                'malkauns': {
                    'scale_pattern': [0, 300, 500, 800, 1000],  # S g M d n
                    'forbidden_notes': [200, 400, 600, 700, 900, 1100],  # R, G, m2, P, D, N
                    'characteristic_phrases': [
                        [0, 300, 500, 800],     # S g M d
                        [1000, 800, 500, 300], # n d M g
                        [500, 300, 0],          # M g S
                    ],
                    'emphasis_notes': [0, 300, 500, 800],
                    'time_signature': 'late_night',
                    'emotional_content': 'deep_meditative',
                    'tempo_range': (40, 80),
                    'ornament_style': 'subtle_meend',
                },
                
                'bhupali': {
                    'scale_pattern': [0, 200, 400, 700, 900],  # S R G P D
                    'forbidden_notes': [500, 1100],  # Ma, Ni absent
                    'characteristic_phrases': [
                        [0, 200, 400, 700, 900],   # S R G P D
                        [900, 700, 400, 200, 0],   # D P G R S
                        [0, 400, 700],             # S G P
                    ],
                    'emphasis_notes': [0, 400, 700],
                    'time_signature': 'evening',
                    'emotional_content': 'peaceful_sublime',
                    'tempo_range': (60, 110),
                },
                
                # Added: Desh and Rageshree (common confusion pair)
                'desh': {
                    # Desh: Pa present; typical pakad includes P N S', R' N D P, M G R G S
                    'scale_pattern': [0, 200, 400, 500, 700, 900, 1100],
                    'forbidden_notes': [],
                    'characteristic_phrases': [
                        [700, 1100, 1200],          # P N S'
                        [1200, 1100, 900, 700],     # S' N D P
                        [500, 400, 200, 400, 0],    # M G R G S
                    ],
                    'emphasis_notes': [0, 500, 700, 1100],
                    'time_signature': 'evening',
                    'emotional_content': 'romantic_festive',
                    'tempo_range': (60, 120),
                },
                'rageshree': {
                    # Rageshree: Pa omitted; pakad: G M D M G R S
                    'scale_pattern': [0, 200, 400, 500, 900, 1200],  # No Pa, often Ni used weakly or omitted
                    'forbidden_notes': [700],  # Pa is absent
                    'characteristic_phrases': [
                        [400, 500, 900, 500, 400, 200, 0],  # G M D M G R S
                        [0, 200, 400, 500, 900],             # S R G M D
                    ],
                    'emphasis_notes': [0, 400, 500],
                    'time_signature': 'late_evening',
                    'emotional_content': 'romantic_serene',
                    'tempo_range': (60, 110),
                },
            }
    
    def _load_confusion_patterns(self):
        """Load common confusion patterns and their resolution strategies"""
        
        return {
            ('hamsadhvani', 'mohanam'): {
                'distinguishing_feature': 'ni_vs_dha_presence',
                'resolution_method': self._resolve_ni_vs_dha,
                'confidence_threshold': 0.3,
                'typical_confusion_reason': 'Both pentatonic, differ only in 5th note'
            },
            
            # Added: Hamsadhvani vs Kalyani (Ma and Dha presence differentiate)
            ('hamsadhvani', 'kalyani'): {
                'distinguishing_feature': 'ma_dha_presence',
                'resolution_method': self._resolve_ma_dha_presence,
                'confidence_threshold': 0.3,
                'typical_confusion_reason': 'Kalyani has Ma and Dha; Hamsadhvani omits both'
            },
            
            ('yaman', 'bilaval'): {
                'distinguishing_feature': 'tivra_ma_presence',
                'resolution_method': self._resolve_tivra_ma,
                'confidence_threshold': 0.4,
                'typical_confusion_reason': 'Differ only in Ma (4th note) - sharp vs natural'
            },
            
            ('malkauns', 'hindolam'): {
                'distinguishing_feature': 'note_pattern_emphasis',
                'resolution_method': self._resolve_pentatonic_minor,
                'confidence_threshold': 0.25,
                'typical_confusion_reason': 'Both pentatonic with minor tonality'
            },
            
            ('bhairav', 'todi'): {
                'distinguishing_feature': 'ma_type_and_emphasis',
                'resolution_method': self._resolve_morning_ragas,
                'confidence_threshold': 0.3,
                'typical_confusion_reason': 'Both morning ragas with komal notes'
            },
            
            # Added: Desh vs Rageshree (Pa presence differentiates)
            ('desh', 'rageshree'): {
                'distinguishing_feature': 'pa_presence',
                'resolution_method': self._resolve_pa_presence,
                'confidence_threshold': 0.3,
                'typical_confusion_reason': 'Desh uses Pa; Rageshree omits Pa'
            },
            
            # Added: Karaharapriya vs Kedaragaula (G2 vs G3 emphasis)
            ('karaharapriya', 'kedaragaula'): {
                'distinguishing_feature': 'ga_type',
                'resolution_method': self._resolve_ga_type_carnatic,
                'confidence_threshold': 0.3,
                'typical_confusion_reason': 'Different Ga (G2 ~300 vs G3 ~400) and phrase shapes'
            },
        }
    
    def apply_expert_analysis(self, predictions, pitch_cents, features):
        """Apply comprehensive expert analysis"""
        
        if not predictions or len(pitch_cents) == 0:
            return predictions
            
        # Normalize raga names in incoming predictions
        normalized_predictions = []
        for raga_name, confidence in predictions:
            normalized_predictions.append((self._normalize_name(raga_name), confidence))
        
        enhanced_predictions = []
        
        for raga_name, confidence in normalized_predictions:
            adjusted_confidence = confidence
            
            if raga_name in self.raga_knowledge:
                knowledge = self.raga_knowledge[raga_name]
                
                # Apply various expert rules
                forbidden_penalty = self._check_forbidden_notes(pitch_cents, knowledge.get('forbidden_notes', []))
                phrase_bonus = self._check_characteristic_phrases(pitch_cents, knowledge.get('characteristic_phrases', []))
                emphasis_bonus = self._check_note_emphasis(pitch_cents, knowledge.get('emphasis_notes', []))
                tempo_consistency = self._check_tempo_consistency(features, knowledge.get('tempo_range', (60, 200)))
                vakra_bonus = self._check_vakra_patterns(pitch_cents, knowledge.get('vakra_phrases', []))
                
                # Combine all factors
                total_penalty = forbidden_penalty
                total_bonus = phrase_bonus + emphasis_bonus + tempo_consistency + vakra_bonus
                
                adjusted_confidence = confidence * (1 - total_penalty) * (1 + total_bonus)
                adjusted_confidence = min(0.95, max(0.05, adjusted_confidence))  # Clamp
                
            enhanced_predictions.append((raga_name, adjusted_confidence))
        
        # Sort by adjusted confidence
        enhanced_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return enhanced_predictions
    
    def resolve_confusion_pairs(self, predictions, pitch_cents, features):
        """Resolve common confusions between similar ragas"""
        
        if len(predictions) < 2:
            return predictions
            
        # Normalize top candidates
        top1_raga, top1_conf = self._normalize_name(predictions[0][0]), predictions[0][1]
        top2_raga, top2_conf = self._normalize_name(predictions[1][0]), predictions[1][1]
        
        # Check for known confusion pairs
        pair_key = (top1_raga, top2_raga)
        reverse_pair = (top2_raga, top1_raga)
        
        confusion_info = None
        if pair_key in self.confusion_patterns:
            confusion_info = self.confusion_patterns[pair_key]
            primary, secondary = top1_raga, top2_raga
        elif reverse_pair in self.confusion_patterns:
            confusion_info = self.confusion_patterns[reverse_pair]
            primary, secondary = top2_raga, top1_raga
        
        if confusion_info:
            # Apply specific resolution
            resolution_result = confusion_info['resolution_method'](pitch_cents, features, primary, secondary)
            
            if resolution_result:
                suggested_raga, confidence_boost = resolution_result
                
                # Adjust predictions based on resolution
                adjusted_predictions = []
                for raga, conf in predictions:
                    raga_norm = self._normalize_name(raga)
                    if raga_norm == suggested_raga:
                        new_conf = min(0.92, conf + confidence_boost)
                    else:
                        new_conf = conf * 0.85  # Slight reduction for others
                    adjusted_predictions.append((raga_norm, new_conf))
                
                return sorted(adjusted_predictions, key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def _check_forbidden_notes(self, pitch_cents, forbidden_notes):
        """Check penalty for forbidden note presence"""
        if not forbidden_notes or len(pitch_cents) == 0:
            return 0.0
            
        total_penalty = 0.0
        for forbidden_cent in forbidden_notes:
            # Count occurrences within 30 cents tolerance
            violations = np.sum(np.abs(pitch_cents - forbidden_cent) < 30)
            penalty = (violations / len(pitch_cents)) * 0.6  # Up to 60% penalty
            total_penalty += penalty
        
        return min(0.8, total_penalty)  # Cap at 80% penalty
    
    def _check_characteristic_phrases(self, pitch_cents, characteristic_phrases):
        """Check bonus for characteristic phrase presence"""
        if not characteristic_phrases or len(pitch_cents) == 0:
            return 0.0
            
        total_bonus = 0.0
        for phrase in characteristic_phrases:
            match_strength = self._find_phrase_match(pitch_cents, phrase)
            total_bonus += match_strength * 0.25  # Up to 25% bonus per phrase
        
        return min(0.6, total_bonus)  # Cap total phrase bonus
    
    def _check_note_emphasis(self, pitch_cents, emphasis_notes):
        """Check bonus for proper emphasis of important notes"""
        if not emphasis_notes or len(pitch_cents) == 0:
            return 0.0
            
        total_emphasis = 0.0
        for note_cent in emphasis_notes:
            # Calculate time spent near this note (within 25 cents)
            note_time = np.sum(np.abs(pitch_cents - note_cent) < 25) / len(pitch_cents)
            total_emphasis += note_time
        
        # Normalize and convert to bonus
        expected_emphasis = 0.6  # Expect 60% time on emphasis notes
        emphasis_ratio = total_emphasis / expected_emphasis
        
        return min(0.3, emphasis_ratio * 0.2)  # Up to 20% bonus, capped at 30%
    
    def _check_tempo_consistency(self, features, tempo_range):
        """Check tempo consistency with raga expectations"""
        estimated_tempo = features.get('tempo', 100)
        min_tempo, max_tempo = tempo_range
        
        if min_tempo <= estimated_tempo <= max_tempo:
            return 0.1  # 10% bonus for correct tempo
        else:
            # Small penalty for being outside range
            distance = min(abs(estimated_tempo - min_tempo), abs(estimated_tempo - max_tempo))
            penalty = min(0.15, distance / 200)  # Up to 15% penalty
            return -penalty
    
    def _check_vakra_patterns(self, pitch_cents, vakra_phrases):
        """Check for characteristic zigzag (vakra) melodic patterns"""
        if not vakra_phrases or len(pitch_cents) == 0:
            return 0.0
            
        total_bonus = 0.0
        for vakra_phrase in vakra_phrases:
            match_strength = self._find_phrase_match(pitch_cents, vakra_phrase)
            total_bonus += match_strength * 0.15  # Up to 15% bonus per vakra pattern
        
        return min(0.3, total_bonus)  # Cap vakra bonus
    
    def _find_phrase_match(self, pitch_cents, target_phrase):
        """Find how well a target phrase matches the pitch sequence"""
        if len(pitch_cents) < len(target_phrase):
            return 0.0
            
        best_match = 0.0
        tolerance = 40  # 40 cents tolerance
        
        # Sliding window search
        for i in range(len(pitch_cents) - len(target_phrase) + 1):
            window = pitch_cents[i:i + len(target_phrase)]
            
            # Calculate phrase similarity
            similarity_score = 0.0
            for j, target_note in enumerate(target_phrase):
                if j < len(window):
                    diff = abs(window[j] - target_note)
                    if diff < tolerance:
                        similarity_score += 1 - (diff / tolerance)
            
            match_strength = similarity_score / len(target_phrase)
            best_match = max(best_match, match_strength)
        
        return best_match
    
    # Specific resolution methods for confusion pairs
    def _resolve_ma_dha_presence(self, pitch_cents, features, primary, secondary):
        """Resolve Hamsadhvani vs Kalyani using Ma (600) and Dha (900) presence"""
        ma_strength = np.sum(np.abs(pitch_cents - 600) < 30) / len(pitch_cents)
        dha_strength = np.sum(np.abs(pitch_cents - 900) < 30) / len(pitch_cents)
        # If both Ma and Dha are weak, favor Hamsadhvani; if present, favor Kalyani
        if (ma_strength + dha_strength) < 0.08:
            return ('hamsadhvani', 0.2)
        elif (ma_strength > 0.08) or (dha_strength > 0.08):
            return ('kalyani', 0.2)
        return None

    def _resolve_pa_presence(self, pitch_cents, features, primary, secondary):
        """Resolve Desh vs Rageshree using Pa (700) presence/absence"""
        pa_strength = np.sum(np.abs(pitch_cents - 700) < 30) / len(pitch_cents)
        if pa_strength > 0.12:
            return ('desh', 0.25)
        elif pa_strength < 0.05:
            return ('rageshree', 0.25)
        return None

    def _resolve_ga_type_carnatic(self, pitch_cents, features, primary, secondary):
        """Resolve Karaharapriya (G2 ~300 cents) vs Kedaragaula (G3 ~400 cents)"""
        g2_strength = np.sum(np.abs(pitch_cents - 300) < 30) / len(pitch_cents)
        g3_strength = np.sum(np.abs(pitch_cents - 400) < 30) / len(pitch_cents)
        # Bias threshold slightly as Ga often oscillates; require clear separation
        if g2_strength > g3_strength * 1.3:
            return ('karaharapriya', 0.25)
        elif g3_strength > g2_strength * 1.3:
            return ('kedaragaula', 0.25)
        return None
    def _resolve_ni_vs_dha(self, pitch_cents, features, primary, secondary):
        """Resolve Hamsadhvani (Ni) vs Mohanam (Dha) confusion"""
        ni_strength = np.sum(np.abs(pitch_cents - 1100) < 35) / len(pitch_cents)  # Ni around 1100 cents
        dha_strength = np.sum(np.abs(pitch_cents - 900) < 35) / len(pitch_cents)   # Dha around 900 cents
        
        print(f"🔍 Ni vs Dha analysis: Ni strength = {ni_strength:.3f}, Dha strength = {dha_strength:.3f}")
        
        if ni_strength > dha_strength * 1.8:
            return ('hamsadhvani', 0.35)  # Strong evidence for Hamsadhvani
        elif dha_strength > ni_strength * 1.8:
            return ('mohanam', 0.3)  # Strong evidence for Mohanam
        elif ni_strength > dha_strength * 1.2:
            return ('hamsadhvani', 0.15)  # Moderate evidence for Hamsadhvani
        
        return None  # Unclear evidence
    
    def _resolve_tivra_ma(self, pitch_cents, features, primary, secondary):
        """Resolve Yaman (tivra Ma) vs Bilaval (shuddha Ma) confusion"""
        tivra_ma_strength = np.sum(np.abs(pitch_cents - 600) < 30) / len(pitch_cents)    # F# ~600 cents
        shuddha_ma_strength = np.sum(np.abs(pitch_cents - 500) < 30) / len(pitch_cents)  # F ~500 cents
        
        print(f"🔍 Ma analysis: Tivra Ma = {tivra_ma_strength:.3f}, Shuddha Ma = {shuddha_ma_strength:.3f}")
        
        if tivra_ma_strength > shuddha_ma_strength * 2:
            return ('yaman', 0.4)  # Strong evidence for Yaman
        elif shuddha_ma_strength > tivra_ma_strength * 2:
            return ('bilaval', 0.3)  # Evidence for Bilaval
        
        return None
    
    def _resolve_pentatonic_minor(self, pitch_cents, features, primary, secondary):
        """Resolve Malkauns vs Hindolam confusion"""
        # Both are pentatonic minor-type ragas, distinguish by emphasis patterns
        
        # Malkauns emphasizes komal Ga (300 cents) and komal Ni (1000 cents)
        malkauns_notes = [0, 300, 500, 800, 1000]  # S g M d n
        hindolam_notes = [0, 300, 500, 800, 1000]   # Same scale, different emphasis
        
        # Check phrase patterns (Malkauns has more Ma-centric phrases)
        ma_emphasis = np.sum(np.abs(pitch_cents - 500) < 30) / len(pitch_cents)
        
        if ma_emphasis > 0.2:  # Strong Ma emphasis suggests Malkauns
            return ('malkauns', 0.2)
        
        return None
    
    def _resolve_morning_ragas(self, pitch_cents, features, primary, secondary):
        """Resolve Bhairav vs Todi confusion"""
        # Check for tivra Ma (Todi) vs shuddha Ma (Bhairav)
        tivra_ma = np.sum(np.abs(pitch_cents - 600) < 30) / len(pitch_cents)
        shuddha_ma = np.sum(np.abs(pitch_cents - 500) < 30) / len(pitch_cents)
        
        if tivra_ma > shuddha_ma * 1.5:
            return ('todi', 0.25)  # Todi has tivra Ma
        elif shuddha_ma > tivra_ma * 1.5:
            return ('bhairav', 0.2)  # Bhairav has shuddha Ma
        
        return None
    
    def get_raga_explanation(self, predicted_raga, pitch_cents, confidence):
        """Provide detailed explanation of why this raga was predicted"""
        
        if predicted_raga not in self.raga_knowledge:
            return {
                'prediction': predicted_raga,
                'confidence': confidence,
                'explanation': 'Limited knowledge available for this raga'
            }
        
        knowledge = self.raga_knowledge[predicted_raga]
        explanation = {
            'prediction': predicted_raga,
            'confidence': confidence,
            'scale_pattern': knowledge.get('scale_pattern', []),
            'tradition': self.tradition,
        }
        
        # Analyze note usage
        note_analysis = {}
        for note_name, cent_value in [('Sa', 0), ('Re', 200), ('Ga', 400), ('Ma', 500), 
                                     ('Pa', 700), ('Dha', 900), ('Ni', 1100)]:
            usage = np.sum(np.abs(pitch_cents - cent_value) < 30) / len(pitch_cents) if len(pitch_cents) > 0 else 0
            note_analysis[note_name] = {
                'usage_percentage': usage * 100,
                'expected': cent_value in knowledge.get('emphasis_notes', []),
                'forbidden': cent_value in knowledge.get('forbidden_notes', [])
            }
        
        explanation['note_analysis'] = note_analysis
        
        # Check for characteristic phrases
        phrase_matches = []
        for i, phrase in enumerate(knowledge.get('characteristic_phrases', [])):
            match_strength = self._find_phrase_match(pitch_cents, phrase)
            if match_strength > 0.3:  # Only report significant matches
                phrase_matches.append({
                    'phrase_index': i,
                    'match_strength': match_strength,
                    'phrase_cents': phrase
                })
        
        explanation['phrase_matches'] = phrase_matches
        explanation['musical_context'] = {
            'time_signature': knowledge.get('time_signature', 'any'),
            'emotional_content': knowledge.get('emotional_content', 'neutral'),
            'tempo_range': knowledge.get('tempo_range', (60, 200))
        }
        
        return explanation

# Convenience functions for integration
def apply_expert_knowledge(predictions, pitch_cents, features, tradition='Carnatic'):
    """Apply expert knowledge to improve predictions"""
    expert = MusicologicalExpert(tradition)
    
    # Apply expert analysis
    enhanced_predictions = expert.apply_expert_analysis(predictions, pitch_cents, features)
    
    # Resolve confusion pairs
    final_predictions = expert.resolve_confusion_pairs(enhanced_predictions, pitch_cents, features)
    
    return final_predictions

def get_detailed_explanation(predicted_raga, pitch_cents, confidence, tradition='Carnatic'):
    """Get detailed explanation for a prediction"""
    expert = MusicologicalExpert(tradition)
    return expert.get_raga_explanation(predicted_raga, pitch_cents, confidence)

if __name__ == "__main__":
    # Test the expert system
    expert = MusicologicalExpert('Carnatic')
    
    # Simulate pitch sequence for Hamsadhvani
    # Hamsadhvani: S R G P N (0, 200, 400, 700, 1100 cents)
    test_pitches = np.array([0, 200, 400, 700, 1100, 1200, 1100, 700, 400, 200, 0] * 10)
    test_pitches += np.random.normal(0, 15, len(test_pitches))  # Add some variation
    
    test_predictions = [('hamsadhvani', 0.4), ('mohanam', 0.35), ('sankarabharanam', 0.25)]
    test_features = {'tempo': 100}
    
    # Test expert analysis
    enhanced = expert.apply_expert_analysis(test_predictions, test_pitches, test_features)
    print("Enhanced predictions:", enhanced)
    
    # Test confusion resolution
    final = expert.resolve_confusion_pairs(enhanced, test_pitches, test_features)
    print("Final predictions:", final)
    
    # Test explanation
    explanation = expert.get_raga_explanation(final[0][0], test_pitches, final[0][1])
    print("Explanation keys:", list(explanation.keys()))