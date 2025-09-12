# famous_ragas.py
# Configuration for focusing on 20-30 most famous ragas

# Top 15 Hindustani Ragas (most commonly performed)
FAMOUS_HINDUSTANI_RAGAS = [
    'bilaval',      # Basic scale, morning raga
    'bhairav',      # Early morning, devotional
    'yaman',        # Evening, very popular
    'kafi',         # Semi-classical, folk influence
    'bhupali',      # Pentatonic, evening
    'malkauns',     # Pentatonic, late night
    'darbari',      # Serious, grand raga
    'marwa',        # Evening, unique mood
    'pooriya',      # Sunset raga
    'bageshri',     # Night raga, romantic
    'bihag',        # Night, playful
    'jaijaiwanti',  # Night, devotional
    'kedar',        # Night, monsoon
    'gaud_sarang',  # Rainy season
    'shree'         # Evening, majestic
]

# Top 15 Carnatic Ragas (most commonly performed)  
FAMOUS_CARNATIC_RAGAS = [
    'sankarabharanam',  # Basic scale (Bilawal equivalent)
    'kalyani',          # Yaman equivalent, auspicious
    'kharaharapriya',   # Natural minor scale
    'todi',             # Morning raga, serious
    'bhairavi',         # All times, devotional
    'hamsadhvani',      # Pentatonic, auspicious
    'mohanam',          # Pentatonic, Bhupali equivalent
    'hindolam',         # Pentatonic, Malkauns equivalent
    'saveri',           # Morning, peaceful
    'kambhoji',         # Evening, joyful
    'abhogi',           # Pentatonic, evening
    'shanmukhapriya',   # Complex, challenging
    'charukeshi',       # Evening, beautiful
    'mayamalavagowla',  # Basic, Bhairav equivalent
    'begada'            # Popular, versatile
]

def get_famous_ragas_list(tradition):
    """Return list of famous ragas for given tradition"""
    if tradition.lower() in ['hindustani', 'h']:
        return FAMOUS_HINDUSTANI_RAGAS
    elif tradition.lower() in ['carnatic', 'c']:
        return FAMOUS_CARNATIC_RAGAS
    else:
        return FAMOUS_HINDUSTANI_RAGAS + FAMOUS_CARNATIC_RAGAS

def filter_raga_predictions(predictions, tradition, top_k=5):
    """Filter predictions to only include famous ragas"""
    famous_ragas = get_famous_ragas_list(tradition)
    
    # Filter predictions to only include famous ragas
    filtered_predictions = []
    for raga, confidence in predictions:
        if raga.lower() in [r.lower() for r in famous_ragas]:
            filtered_predictions.append((raga, confidence))
    
    # Sort by confidence and return top_k
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    return filtered_predictions[:top_k]

def is_famous_raga(raga_name, tradition):
    """Check if a raga is in the famous ragas list"""
    famous_ragas = get_famous_ragas_list(tradition)
    return raga_name.lower() in [r.lower() for r in famous_ragas]

# Raga characteristics for better recognition (optional enhancement)
RAGA_CHARACTERISTICS = {
    # Hindustani
    'yaman': {'scale': 'C D E F# G A B', 'time': 'evening', 'mood': 'romantic'},
    'bhairav': {'scale': 'C Db E F G Ab B', 'time': 'early_morning', 'mood': 'devotional'},
    'malkauns': {'scale': 'C Eb F Ab Bb', 'time': 'late_night', 'mood': 'serious'},
    'bhupali': {'scale': 'C D E G A', 'time': 'evening', 'mood': 'peaceful'},
    'bilaval': {'scale': 'C D E F G A B', 'time': 'morning', 'mood': 'pure'},
    'kafi': {'scale': 'C D Eb F G A Bb', 'time': 'any', 'mood': 'folk'},
    
    # Carnatic  
    'sankarabharanam': {'scale': 'C D E F G A B', 'time': 'any', 'mood': 'devotional'},
    'kalyani': {'scale': 'C D E F# G A B', 'time': 'evening', 'mood': 'auspicious'},
    'todi': {'scale': 'C Db Eb F# G Ab B', 'time': 'morning', 'mood': 'serious'},
    'hamsadhvani': {'scale': 'C D E G B', 'time': 'evening', 'mood': 'joyful'},
    'mohanam': {'scale': 'C D E G A', 'time': 'evening', 'mood': 'peaceful'},
    'hindolam': {'scale': 'C Eb F Ab Bb', 'time': 'late_night', 'mood': 'serious'},
    'kharaharapriya': {'scale': 'C D Eb F G A Bb', 'time': 'evening', 'mood': 'devotional'},
    'bhairavi': {'scale': 'C Db Eb F G Ab Bb', 'time': 'any', 'mood': 'devotional'},
}

def get_raga_info(raga_name):
    """Get additional information about a raga"""
    return RAGA_CHARACTERISTICS.get(raga_name.lower(), {
        'scale': 'Unknown', 
        'time': 'any', 
        'mood': 'neutral'
    })