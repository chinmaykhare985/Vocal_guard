"""Utility to extract voice/acoustic features from an audio file.

This module uses `parselmouth` (Praat bindings) to compute a set of
features commonly used in Parkinson's detection datasets (MDVP,
jitter/shimmer, HNR, etc.). Some advanced non-linear features are
left as placeholders and should be replaced with specialized
implementations if required.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import sys
import json
import nolds
from scipy.spatial.distance import pdist, squareform


def extract_features(file_path: str):
    """Extract acoustic features from `file_path` and return a dict.

    Args:
        file_path: path to a WAV audio file.

    Returns:
        dict mapping feature names to numeric values, or an error dict.
    """
    try:
        # Load audio file into a parselmouth Sound object
        sound = parselmouth.Sound(file_path)
    except Exception as e:
        return {"error": f"Could not load audio: {str(e)}"}

    # --- Pitch Analysis ---
    # Create pitch object
    pitch = sound.to_pitch()
    
    # MDVP:Fo(Hz) - Average fundamental frequency
    fo_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    
    # MDVP:Fhi(Hz) - Maximum pitch
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    
    # MDVP:Flo(Hz) - Minimum pitch
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # --- Jitter & Shimmer (PointProcess) ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    
    # MDVP:Jitter(%)
    jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # MDVP:Jitter(Abs)
    jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    
    # MDVP:RAP
    rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    
    # MDVP:PPQ
    ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Jitter:DDP (3 * RAP)
    ddp = 3 * rap

    # MDVP:Shimmer
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # MDVP:Shimmer(dB)
    shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Shimmer:APQ3
    apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Shimmer:APQ5
    apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # MDVP:APQ (APQ11 in Praat often maps to generic APQ in datasets)
    apq = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Shimmer:DDA (3 * APQ3)
    dda = 3 * apq3

    # --- Harmonicity ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    
    # HNR
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    # NHR (Approximate using 1/HNR or similar, but Praat doesn't have direct NHR. 
    # Usually calculated as 1 - HNR_ratio or variable. 
    hnr_value = 10 ** (hnr / 10)
    nhr = 1 / hnr_value if hnr_value != 0 else 0

    # --- Non-linear / Advanced Features ---
    # Convert sound to numpy array for signal processing
    signal = sound.values[0] # Handle stereo? usually mono for this analysis
    
    # helper for RPDE
    def calculate_rpde(signal, dim=4, tau=1, epsilon=None):
        try:
            # Simple embedding
            N = len(signal)
            if N < 500: return 0.0
            
            # Subsample if too large to avoid OOM
            if N > 2000:
                signal = signal[::N//2000] # Target ~2000 points
                N = len(signal)
            
            if epsilon is None:
                epsilon = 0.1 * np.std(signal)
            
            # 1. Embed
            n_vectors = N - (dim - 1) * tau
            vectors = np.array([signal[i : i + dim * tau : tau] for i in range(n_vectors)])
            
            # 2. Pairwise distances
            dists = pdist(vectors, metric='chebyshev')
            # 3. Recurrence matrix
            recurrence = (dists < epsilon).astype(int)
            dist_matrix = squareform(recurrence)
            
            # 4. Calculate recurrence times
            intervals = []
            for i in range(len(dist_matrix)):
                indices = np.where(dist_matrix[i])[0]
                diffs = np.diff(indices)
                intervals.extend(diffs)
            
            if not intervals:
                return 0.0
                
            # 5. Entropy
            intervals = np.array(intervals)
            counts = np.bincount(intervals)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log(probs))
            
            h_norm = h / np.log(len(counts)) if len(counts) > 1 else 0
            return h_norm

        except Exception as e:
            print(f"Algorithm Error: {e}", file=sys.stderr)
            return 0.0

    # RPDE (Recurrence Period Density Entropy)
    rpde = calculate_rpde(signal)
    
    # DFA (Detrended Fluctuation Analysis)
    try:
        # nolds.dfa requires a 1D array
        dfa_signal = signal
        # Subsample for speed
        if len(dfa_signal) > 2000:
             dfa_signal = dfa_signal[::len(dfa_signal)//2000]
        dfa = nolds.dfa(dfa_signal)
    except Exception as e:
        print(f"DFA Error: {e}", file=sys.stderr)
        dfa = 0.0

    # PPE (Pitch Period Entropy)
    # Measures entropy of pitch variations.
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0] # Remove unvoiced
    
    if len(pitch_values) > 0:
        log_pitch = np.log2(pitch_values)
        ppe = np.std(log_pitch) 
        
        # Spread1: Standard deviation of semitone pitch variations
        spread1 = 12 * ppe 
        
        # Spread2: Standard deviation of differences in semitones (delta)
        diff_log_pitch = np.diff(log_pitch)
        spread2 = 12 * np.std(diff_log_pitch) if len(diff_log_pitch) > 0 else 0.0
        
    else:
        ppe = 0.0
        spread1 = 0.0
        spread2 = 0.0
        
    # D2 (Correlation Dimension)
    try:
        d2_signal = signal
        if len(d2_signal) > 1500:
            d2_signal = d2_signal[::len(d2_signal)//1500] # Take max 1500 points
        d2 = nolds.corr_dim(d2_signal, emb_dim=4)
    except Exception as e:
        print(f"D2 Error: {e}", file=sys.stderr)
        d2 = 0.0

    features = {
        "MDVP:Fo(Hz)": fo_mean,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_percent,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    }
            
    # Clean NaNs
    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0.0
            
    return features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python voice_features.py <audio_file>"}))
        sys.exit(1)

    file_path = sys.argv[1]
    result = extract_features(file_path)
    print(json.dumps(result, indent=4))
