import numpy as np
import scipy.io.wavfile as wav
import os
from voice_features import extract_features
import json

def create_synthetic_sound(filename="test.wav", duration=1.0, sr=44100, f0=150.0):
    t = np.linspace(0, duration, int(sr * duration))
    # Create a signal with some jitter and noise
    # Base signal
    signal = np.sin(2 * np.pi * f0 * t)
    # Add some harmonics
    signal += 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    # Add random noise
    signal += 0.05 * np.random.normal(size=len(t))
    # Frequency modulation (jitter)
    signal += 0.1 * np.sin(2 * np.pi * (f0 + 5 * np.sin(2 * np.pi * 10 * t)) * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    wav.write(filename, sr, (signal * 32767).astype(np.int16))
    return filename

def test_features():
    filename = create_synthetic_sound()
    try:
        print(f"Analyzing {filename}...")
        features = extract_features(filename)
        print("Features extracted successfully:")
        print(json.dumps(features, indent=2))
        
        # Check specific new features
        new_features = ["RPDE", "DFA", "spread1", "spread2", "D2"]
        missing = []
        zeros = []
        for nf in new_features:
            if nf not in features:
                missing.append(nf)
            elif features[nf] == 0.0:
                 zeros.append(nf)
        
        if missing:
            print(f"FAIL: Missing features: {missing}")
        elif zeros:
            print(f"WARNING: Features are zero (might be expected for perfect sine but check): {zeros}")
        else:
            print("PASS: All new features have non-zero values.")
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_features()
