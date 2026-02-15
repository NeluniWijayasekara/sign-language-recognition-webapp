import numpy as np

def pad_or_truncate(sequence, max_len):
    sequence = np.array(sequence)
    if sequence.shape[0] >= max_len:
        return sequence[:max_len]
    padding = np.zeros((max_len - sequence.shape[0], sequence.shape[1]))
    return np.vstack([sequence, padding])
