import numpy as np

TOTAL_FEATURES = 201  # 75 (pose) + 63 (left hand) + 63 (right hand)

def pad_or_truncate(sequence, max_len):
    """
    Pad or truncate sequence to fixed length
    
    Args:
        sequence: list or numpy array of shape (n_frames, n_features)
        max_len: desired sequence length
    
    Returns:
        numpy array of shape (max_len, n_features)
    """
    # Convert to numpy array
    sequence = np.array(sequence)
    
    # Handle empty sequence
    if len(sequence) == 0:
        print("Warning: Empty sequence, returning zeros")
        return np.zeros((max_len, TOTAL_FEATURES))
    
    # Ensure 2D array
    if len(sequence.shape) == 1:
        # If 1D, assume it's a single frame
        if sequence.shape[0] == TOTAL_FEATURES:
            sequence = sequence.reshape(1, -1)
        else:
            print(f"Warning: Invalid sequence shape {sequence.shape}, returning zeros")
            return np.zeros((max_len, TOTAL_FEATURES))
    
    # Check feature dimension
    if sequence.shape[1] != TOTAL_FEATURES:
        print(f"Warning: Expected {TOTAL_FEATURES} features, got {sequence.shape[1]}")
        # Fix feature dimension
        if sequence.shape[1] > TOTAL_FEATURES:
            sequence = sequence[:, :TOTAL_FEATURES]
        else:
            padding = np.zeros((sequence.shape[0], TOTAL_FEATURES - sequence.shape[1]))
            sequence = np.hstack([sequence, padding])
    
    # Handle sequence length
    if sequence.shape[0] >= max_len:
        # Take evenly spaced frames if we have too many
        if sequence.shape[0] > max_len:
            indices = np.linspace(0, sequence.shape[0]-1, max_len, dtype=int)
            return sequence[indices]
        return sequence[:max_len]
    
    # Pad if shorter
    padding = np.zeros((max_len - sequence.shape[0], sequence.shape[1]))
    return np.vstack([sequence, padding])