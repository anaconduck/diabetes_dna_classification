import numpy as np
from src.encoding.kmer import build_kmers

def extract_multi_scale_kmers(dataset, scales=[3, 4, 5]):
    """
    Extracts k-mers at multiple scales for each sequence in the dataset.
    Reuses the baseline build_kmers function.
    
    Args:
        dataset: A pandas DataFrame or dict containing a 'sequence' column.
        scales: List of k-mer sizes to extract.
        
    Returns:
        multi_kmers: A dictionary where keys are scales (e.g., 3) and 
                     values are lists of k-mer sequences.
        max_len: The maximum length (usually length of sequence when k=min(scales)).
    """
    multi_kmers = {scale: [] for scale in scales}
    max_len = 0
    
    for seq in dataset['sequence']:
        # Generate k-mers for all scales
        seq_multi = {}
        for scale in scales:
            kmers = build_kmers(seq, scale)
            seq_multi[scale] = kmers
            
            if len(kmers) > max_len:
                max_len = len(kmers)
                
        # Store in the dictionary
        for scale in scales:
            multi_kmers[scale].append(seq_multi[scale])
            
    # Apply Post-Padding so all sequences across all scales have identical lengths
    padded_multi_kmers = {scale: [] for scale in scales}
    
    for scale in scales:
        for kmers in multi_kmers[scale]:
            pad_length = max_len - len(kmers)
            padded = kmers + ['[PAD]'] * pad_length
            padded_multi_kmers[scale].append(padded)
            
    return padded_multi_kmers, max_len
