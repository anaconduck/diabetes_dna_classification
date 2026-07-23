import numpy as np
from src.encoding.kmer import build_kmers

def extract_multi_scale_kmers(dataset, scales=[3, 4, 5]):
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
                
        for scale in scales:
            multi_kmers[scale].append(seq_multi[scale])
            
    padded_multi_kmers = {scale: [] for scale in scales}
    
    for scale in scales:
        for kmers in multi_kmers[scale]:
            pad_length = max_len - len(kmers)
            padded = kmers + ['[PAD]'] * pad_length
            padded_multi_kmers[scale].append(padded)
            
    return padded_multi_kmers, max_len
