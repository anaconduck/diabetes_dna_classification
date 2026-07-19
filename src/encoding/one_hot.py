import numpy as np
import itertools

def get_kmer_vocabulary(ksize):
    """
    Generates all possible DNA k-mers of length `ksize`.
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=ksize)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def create_one_hot_embeddings(features, ksize):
    """
    Converts a list of k-mer sequences (features) into K-mer Frequency Spectrum (Bag-of-K-mers).
    Note: Function is named one_hot for legacy compatibility, but computes normalized frequencies.
    Returns a numpy array of shape (num_samples, 4^ksize).
    """
    vocab = get_kmer_vocabulary(ksize)
    vocab_size = len(vocab)
    
    embeddings = []
    for seq_kmers in features:
        # Create a frequency vector (Bag of K-mers)
        vec = np.zeros(vocab_size)
        for kmer in seq_kmers:
            if kmer in vocab:
                vec[vocab[kmer]] += 1
        
        # Normalize the frequency vector
        total_kmers = len(seq_kmers)
        if total_kmers > 0:
            vec = vec / total_kmers
            
        embeddings.append(vec)
        
    return np.array(embeddings)
