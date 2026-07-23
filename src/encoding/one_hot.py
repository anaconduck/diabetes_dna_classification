import numpy as np
import itertools

def get_kmer_vocabulary(ksize):
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=ksize)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def create_one_hot_embeddings(features, ksize):
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
