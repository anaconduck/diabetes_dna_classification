import numpy as np
import os
from gensim.models import Word2Vec
from src.encoding.kmer import create_seqs_embedding

def train_adaptive_word2vec(multi_kmers_train, config, save_dir="weights"):
    """
    Trains a single Word2Vec model on a mixed corpus containing k-mers from all scales.
    
    Args:
        multi_kmers_train: dict {scale: [list of kmer sequences for all samples]}
        config: config dict
    """
    mixed_corpus = []
    
    # Combine all scales into one giant corpus
    for scale, sequences in multi_kmers_train.items():
        # Remove padding tokens for Word2Vec training
        for seq in sequences:
            clean_seq = [token for token in seq if token != '[PAD]']
            mixed_corpus.append(clean_seq)
            
    min_count = config['encoding']['word2vec'].get('min_count', 1)
    window = config['encoding']['word2vec'].get('window', 5)
    vector_size = config['encoding']['word2vec'].get('vector_size', 100)
    workers = config['encoding']['word2vec'].get('workers', 4)
    
    w2v_model = Word2Vec(mixed_corpus, 
                         min_count=min_count, 
                         window=window, 
                         vector_size=vector_size, 
                         workers=workers)
    
    os.makedirs(save_dir, exist_ok=True)
    w2v_model.save(os.path.join(save_dir, "adaptive_word2vec.model"))
    
    return w2v_model

def create_adaptive_embeddings(w2v_model, multi_kmers, scales, vector_size):
    """
    Creates embeddings for each scale and stacks them.
    
    Returns:
        numpy array of shape (samples, vector_size, num_scales)
    """
    num_samples = len(multi_kmers[scales[0]])
    num_scales = len(scales)
    
    # Preallocate output array
    stacked_embeddings = np.zeros((num_samples, vector_size, num_scales))
    
    # Get embeddings for each scale and place them into the stacked array
    for i, scale in enumerate(scales):
        # We must remove [PAD] tokens before passing to the baseline embedding function
        # since the baseline averages over sequence length.
        clean_sequences = []
        for seq in multi_kmers[scale]:
            clean_seq = [token for token in seq if token != '[PAD]']
            clean_sequences.append(clean_seq)
            
        scale_embedding = create_seqs_embedding(w2v_model, clean_sequences, vector_size)
        
        # scale_embedding is (samples, vector_size)
        stacked_embeddings[:, :, i] = scale_embedding
        
    return stacked_embeddings

def create_adaptive_sequence_embeddings(w2v_model, multi_kmers, scales, vector_size, max_len):
    """
    Creates sequence embeddings for each scale and stacks them.
    
    Returns:
        numpy array of shape (samples, max_len, vector_size, num_scales)
    """
    from src.encoding.kmer import create_seqs_sequence_embedding
    
    num_samples = len(multi_kmers[scales[0]])
    num_scales = len(scales)
    
    # Preallocate output array
    stacked_embeddings = np.zeros((num_samples, max_len, vector_size, num_scales))
    
    # Get sequence embeddings for each scale
    for i, scale in enumerate(scales):
        scale_embedding = create_seqs_sequence_embedding(
            w2v_model, 
            multi_kmers[scale], 
            vector_size, 
            max_len
        )
        # scale_embedding is (samples, max_len, vector_size)
        stacked_embeddings[:, :, :, i] = scale_embedding
        
    return stacked_embeddings
