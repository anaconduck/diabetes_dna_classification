from src.encoding.kmer import extract_kmers, train_or_load_word2vec, create_seqs_embedding, extract_kmers_with_padding
from src.encoding.one_hot import create_one_hot_embeddings
from src.adaptive_kmer.adaptive_generator import extract_multi_scale_kmers
from src.adaptive_kmer.adaptive_representation import train_adaptive_word2vec, create_adaptive_embeddings, create_adaptive_sequence_embeddings

class EncoderFactory:
    def __init__(self, config):
        self.config = config
        self.encoding_type = config.get('encoding', {}).get('type', 'kmer_word2vec')
        
    def extract_features(self, dataset):
        ksize = self.config['encoding'].get('kmer_size', 3)
        model_type = self.config.get('models', {}).get('type', 'lstm')
        is_sequential = model_type in ['lstm', 'attention_lstm', 'cnn']
        
        if self.encoding_type == 'kmer_word2vec':
            if is_sequential:
                return extract_kmers_with_padding(dataset, ksize)
            else:
                features, labels = extract_kmers(dataset, ksize)
                return (features, None), labels
                
        elif self.encoding_type == 'one_hot':
            features, labels = extract_kmers(dataset, ksize)
            return (features, None), labels
        elif self.encoding_type == 'adaptive_word2vec':
            scales = self.config['encoding'].get('scales', [3, 4, 5])
            features, max_len = extract_multi_scale_kmers(dataset, scales)
            
            labels_list = []
            for index, row in dataset.iterrows():
                labels_list.append(row['class'])
                
            return (features, max_len), labels_list
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
    def train_embedder(self, features_train, save_dir):
        if self.encoding_type == 'kmer_word2vec':
            kmers_train = features_train[0]
            return train_or_load_word2vec(kmers_train, self.config, save_dir)
        elif self.encoding_type == 'one_hot':
            return None 
        elif self.encoding_type == 'adaptive_word2vec':
            multi_kmers_train = features_train[0]
            return train_adaptive_word2vec(multi_kmers_train, self.config, save_dir)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
            
    def create_embeddings(self, embedder_model, features):
        model_type = self.config.get('models', {}).get('type', 'lstm')
        is_sequential = model_type in ['lstm', 'attention_lstm', 'cnn']
        
        if self.encoding_type == 'kmer_word2vec':
            vector_size = self.config['encoding']['word2vec']['vector_size']
            multi_kmers, max_len = features
            if is_sequential:
                from src.encoding.kmer import create_seqs_sequence_embedding
                return create_seqs_sequence_embedding(embedder_model, multi_kmers, vector_size, max_len)
            else:
                return create_seqs_embedding(embedder_model, multi_kmers, vector_size)
        elif self.encoding_type == 'one_hot':
            ksize = self.config['encoding'].get('kmer_size', 3)
            multi_kmers, _ = features
            return create_one_hot_embeddings(multi_kmers, ksize)
        elif self.encoding_type == 'adaptive_word2vec':
            scales = self.config['encoding'].get('scales', [3, 4, 5])
            vector_size = self.config['encoding']['word2vec']['vector_size']
            multi_kmers, max_len = features
            if is_sequential:
                return create_adaptive_sequence_embeddings(embedder_model, multi_kmers, scales, vector_size, max_len)
            else:
                return create_adaptive_embeddings(embedder_model, multi_kmers, scales, vector_size)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
