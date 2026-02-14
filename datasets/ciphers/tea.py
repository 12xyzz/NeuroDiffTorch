import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Tea_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x40000000,0x00000000), neg='real_encryption', batch_size=10000):
        self.cipher = Tea()
        self.single_key, self.key_array = self._generate_keys(n, key_mode, key)
        self.n = n
        self.nr = nr
        self.diff = diff
        self.neg = neg
        self.batch_size = batch_size
        self.X = None
        self.Y = None
        self.training_plaintexts = None
    
    def generate_dataset(self):
        if self.X is None or self.Y is None:
            self.X, self.Y = self.generate_training_data(self.n, self.nr, self.diff, self.neg, self.batch_size)

    def __len__(self):
        if self.Y is None:
            raise RuntimeError("Dataset not generated. Call generate_dataset() first.")
        return len(self.Y)

    def __getitem__(self, idx):
        if self.X is None or self.Y is None:
            raise RuntimeError("Dataset not generated. Call generate_dataset() first.")
        return self.X[idx], self.Y[idx]

    def generate_plaintext_triples(self, n, diff):
        p1_l = np.frombuffer(urandom(4*n), dtype=np.uint32)
        p1_r = np.frombuffer(urandom(4*n), dtype=np.uint32)

        p1_prime_l = p1_l ^ diff[0]
        p1_prime_r = p1_r ^ diff[1]
        
        p2_l = np.frombuffer(urandom(4*n), dtype=np.uint32)
        p2_r = np.frombuffer(urandom(4*n), dtype=np.uint32)
        
        p1 = np.column_stack([p1_l, p1_r])
        p1_prime = np.column_stack([p1_prime_l, p1_prime_r])
        p2 = np.column_stack([p2_l, p2_r])
        
        return p1, p1_prime, p2
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((0, 2), dtype=np.uint32), np.zeros((0, 2), dtype=np.uint32)
            p1_l = p1[:, 0]
            p1_r = p1[:, 1]
            p2_l = p2[:, 0]
            p2_r = p2[:, 1]
        else:
            if len(p1) != len(p2):
                raise ValueError("plaintext list length must be the same")
            n = len(p1)
            if n == 0:
                return [], []
            p1_l = np.array([pt[0] for pt in p1], dtype=np.uint32)
            p1_r = np.array([pt[1] for pt in p1], dtype=np.uint32)
            p2_l = np.array([pt[0] for pt in p2], dtype=np.uint32)
            p2_r = np.array([pt[1] for pt in p2], dtype=np.uint32)
        
        c1_l, c1_r = self.cipher.encrypt((p1_l, p1_r), self.key_array, nr)
        c2_l, c2_r = self.cipher.encrypt((p2_l, p2_r), self.key_array, nr)
        
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            c1 = np.column_stack([c1_l, c1_r])
            c2 = np.column_stack([c2_l, c2_r])
            return c1, c2
        else:
            c1_list = list(zip(c1_l, c1_r))
            c2_list = list(zip(c2_l, c2_r))
            return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=(0x40000000,0x00000000), neg='real_encryption', batch_size=10000):
        if neg not in ['real_encryption', 'random_bits']:
            raise ValueError(f"Invalid negative_samples: {neg}. Must be 'real_encryption' or 'random_bits'")
        
        p1, p1_prime, p2 = self.generate_plaintext_triples(n, diff)
        chosen = np.random.random(n) > 0.5
        
        if neg == 'real_encryption':
            self.training_plaintexts = np.concatenate([p1, p1_prime, p2], axis=0)
        else:
            self.training_plaintexts = np.concatenate([p1, p1_prime], axis=0)
        
        if neg == 'real_encryption':
            p2_selected = np.where(chosen[:, np.newaxis], p1_prime, p2)
            c1, c2 = self.encrypt_plaintext_pairs(p1, p2_selected, nr)
            Y = chosen.astype(np.uint8)
            
            c1_l = c1[:, 0]
            c1_r = c1[:, 1]
            c2_l = c2[:, 0]
            c2_r = c2[:, 1]
            
            X = self.cipher.to_bits([c1_l, c1_r, c2_l, c2_r])
        else:
            c1, c1_prime = self.encrypt_plaintext_pairs(p1, p1_prime, nr)
            Y = chosen.astype(np.uint8)
            
            c1_l = c1[:, 0]
            c1_r = c1[:, 1]
            c1_prime_l = c1_prime[:, 0]
            c1_prime_r = c1_prime[:, 1]
            
            feature_dim = 4 * self.cipher.word_size
            X = np.zeros((n, feature_dim), dtype=np.uint8)
            
            pos_mask = Y == 1
            if np.any(pos_mask):
                pos_X = self.cipher.to_bits([c1_l[pos_mask], c1_r[pos_mask], 
                                            c1_prime_l[pos_mask], c1_prime_r[pos_mask]])
                X[pos_mask] = pos_X
            
            neg_mask = Y == 0
            if np.any(neg_mask):
                num_neg = int(np.sum(neg_mask))
                rand_bits = (np.frombuffer(urandom(num_neg * feature_dim), dtype=np.uint8) & 1)\
                            .reshape(num_neg, feature_dim)
                X[neg_mask] = rand_bits
        
        return X, Y

    def _generate_keys(self, n, key_mode='random', key=None):
        if key_mode == 'random':
            single_key = None
            key_array = np.frombuffer(urandom(16*n), dtype=np.uint32).reshape(4, -1)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(16), dtype=np.uint32)
            key_array = single_key.reshape(4, 1).repeat(n, axis=1)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("key_mode is 'input_fixed' but no key provided")
            single_key = key
            key_array = np.array(key, dtype=np.uint32)
            if len(key_array) != 4:
                raise ValueError(f"Key must be 4 32-bit words for TEA, got {len(key_array)}")
            key_array = key_array.reshape(4, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Tea:
    def __init__(self, word_size=32):
        self.word_size = word_size
        self.mask_val = 2 ** 32 - 1

    def add_mod(self, v1, v2):
        return (v1 + v2) & self.mask_val

    def encrypt(self, p, k, r):
        v0, v1 = p[0], p[1]
        k0, k1, k2, k3 = k[0], k[1], k[2], k[3]
        delta = 0x9E3779B9
        s = 0
        for i in range(r):
            s = self.add_mod(s, delta)
            v0 = self.add_mod(v0, (self.add_mod((v1 << 4), k0) ^ self.add_mod(v1, s) ^ self.add_mod((v1 >> 5), k1)))
            v1 = self.add_mod(v1, (self.add_mod((v0 << 4), k2) ^ self.add_mod(v0, s) ^ self.add_mod((v0 >> 5), k3)))
        return (v0, v1)

    def decrypt(self, c, k, r):
        v0, v1 = c[0], c[1]
        k0, k1, k2, k3 = k[0], k[1], k[2], k[3]
        delta = 0x9E3779B9
        s = (delta * r) & self.mask_val
        for i in range(r):
            v1 = (v1 - ((self.add_mod((v0 << 4), k2) ^ self.add_mod(v0, s) ^ self.add_mod((v0 >> 5), k3)))) & self.mask_val
            s = (s - delta) & self.mask_val
            v0 = (v0 - ((self.add_mod((v1 << 4), k0) ^ self.add_mod(v1, s) ^ self.add_mod((v1 >> 5), k1)))) & self.mask_val
        return (v0, v1)

    def check_testvector(self):
        # No unified standard test vectors for TEA
        pass
    
    def to_bits(self, arr):
        block_num = len(arr)
        sample_num = len(arr[0])
        X = np.zeros((block_num * self.word_size, sample_num), dtype=np.uint8)
        for i in range(block_num * self.word_size):
            block_idx = i // self.word_size
            bit_offset = self.word_size - (i % self.word_size) - 1
            X[i] = (arr[block_idx] >> bit_offset) & 1
        return X.transpose()
