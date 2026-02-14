import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Speck_64_128_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x00000080,0x80000000), neg='real_encryption', batch_size=10000):
        self.cipher = Speck_64_128()
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
            p1_l = np.array([p[0] for p in p1], dtype=np.uint32)
            p1_r = np.array([p[1] for p in p1], dtype=np.uint32)
            p2_l = np.array([p[0] for p in p2], dtype=np.uint32)
            p2_r = np.array([p[1] for p in p2], dtype=np.uint32)
        
        ks = self.cipher.expand_key(self.key_array, nr)
        c1_l, c1_r = self.cipher.encrypt((p1_l, p1_r), ks)
        c2_l, c2_r = self.cipher.encrypt((p2_l, p2_r), ks)
        
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            c1 = np.column_stack([c1_l, c1_r])
            c2 = np.column_stack([c2_l, c2_r])
            return c1, c2
        else:
            c1_list = list(zip(c1_l, c1_r))
            c2_list = list(zip(c2_l, c2_r))
            return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=(0x00000080,0x80000000), neg='real_encryption', batch_size=10000):
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
                raise ValueError(f"Key must be 4 32-bit words for Speck64/128, got {len(key_array)}")
            key_array = key_array.reshape(4, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Speck_64_128:
    def __init__(self, word_size=32, alpha=8, beta=3):
        self.word_size = word_size
        self.alpha = alpha
        self.beta = beta
        self.mask_val = 2 ** word_size - 1

    def rol(self, x, k):
        return ((x << k) & self.mask_val) | (x >> (self.word_size - k))

    def ror(self, x, k):
        return (x >> k) | ((x << (self.word_size - k)) & self.mask_val)

    def enc_one_round(self, p, k):
        c0, c1 = p[0], p[1]
        c0 = self.ror(c0, self.alpha)
        c0 = (c0 + c1) & self.mask_val
        c0 = c0 ^ k
        c1 = self.rol(c1, self.beta)
        c1 = c1 ^ c0
        return (c0, c1)

    def dec_one_round(self, c, k):
        c0, c1 = c[0], c[1]
        c1 = c1 ^ c0
        c1 = self.ror(c1, self.beta)
        c0 = c0 ^ k
        c0 = (c0 - c1) & self.mask_val
        c0 = self.rol(c0, self.alpha)
        return (c0, c1)

    def expand_key(self, k, t):
        ks = [0 for _ in range(t)]
        ks[0] = k[len(k) - 1]
        l = list(reversed(k[:len(k) - 1]))
        for i in range(t - 1):
            l[i % 3], ks[i + 1] = self.enc_one_round((l[i % 3], ks[i]), i)
        return ks

    def encrypt(self, p, ks):
        x, y = p[0], p[1]
        for k in ks:
            x, y = self.enc_one_round((x, y), k)
        return (x, y)

    def decrypt(self, c, ks):
        x, y = c[0], c[1]
        for k in reversed(ks):
            x, y = self.dec_one_round((x, y), k)
        return (x, y)

    def check_testvector(self):
        key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
        pt = (0x3b726574, 0x7475432d)
        ks = self.expand_key(np.array(key, dtype=np.uint32).reshape(4, 1), 27)
        ct = self.encrypt(pt, ks)
        if ct == (0x8c6fa548, 0x454e028b):
            print("Testvector verified.")
            return True
        else:
            print("Testvector not verified.")
            return False
    
    def to_bits(self, arr):
        block_num = len(arr)
        sample_num = len(arr[0])
        X = np.zeros((block_num * self.word_size, sample_num), dtype=np.uint8)
        for i in range(block_num * self.word_size):
            block_idx = i // self.word_size
            bit_offset = self.word_size - (i % self.word_size) - 1
            X[i] = (arr[block_idx] >> bit_offset) & 1
        return X.transpose()
