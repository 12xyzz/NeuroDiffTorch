import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Chacha_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x00000000,0x00000000,0x00000000,0x00008000), neg='real_encryption', batch_size=10000):
        self.cipher = Chacha()
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
        p1 = np.frombuffer(urandom(16*n), dtype=np.uint32).reshape(n, 4)
        diff_array = np.array(diff, dtype=np.uint32)
        p1_prime = p1 ^ diff_array
        p2 = np.frombuffer(urandom(16*n), dtype=np.uint32).reshape(n, 4)
        return p1, p1_prime, p2
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((16, 0), dtype=np.uint32), np.zeros((16, 0), dtype=np.uint32)
            c1 = self.cipher.encrypt(p1.T, self.key_array, nr)
            c2 = self.cipher.encrypt(p2.T, self.key_array, nr)
            return c1, c2
        if len(p1) != len(p2):
            raise ValueError("plaintext list length must be the same")
        n = len(p1)
        if n == 0:
            return [], []
        p1_arr = np.array([list(p) for p in p1], dtype=np.uint32).T
        p2_arr = np.array([list(p) for p in p2], dtype=np.uint32).T
        c1 = self.cipher.encrypt(p1_arr, self.key_array, nr)
        c2 = self.cipher.encrypt(p2_arr, self.key_array, nr)
        c1_list = [tuple(c1[:, i]) for i in range(n)]
        c2_list = [tuple(c2[:, i]) for i in range(n)]
        return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=(0x00000000,0x00000000,0x00000000,0x00008000), neg='real_encryption', batch_size=10000):
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
            X = self.cipher.to_bits([c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3]])
        else:
            c1, c1_prime = self.encrypt_plaintext_pairs(p1, p1_prime, nr)
            Y = chosen.astype(np.uint8)
            feature_dim = 8 * self.cipher.word_size
            X = np.zeros((n, feature_dim), dtype=np.uint8)
            pos_mask = Y == 1
            if np.any(pos_mask):
                pos_X = self.cipher.to_bits([c1[0][pos_mask], c1[1][pos_mask], c1[2][pos_mask], c1[3][pos_mask],
                                            c1_prime[0][pos_mask], c1_prime[1][pos_mask],
                                            c1_prime[2][pos_mask], c1_prime[3][pos_mask]])
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
            key_array = np.frombuffer(urandom(32*n), dtype=np.uint32).reshape(8, -1)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(32), dtype=np.uint32)
            key_array = single_key.reshape(8, 1).repeat(n, axis=1)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("key_mode is 'input_fixed' but no key provided")
            single_key = key
            key_array = np.array(key, dtype=np.uint32)
            if len(key_array) != 8:
                raise ValueError(f"Key must be 8 32-bit words for ChaCha, got {len(key_array)}")
            key_array = key_array.reshape(8, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Chacha:
    def __init__(self, word_size=32):
        self.word_size = word_size
        self.rotations = [16, 12, 8, 7]

    def lrot(self, a, r):
        return (a << r) | (a >> (self.word_size - r))

    def quarter_round(self, a1, b1, c1, d1):
        a = a1 + b1
        d = self.lrot(d1 ^ a, self.rotations[0])
        c = c1 + d
        b = self.lrot(b1 ^ c, self.rotations[1])
        a += b
        d = self.lrot(d ^ a, self.rotations[2])
        c += d
        b = self.lrot(b ^ c, self.rotations[3])
        return a, b, c, d

    def encrypt(self, p, k, r, add_with_X0=False):
        k1, k2, k3, k4, k5, k6, k7, k8 = k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]
        p1, p2, n1, n2 = p[0], p[1], p[2], p[3]
        
        c1 = np.repeat(np.uint32([1634760805]), len(k1))
        c2 = np.repeat(np.uint32([857760878]), len(k1))
        c3 = np.repeat(np.uint32([2036477234]), len(k1))
        c4 = np.repeat(np.uint32([1797285236]), len(k1))
        
        X = np.array([c1, c2, c3, c4, k1, k2, k3, k4, k5, k6, k7, k8, p1, p2, n1, n2], dtype=np.uint32)
        
        for rr in range(r):
            if rr % 2 == 0:
                X[0], X[4], X[8], X[12] = self.quarter_round(X[0], X[4], X[8], X[12])
                X[1], X[5], X[9], X[13] = self.quarter_round(X[1], X[5], X[9], X[13])
                X[2], X[6], X[10], X[14] = self.quarter_round(X[2], X[6], X[10], X[14])
                X[3], X[7], X[11], X[15] = self.quarter_round(X[3], X[7], X[11], X[15])
            else:
                X[0], X[5], X[10], X[15] = self.quarter_round(X[0], X[5], X[10], X[15])
                X[1], X[6], X[11], X[12] = self.quarter_round(X[1], X[6], X[11], X[12])
                X[2], X[7], X[8], X[13] = self.quarter_round(X[2], X[7], X[8], X[13])
                X[3], X[4], X[9], X[14] = self.quarter_round(X[3], X[4], X[9], X[14])
        
        if add_with_X0:
            return X + np.array([c1, c2, c3, c4, k1, k2, k3, k4, k5, k6, k7, k8, p1, p2, n1, n2], dtype=np.uint32)
        return X

    def check_testvector(self):
        k = np.array([[0x03020100], [0x07060504], [0x0b0a0908], [0x0f0e0d0c],
                      [0x13121110], [0x17161514], [0x1b1a1918], [0x1f1e1d1c]], dtype=np.uint32).reshape(8, 1)
        p = np.array([[0x00000001], [0x09000000], [0x4a000000], [0x00000000]], dtype=np.uint32).reshape(4, 1)
        stream = np.array([
            0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
            0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
            0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
            0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2
        ], dtype=np.uint32)

        result = self.encrypt(p, k, 20, add_with_X0=True)
        if np.all(result.flatten() == stream):
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
