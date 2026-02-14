import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Lea_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x00000080,0x00000080,0x00000080,0x04000080), neg='real_encryption', batch_size=10000):
        self.cipher = Lea()
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
        
        diff_array = np.array([diff[0], diff[1], diff[2], diff[3]], dtype=np.uint32)
        p1_prime = p1 ^ diff_array
        
        p2 = np.frombuffer(urandom(16*n), dtype=np.uint32).reshape(n, 4)
        
        return p1, p1_prime, p2
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((4, 0), dtype=np.uint32), np.zeros((4, 0), dtype=np.uint32)
            p1_t = p1.T.astype(np.uint32)
            p2_t = p2.T.astype(np.uint32)
            c1 = self.cipher.encrypt(p1_t, self.key_array, nr)
            c2 = self.cipher.encrypt(p2_t, self.key_array, nr)
            return c1, c2
        else:
            if len(p1) != len(p2):
                raise ValueError("plaintext list length must be the same")
            n = len(p1)
            if n == 0:
                return [], []
            p1_t = np.array([list(pt) for pt in p1], dtype=np.uint32).T
            p2_t = np.array([list(pt) for pt in p2], dtype=np.uint32).T
            c1 = self.cipher.encrypt(p1_t, self.key_array, nr)
            c2 = self.cipher.encrypt(p2_t, self.key_array, nr)
            c1_list = [tuple(c1[:, i]) for i in range(n)]
            c2_list = [tuple(c2[:, i]) for i in range(n)]
            return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=(0x00000080,0x00000080,0x00000080,0x04000080), neg='real_encryption', batch_size=10000):
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
                raise ValueError(f"Key must be 4 32-bit words for LEA, got {len(key_array)}")
            key_array = key_array.reshape(4, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Lea:
    def __init__(self, word_size=32):
        self.word_size = word_size
        self.mask_val = 2 ** word_size - 1
        self.DELTA = np.array([0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec, 
                               0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957], dtype=np.uint32)

    def rol(self, x, k):
        return ((x << k) & self.mask_val) | (x >> (self.word_size - k))

    def ror(self, x, k):
        return (x >> k) | ((x << (self.word_size - k)) & self.mask_val)

    def expand_key(self, K, t):
        ks = []
        tmp = [K[i].copy() for i in range(4)]
        for i in range(t):
            tmp[0] = self.rol((tmp[0] + self.rol(self.DELTA[i % 4], i)) & self.mask_val, 1)
            tmp[1] = self.rol((tmp[1] + self.rol(self.DELTA[i % 4], i+1)) & self.mask_val, 3)
            tmp[2] = self.rol((tmp[2] + self.rol(self.DELTA[i % 4], i+2)) & self.mask_val, 6)
            tmp[3] = self.rol((tmp[3] + self.rol(self.DELTA[i % 4], i+3)) & self.mask_val, 11)
            ks.append(np.array([tmp[0], tmp[1], tmp[2], tmp[1], tmp[3], tmp[1]]))
        return np.array(ks)

    def encrypt(self, p, k, r):
        P = p.copy()
        K = k.copy()
        if P.dtype == np.uint32:
            P = P.byteswap()
            K = K.byteswap()
        ks = self.expand_key(K, r)
        for i in range(r):
            p0, p1, p2, p3 = P[0].copy(), P[1].copy(), P[2].copy(), P[3].copy()
            k0, k1, k2, k3, k4, k5 = ks[i]
            P[3] = p0
            P[0] = self.rol(((p0 ^ k0) + (p1 ^ k1)) & self.mask_val, 9)
            P[1] = self.ror(((p1 ^ k2) + (p2 ^ k3)) & self.mask_val, 5)
            P[2] = self.ror(((p2 ^ k4) + (p3 ^ k5)) & self.mask_val, 3)
        if P.dtype == np.uint32:
            P = P.byteswap()
        return P

    def check_testvector(self):
        p = np.array([0x10111213, 0x14151617, 0x18191a1b, 0x1c1d1e1f], dtype=np.uint32).reshape(4, 1)
        k = np.array([0x0f1e2d3c, 0x4b5a6978, 0x8796a5b4, 0xc3d2e1f0], dtype=np.uint32).reshape(4, 1)
        e = np.array([0x9fc84e35, 0x28c6c618, 0x5532c7a7, 0x04648bfd], dtype=np.uint32)
        c = self.encrypt(p, k, 24)
        if np.all(c.flatten() == e):
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
