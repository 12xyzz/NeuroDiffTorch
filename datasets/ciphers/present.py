import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Present_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=0xd000000, neg='real_encryption', batch_size=10000):
        self.cipher = Present()
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
        # PRESENT80 uses 64-bit blocks
        if isinstance(diff, (tuple, list)):
            raise TypeError(f"diff must be a single integer value, not a {type(diff).__name__}. "
                          f"For PRESENT-80, use diff=<int> instead of diff=(<int>, <int>)")
        
        p1 = np.frombuffer(urandom(8*n), dtype=np.uint8).reshape(n, 8)
        p1_bits = self.cipher.from_bytes_to_bits(p1)
        
        diff_bytes = np.frombuffer(np.array([diff], dtype=np.uint64).tobytes(), dtype=np.uint8).reshape(1, 8)
        diff_bits = self.cipher.from_bytes_to_bits(diff_bytes)
        p1_prime_bits = p1_bits ^ diff_bits
        
        p2 = np.frombuffer(urandom(8*n), dtype=np.uint8).reshape(n, 8)
        p2_bits = self.cipher.from_bytes_to_bits(p2)
        
        return p1_bits, p1_prime_bits, p2_bits
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((0, 64), dtype=np.uint8), np.zeros((0, 64), dtype=np.uint8)
            p1_bits = p1
            p2_bits = p2
        else:
            if len(p1) != len(p2):
                raise ValueError("plaintext list length must be the same")
            n = len(p1)
            if n == 0:
                return [], []
            p1_bits = np.array([list(p) for p in p1], dtype=np.uint8)
            p2_bits = np.array([list(p) for p in p2], dtype=np.uint8)
        
        c1_bits = self.cipher.encrypt(p1_bits, self.key_array, nr)
        c2_bits = self.cipher.encrypt(p2_bits, self.key_array, nr)
        
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            return c1_bits, c2_bits
        else:
            c1_list = [tuple(c1_bits[i]) for i in range(n)]
            c2_list = [tuple(c2_bits[i]) for i in range(n)]
            return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=0xd000000, neg='real_encryption', batch_size=10000):
        if neg not in ['real_encryption', 'random_bits']:
            raise ValueError(f"Invalid negative_samples: {neg}. Must be 'real_encryption' or 'random_bits'")
        
        p1_bits, p1_prime_bits, p2_bits = self.generate_plaintext_triples(n, diff)
        
        chosen = np.random.random(n) > 0.5

        if neg == 'real_encryption':
            self.training_plaintexts = np.concatenate([p1_bits, p1_prime_bits, p2_bits], axis=0)
        else:
            self.training_plaintexts = np.concatenate([p1_bits, p1_prime_bits], axis=0)
        
        if neg == 'real_encryption':
            p2_selected = np.where(chosen[:, np.newaxis], p1_prime_bits, p2_bits)
            c1_bits = self.cipher.encrypt(p1_bits, self.key_array, nr)
            c2_bits = self.cipher.encrypt(p2_selected, self.key_array, nr)
            Y = chosen.astype(np.uint8)
            
            X = np.concatenate([c1_bits, c2_bits], axis=1)
        else:
            c1_bits = self.cipher.encrypt(p1_bits, self.key_array, nr)
            c1_prime_bits = self.cipher.encrypt(p1_prime_bits, self.key_array, nr)
            Y = chosen.astype(np.uint8)
            
            feature_dim = 2 * 64
            X = np.zeros((n, feature_dim), dtype=np.uint8)
            
            pos_mask = Y == 1
            if np.any(pos_mask):
                pos_X = np.concatenate([c1_bits[pos_mask], c1_prime_bits[pos_mask]], axis=1)
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
            key_bytes = np.frombuffer(urandom(10*n), dtype=np.uint8).reshape(n, 10)
            key_array = self.cipher.from_bytes_to_bits(key_bytes)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(10), dtype=np.uint8).reshape(1, 10)
            key_bits = self.cipher.from_bytes_to_bits(single_key)
            key_array = np.repeat(key_bits, n, axis=0)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("key_mode is 'input_fixed' but no key provided")
            if isinstance(key, (list, tuple, np.ndarray)):
                key_bytes = np.array(key, dtype=np.uint8).reshape(1, -1)
                if key_bytes.shape[1] != 10:
                    raise ValueError(f"Key must be 10 bytes (80 bits) for PRESENT80, got {key_bytes.shape[1]}")
            else:
                raise ValueError("Key must be array-like")
            key_bits = self.cipher.from_bytes_to_bits(key_bytes)
            key_array = np.repeat(key_bits, n, axis=0)
            single_key = key
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Present:
    def __init__(self, word_size=64):
        self.word_size = 64
        self.key_size = 80
        self.sbox = np.uint8([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2])
        self.pbox = np.uint8([0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                              4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                              8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                              12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63])

    def from_bytes_to_bits(self, arr):
        return np.unpackbits(arr, axis=1)

    def from_bits_to_bytes(self, arr):
        return np.packbits(arr, axis=1)

    def sbox_layer(self, arr):
        num_words = arr.shape[1] // 4
        S = arr.copy()
        for i in range(num_words):
            to_sub = np.zeros(arr.shape[0], dtype=np.uint8)
            for j in range(4):
                pos = 4 * i + j
                to_sub += (2 ** (3 - j)) * arr[:, pos]
            S[:, 4*i:4*(i+1)] = np.unpackbits(self.sbox[to_sub[:, None]], axis=1)[:, -4:]
        return S

    def pbox_layer(self, arr):
        result = arr.copy()
        result[:, self.pbox] = arr[:, np.arange(64)]
        return result

    def inverse_pbox_layer(self, arr):
        result = arr.copy()
        inv_pbox = np.zeros(64, dtype=np.uint8)
        for i in range(64):
            inv_pbox[self.pbox[i]] = i
        result[:, inv_pbox] = arr[:, np.arange(64)]
        return result

    def expand_key(self, k, t):
        ks = []
        key = k.copy()
        for r in range(t):
            ks.append(key[:, :64])
            key = np.roll(key, 19, axis=1)
            key[:, :4] = self.sbox_layer(key[:, :4])
            round_const = np.unpackbits(np.uint8(r+1))
            key[:, -23:-15] ^= round_const[:8]
        return ks

    def encrypt(self, p, k, r):
        ks = self.expand_key(k, r)
        c = p.copy()
        for i in range(r-1):
            c ^= ks[i]
            c = self.sbox_layer(c)
            c = self.pbox_layer(c)
        return c ^ ks[-1]

    def decrypt(self, c, k, r):
        ks = self.expand_key(k, r)
        p = c.copy()
        p ^= ks[-1]
        for i in range(r-2, -1, -1):
            p = self.inverse_pbox_layer(p)
            # Inverse S-box
            p = self.inverse_sbox_layer(p)
            p ^= ks[i]
        return p

    def inverse_sbox_layer(self, arr):
        inv_sbox = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            inv_sbox[self.sbox[i]] = i
        num_words = arr.shape[1] // 4
        S = arr.copy()
        for i in range(num_words):
            to_sub = np.zeros(arr.shape[0], dtype=np.uint8)
            for j in range(4):
                pos = 4 * i + j
                to_sub += (2 ** (3 - j)) * arr[:, pos]
            S[:, 4*i:4*(i+1)] = np.unpackbits(inv_sbox[to_sub[:, None]], axis=1)[:, -4:]
        return S

    def check_testvector(self):
        p = np.zeros((1, 64), dtype=np.uint8)
        k = np.zeros((1, 80), dtype=np.uint8)
        c = self.encrypt(p, k, 32)
        c_bytes = self.from_bits_to_bytes(c)
        expected = 0x5579c1387b228445
        result = int.from_bytes(c_bytes[0].tobytes(), byteorder='big')
        if result == expected:
            print("Testvector verified.")
            return True
        else:
            print(f"Testvector not verified. Got {hex(result)}, expected {hex(expected)}")
            return False
    
    def to_bits(self, arr):
        # arr is already in bits for PRESENT
        return arr
