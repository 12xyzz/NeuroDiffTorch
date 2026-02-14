import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Katan_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=0x80240, neg='real_encryption', batch_size=10000):
        self.cipher = Katan()
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
        if isinstance(diff, (tuple, list)):
            raise TypeError(f"diff must be a single integer value, not a {type(diff).__name__}. "
                          f"For KATAN32, use diff=<int> instead of diff=(<int>, <int>)")
        
        p1 = np.frombuffer(urandom(4*n), dtype=np.uint8).reshape(n, 4)
        p1_bits = self.cipher.from_bytes_to_bits(p1)
        
        diff_bytes = np.frombuffer(np.array([diff], dtype=np.uint32).tobytes(), dtype=np.uint8).reshape(1, 4)
        diff_bits = self.cipher.from_bytes_to_bits(diff_bytes)
        p1_prime_bits = p1_bits ^ diff_bits
        
        p2 = np.frombuffer(urandom(4*n), dtype=np.uint8).reshape(n, 4)
        p2_bits = self.cipher.from_bytes_to_bits(p2)
        
        return p1_bits, p1_prime_bits, p2_bits
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((0, 32), dtype=np.uint8), np.zeros((0, 32), dtype=np.uint8)
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
    
    def generate_training_data(self, n, nr, diff=0x80240, neg='real_encryption', batch_size=10000):
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
            
            feature_dim = 2 * 32
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
                    raise ValueError(f"Key must be 10 bytes (80 bits) for KATAN32, got {key_bytes.shape[1]}")
            else:
                raise ValueError("Key must be array-like")
            key_bits = self.cipher.from_bytes_to_bits(key_bytes)
            key_array = np.repeat(key_bits, n, axis=0)
            single_key = key
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Katan:
    def __init__(self, plain_bits=32, key_bits=80):
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.word_size = plain_bits
        
        if plain_bits == 32:
            self.LEN_L1 = 13
            self.LEN_L2 = 19
            self.X = (None, 12, 7, 8, 5, 3)
            self.Y = (None, 18, 7, 12, 10, 8, 3)
        elif plain_bits == 48:
            self.LEN_L1 = 19
            self.LEN_L2 = 29
            self.X = (None, 18, 12, 15, 7, 6)
            self.Y = (None, 28, 19, 21, 13, 15, 6)
        else:  # 64 bits
            self.LEN_L1 = 25
            self.LEN_L2 = 39
            self.X = (None, 24, 15, 20, 11, 9)
            self.Y = (None, 38, 25, 33, 21, 14, 9)
        
        self.IR = (
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
            1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
            1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
            0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
            1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
            1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1,
            0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
            1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0
        )

    def from_bytes_to_bits(self, arr):
        return np.unpackbits(arr, axis=1)

    def from_bits_to_bytes(self, arr):
        return np.packbits(arr, axis=1)

    def encrypt(self, plaintext, K, nr):
        P = np.flip(plaintext, axis=1)
        ks = np.zeros(shape=(len(P), max(2*nr, 80)), dtype=np.uint8)
        ks[:, :80] = np.flip(K, axis=1)
        
        for i in range(80, nr*2):
            ks[:, i] = ks[:, i-80] ^ ks[:, i-61] ^ ks[:, i-50] ^ ks[:, i-13]
        
        for i in range(nr):
            fa = (P[:, self.LEN_L2 + self.X[1]] ^ P[:, self.LEN_L2 + self.X[2]] ^ 
                  ks[:, 2*i] ^ 
                  (P[:, self.LEN_L2 + self.X[3]] & P[:, self.LEN_L2 + self.X[4]]) ^ 
                  (P[:, self.LEN_L2 + self.X[5]] & self.IR[i]))
            fb = (P[:, self.Y[1]] ^ P[:, self.Y[2]] ^ 
                  (P[:, self.Y[3]] & P[:, self.Y[4]]) ^ 
                  (P[:, self.Y[5]] & P[:, self.Y[6]]) ^ 
                  ks[:, 2*i+1])
            P = np.roll(P, 1, axis=1)
            P[:, 0] = fa
            P[:, self.LEN_L2] = fb
        
        return np.flip(P, axis=1)

    def check_testvector(self):
        p = np.zeros((2, self.plain_bits), dtype=np.uint8)
        p[1] ^= 1
        k = np.zeros((2, self.key_bits), dtype=np.uint8)
        k[0] ^= 1
        C = self.from_bits_to_bytes(self.encrypt(p, k, 254))
        if self.plain_bits == 32:
            expected_0 = 0x7e1ff945
            expected_1 = 0x432e61da
            result_0 = int.from_bytes(C[0].tobytes(), byteorder='big')
            result_1 = int.from_bytes(C[1].tobytes(), byteorder='big')
            if result_0 == expected_0 and result_1 == expected_1:
                print("Testvector verified.")
                return True
            else:
                print(f"Testvector not verified.")
                return False
        return True
    
    def to_bits(self, arr):
        # arr is already in bits for KATAN
        return arr
