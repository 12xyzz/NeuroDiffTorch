import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Speck_32_64_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x0040,0), real_diff=False):
        self.speck = Speck_32_64()
        self.single_key, self.key_array = self._generate_keys(n, key_mode, key)
        
        if real_diff:
            self.X, self.Y = self.real_differences_data(n, nr, diff)
        else:
            self.X, self.Y = self.random_differences_data(n, nr, diff)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def random_differences_data(self, n, nr, diff=(0x0040,0)):
        Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
        plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
        plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]
        num_rand_samples = np.sum(Y==0)
        if num_rand_samples > 0:
            plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
            plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
        ks = self.speck.expand_key(self.key_array, nr)
        ctdata0l, ctdata0r = self.speck.encrypt((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = self.speck.encrypt((plain1l, plain1r), ks)
        X = self.speck.to_bits([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
        return X, Y
        
    def real_differences_data(self, n, nr, diff=(0x0040,0)):
        Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1 
        plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
        plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]
        num_rand_samples = np.sum(Y==0)
        ks = self.speck.expand_key(self.key_array, nr)
        ctdata0l, ctdata0r = self.speck.encrypt((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = self.speck.encrypt((plain1l, plain1r), ks)
        if num_rand_samples > 0:
            k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
            k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
            ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1
            ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1
        X = self.speck.to_bits([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
        return X, Y

    def _generate_keys(self, n, key_mode='random', key=None):
        if key_mode == 'random':
            single_key = None
            key_array = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(8), dtype=np.uint16)
            key_array = single_key.reshape(4, 1).repeat(n, axis=1)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("KEY_MODE is 'input_fixed' but no key provided")
            single_key = key
            key_array = np.array(key, dtype=np.uint16)
            if len(key_array) != 4:
                raise ValueError(f"Key must be 4 16-bit words for Speck32/64, got {len(key_array)}")
            key_array = key_array.reshape(4, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Speck_32_64:
    def __init__(self, word_size=16, alpha=7, beta=2):
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
        key = (0x1918, 0x1110, 0x0908, 0x0100)
        pt = (0x6574, 0x694c)
        ks = self.expand_key(key, 22)
        ct = self.encrypt(pt, ks)
        if ct == (0xa868, 0x42f2):
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