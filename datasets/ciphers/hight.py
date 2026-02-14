import numpy as np
from torch.utils.data import Dataset
from os import urandom


class Hight_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=(0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00), neg='real_encryption', batch_size=10000):
        self.cipher = Hight()
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
        p1 = np.frombuffer(urandom(8*n), dtype=np.uint8).reshape(n, 8)
        diff_array = np.array(diff, dtype=np.uint8)
        p1_prime = p1 ^ diff_array
        p2 = np.frombuffer(urandom(8*n), dtype=np.uint8).reshape(n, 8)
        return p1, p1_prime, p2
    
    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((8, 0), dtype=np.uint8), np.zeros((8, 0), dtype=np.uint8)
            c1 = self.cipher.encrypt(p1.T, self.key_array, nr)
            c2 = self.cipher.encrypt(p2.T, self.key_array, nr)
            return c1, c2
        if len(p1) != len(p2):
            raise ValueError("plaintext list length must be the same")
        n = len(p1)
        if n == 0:
            return [], []
        p1_arr = np.array([list(p) for p in p1], dtype=np.uint8).T
        p2_arr = np.array([list(p) for p in p2], dtype=np.uint8).T
        c1 = self.cipher.encrypt(p1_arr, self.key_array, nr)
        c2 = self.cipher.encrypt(p2_arr, self.key_array, nr)
        c1_list = [tuple(c1[:, i]) for i in range(n)]
        c2_list = [tuple(c2[:, i]) for i in range(n)]
        return c1_list, c2_list
    
    def generate_training_data(self, n, nr, diff=(0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00), neg='real_encryption', batch_size=10000):
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
            X = self.cipher.to_bits([c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7],
                                    c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7]])
        else:
            c1, c1_prime = self.encrypt_plaintext_pairs(p1, p1_prime, nr)
            Y = chosen.astype(np.uint8)
            feature_dim = 16 * 8
            X = np.zeros((n, feature_dim), dtype=np.uint8)
            pos_mask = Y == 1
            if np.any(pos_mask):
                pos_X = self.cipher.to_bits([c1[0][pos_mask], c1[1][pos_mask], c1[2][pos_mask], c1[3][pos_mask],
                                            c1[4][pos_mask], c1[5][pos_mask], c1[6][pos_mask], c1[7][pos_mask],
                                            c1_prime[0][pos_mask], c1_prime[1][pos_mask], c1_prime[2][pos_mask],
                                            c1_prime[3][pos_mask], c1_prime[4][pos_mask], c1_prime[5][pos_mask],
                                            c1_prime[6][pos_mask], c1_prime[7][pos_mask]])
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
            key_array = np.frombuffer(urandom(16*n), dtype=np.uint8).reshape(16, -1)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(16), dtype=np.uint8)
            key_array = single_key.reshape(16, 1).repeat(n, axis=1)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("key_mode is 'input_fixed' but no key provided")
            single_key = key
            key_array = np.array(key, dtype=np.uint8)
            if len(key_array) != 16:
                raise ValueError(f"Key must be 16 bytes (128 bits) for HIGHT, got {len(key_array)}")
            key_array = key_array.reshape(16, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array

class Hight:
    def __init__(self, word_size=8):
        self.word_size = word_size
        
        self.DELTA = np.array([
            0x5A,0x6D,0x36,0x1B,0x0D,0x06,0x03,0x41,0x60,0x30,0x18,0x4C,0x66,0x33,0x59,0x2C,
            0x56,0x2B,0x15,0x4A,0x65,0x72,0x39,0x1C,0x4E,0x67,0x73,0x79,0x3C,0x5E,0x6F,0x37,
            0x5B,0x2D,0x16,0x0B,0x05,0x42,0x21,0x50,0x28,0x54,0x2A,0x55,0x6A,0x75,0x7A,0x7D,
            0x3E,0x5F,0x2F,0x17,0x4B,0x25,0x52,0x29,0x14,0x0A,0x45,0x62,0x31,0x58,0x6C,0x76,
            0x3B,0x1D,0x0E,0x47,0x63,0x71,0x78,0x7C,0x7E,0x7F,0x3F,0x1F,0x0F,0x07,0x43,0x61,
            0x70,0x38,0x5C,0x6E,0x77,0x7B,0x3D,0x1E,0x4F,0x27,0x53,0x69,0x34,0x1A,0x4D,0x26,
            0x13,0x49,0x24,0x12,0x09,0x04,0x02,0x01,0x40,0x20,0x10,0x08,0x44,0x22,0x11,0x48,
            0x64,0x32,0x19,0x0C,0x46,0x23,0x51,0x68,0x74,0x3A,0x5D,0x2E,0x57,0x6B,0x35,0x5A
        ], dtype=np.uint8)
        
        self.HIGHT_F0 = np.array([
            0x00,0x86,0x0D,0x8B,0x1A,0x9C,0x17,0x91,0x34,0xB2,0x39,0xBF,0x2E,0xA8,0x23,0xA5,
            0x68,0xEE,0x65,0xE3,0x72,0xF4,0x7F,0xF9,0x5C,0xDA,0x51,0xD7,0x46,0xC0,0x4B,0xCD,
            0xD0,0x56,0xDD,0x5B,0xCA,0x4C,0xC7,0x41,0xE4,0x62,0xE9,0x6F,0xFE,0x78,0xF3,0x75,
            0xB8,0x3E,0xB5,0x33,0xA2,0x24,0xAF,0x29,0x8C,0x0A,0x81,0x07,0x96,0x10,0x9B,0x1D,
            0xA1,0x27,0xAC,0x2A,0xBB,0x3D,0xB6,0x30,0x95,0x13,0x98,0x1E,0x8F,0x09,0x82,0x04,
            0xC9,0x4F,0xC4,0x42,0xD3,0x55,0xDE,0x58,0xFD,0x7B,0xF0,0x76,0xE7,0x61,0xEA,0x6C,
            0x71,0xF7,0x7C,0xFA,0x6B,0xED,0x66,0xE0,0x45,0xC3,0x48,0xCE,0x5F,0xD9,0x52,0xD4,
            0x19,0x9F,0x14,0x92,0x03,0x85,0x0E,0x88,0x2D,0xAB,0x20,0xA6,0x37,0xB1,0x3A,0xBC,
            0x43,0xC5,0x4E,0xC8,0x59,0xDF,0x54,0xD2,0x77,0xF1,0x7A,0xFC,0x6D,0xEB,0x60,0xE6,
            0x2B,0xAD,0x26,0xA0,0x31,0xB7,0x3C,0xBA,0x1F,0x99,0x12,0x94,0x05,0x83,0x08,0x8E,
            0x93,0x15,0x9E,0x18,0x89,0x0F,0x84,0x02,0xA7,0x21,0xAA,0x2C,0xBD,0x3B,0xB0,0x36,
            0xFB,0x7D,0xF6,0x70,0xE1,0x67,0xEC,0x6A,0xCF,0x49,0xC2,0x44,0xD5,0x53,0xD8,0x5E,
            0xE2,0x64,0xEF,0x69,0xF8,0x7E,0xF5,0x73,0xD6,0x50,0xDB,0x5D,0xCC,0x4A,0xC1,0x47,
            0x8A,0x0C,0x87,0x01,0x90,0x16,0x9D,0x1B,0xBE,0x38,0xB3,0x35,0xA4,0x22,0xA9,0x2F,
            0x32,0xB4,0x3F,0xB9,0x28,0xAE,0x25,0xA3,0x06,0x80,0x0B,0x8D,0x1C,0x9A,0x11,0x97,
            0x5A,0xDC,0x57,0xD1,0x40,0xC6,0x4D,0xCB,0x6E,0xE8,0x63,0xE5,0x74,0xF2,0x79,0xFF
        ], dtype=np.uint8)
        
        self.HIGHT_F1 = np.array([
            0x00,0x58,0xB0,0xE8,0x61,0x39,0xD1,0x89,0xC2,0x9A,0x72,0x2A,0xA3,0xFB,0x13,0x4B,
            0x85,0xDD,0x35,0x6D,0xE4,0xBC,0x54,0x0C,0x47,0x1F,0xF7,0xAF,0x26,0x7E,0x96,0xCE,
            0x0B,0x53,0xBB,0xE3,0x6A,0x32,0xDA,0x82,0xC9,0x91,0x79,0x21,0xA8,0xF0,0x18,0x40,
            0x8E,0xD6,0x3E,0x66,0xEF,0xB7,0x5F,0x07,0x4C,0x14,0xFC,0xA4,0x2D,0x75,0x9D,0xC5,
            0x16,0x4E,0xA6,0xFE,0x77,0x2F,0xC7,0x9F,0xD4,0x8C,0x64,0x3C,0xB5,0xED,0x05,0x5D,
            0x93,0xCB,0x23,0x7B,0xF2,0xAA,0x42,0x1A,0x51,0x09,0xE1,0xB9,0x30,0x68,0x80,0xD8,
            0x1D,0x45,0xAD,0xF5,0x7C,0x24,0xCC,0x94,0xDF,0x87,0x6F,0x37,0xBE,0xE6,0x0E,0x56,
            0x98,0xC0,0x28,0x70,0xF9,0xA1,0x49,0x11,0x5A,0x02,0xEA,0xB2,0x3B,0x63,0x8B,0xD3,
            0x2C,0x74,0x9C,0xC4,0x4D,0x15,0xFD,0xA5,0xEE,0xB6,0x5E,0x06,0x8F,0xD7,0x3F,0x67,
            0xA9,0xF1,0x19,0x41,0xC8,0x90,0x78,0x20,0x6B,0x33,0xDB,0x83,0x0A,0x52,0xBA,0xE2,
            0x27,0x7F,0x97,0xCF,0x46,0x1E,0xF6,0xAE,0xE5,0xBD,0x55,0x0D,0x84,0xDC,0x34,0x6C,
            0xA2,0xFA,0x12,0x4A,0xC3,0x9B,0x73,0x2B,0x60,0x38,0xD0,0x88,0x01,0x59,0xB1,0xE9,
            0x3A,0x62,0x8A,0xD2,0x5B,0x03,0xEB,0xB3,0xF8,0xA0,0x48,0x10,0x99,0xC1,0x29,0x71,
            0xBF,0xE7,0x0F,0x57,0xDE,0x86,0x6E,0x36,0x7D,0x25,0xCD,0x95,0x1C,0x44,0xAC,0xF4,
            0x31,0x69,0x81,0xD9,0x50,0x08,0xE0,0xB8,0xF3,0xAB,0x43,0x1B,0x92,0xCA,0x22,0x7A,
            0xB4,0xEC,0x04,0x5C,0xD5,0x8D,0x65,0x3D,0x76,0x2E,0xC6,0x9E,0x17,0x4F,0xA7,0xFF
        ], dtype=np.uint8)

    def key_schedule(self, key):
        n = key.shape[1] if key.ndim > 1 else 1
        if key.ndim == 1:
            key = key.reshape(16, 1)
        WK = np.zeros((8, n), dtype=np.uint8)
        RK = np.zeros((128, n), dtype=np.uint8)
        for i in range(4):
            WK[i] = key[i+12]
            WK[i+4] = key[i]
        for i in range(8):
            for j in range(8):
                RK[16*i+j] = (key[(j-i) & 7].astype(np.uint16) + self.DELTA[16*i+j]) & 0xff
            for j in range(8):
                RK[16*i+j+8] = (key[((j-i) & 7) + 8].astype(np.uint16) + self.DELTA[16*i+j+8]) & 0xff
        return WK, RK

    def round_enc(self, round_key, pt, k):
        X = pt.copy()
        X[1] = X[0]
        X[3] = X[2]
        X[5] = X[4]
        X[7] = X[6]
        X[0] = (pt[7] ^ (self.HIGHT_F0[pt[6]] + round_key[4*k+3]))
        X[2] = (pt[1] + (self.HIGHT_F1[pt[0]] ^ round_key[4*k]))
        X[4] = (pt[3] ^ (self.HIGHT_F0[pt[2]] + round_key[4*k+1]))
        X[6] = (pt[5] + (self.HIGHT_F1[pt[4]] ^ round_key[4*k+2]))
        return X

    def round_dec(self, round_key, ct, k):
        pt = ct.copy()
        pt[0] = ct[1]
        pt[6] = ct[7]
        pt[7] = ct[0] ^ (self.HIGHT_F0[ct[7]] + round_key[4*k+3])
        pt[4] = ct[5]
        pt[1] = (ct[2].astype(np.uint16) - (self.HIGHT_F1[ct[1]] ^ round_key[4*k]).astype(np.uint16)) & 0xff
        pt[2] = ct[3]
        pt[3] = ct[4] ^ (self.HIGHT_F0[ct[3]] + round_key[4*k+1])
        pt[5] = (ct[6].astype(np.uint16) - (self.HIGHT_F1[ct[5]] ^ round_key[4*k+2]).astype(np.uint16)) & 0xff
        return pt

    def encrypt(self, pt, key, rounds):
        pt = np.flip(pt.copy(), axis=0)
        key = np.flip(key.copy(), axis=0)
        whitening_key, round_key = self.key_schedule(key)
        
        # Initial transformation
        pt[0] = (pt[0] + whitening_key[0])
        pt[2] = (pt[2] ^ whitening_key[1])
        pt[4] = (pt[4] + whitening_key[2])
        pt[6] = (pt[6] ^ whitening_key[3])
        
        for i in range(rounds-1):
            pt = self.round_enc(round_key, pt, i)
        
        # Final round
        X = pt.copy()
        X[1] = pt[1] + (self.HIGHT_F1[pt[0]] ^ round_key[(rounds-1) * 4])
        X[3] = pt[3] ^ (self.HIGHT_F0[pt[2]] + round_key[(rounds-1) * 4 + 1])
        X[5] = pt[5] + (self.HIGHT_F1[pt[4]] ^ round_key[(rounds-1) * 4 + 2])
        X[7] = pt[7] ^ (self.HIGHT_F0[pt[6]] + round_key[(rounds-1) * 4 + 3])
        X[0] += whitening_key[4]
        X[2] ^= whitening_key[5]
        X[4] += whitening_key[6]
        X[6] ^= whitening_key[7]
        
        return np.flip(X, axis=0)

    def decrypt(self, ct, key, rounds):
        ct = np.flip(ct.copy(), axis=0)
        key = np.flip(key.copy(), axis=0)
        whitening_key, round_key = self.key_schedule(key)

        # Reverse final whitening
        Y = ct.copy()
        Y[0] = (ct[0].astype(np.uint16) - whitening_key[4].astype(np.uint16)) & 0xff
        Y[2] = ct[2] ^ whitening_key[5]
        Y[4] = (ct[4].astype(np.uint16) - whitening_key[6].astype(np.uint16)) & 0xff
        Y[6] = ct[6] ^ whitening_key[7]
        Y[1], Y[3], Y[5], Y[7] = ct[1], ct[3], ct[5], ct[7]

        # Inverse of final round
        pt = Y.copy()
        pt[0], pt[2], pt[4], pt[6] = Y[0], Y[2], Y[4], Y[6]
        pt[1] = (Y[1].astype(np.uint16) - (self.HIGHT_F1[Y[0]] ^ round_key[(rounds-1)*4]).astype(np.uint16)) & 0xff
        pt[3] = Y[3] ^ (self.HIGHT_F0[Y[2]] + round_key[(rounds-1)*4+1])
        pt[5] = (Y[5].astype(np.uint16) - (self.HIGHT_F1[Y[4]] ^ round_key[(rounds-1)*4+2]).astype(np.uint16)) & 0xff
        pt[7] = Y[7] ^ (self.HIGHT_F0[Y[6]] + round_key[(rounds-1)*4+3])

        # Reverse middle rounds
        for i in range(rounds - 2, -1, -1):
            pt = self.round_dec(round_key, pt, i)
        
        # Reverse initial whitening
        pt[0] = (pt[0].astype(np.uint16) - whitening_key[0].astype(np.uint16)) & 0xff
        pt[2] = pt[2] ^ whitening_key[1]
        pt[4] = (pt[4].astype(np.uint16) - whitening_key[2].astype(np.uint16)) & 0xff
        pt[6] = pt[6] ^ whitening_key[3]
        return np.flip(pt, axis=0)

    def check_testvector(self):
        k = np.uint8([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]).reshape(16, 1)
        p = np.zeros(8, dtype=np.uint8).reshape(8, 1)
        c = self.encrypt(p.copy(), k, 32)
        expected = np.array([0x00, 0xf4, 0x18, 0xae, 0xd9, 0x4f, 0x03, 0xf2], dtype=np.uint8)
        if np.all(c.flatten() == expected):
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
