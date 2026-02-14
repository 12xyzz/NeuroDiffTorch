import numpy as np
from torch.utils.data import Dataset
from os import urandom


# from NIST FIPS-197
S_BOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

R_CON = np.array([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint8)

# Galois field GF(2^8) multiplication lookup
_GALOIS_LOOKUP = np.zeros((256, 256), dtype=np.uint8)
for a in range(256):
    for b in range(256):
        p, x, y = 0, a, b
        for _ in range(8):
            if y & 1:
                p ^= x
            hi = x & 0x80
            x = (x << 1) & 0xff
            if hi:
                x ^= 0x1B
            y >>= 1
        _GALOIS_LOOKUP[a, b] = p

MIX_MAT = np.array([[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]], dtype=np.uint8)


def _galois_matrix_mult(mat, state):
    n = state.shape[0]
    out = np.zeros_like(state, dtype=np.uint8)
    for i in range(4):
        for k in range(4):
            out[:, i, :] ^= _GALOIS_LOOKUP[mat[i, k], state[:, k, :]]
    return out


class Aes_Dataset(Dataset):
    def __init__(self, n, nr, key_mode='random', key=None, diff=None, neg='real_encryption', batch_size=10000):
        self.cipher = Aes()
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
            self.X, self.Y = self.generate_training_data(
                self.n, self.nr, self.diff, self.neg, self.batch_size)

    def __len__(self):
        if self.Y is None:
            raise RuntimeError("Dataset not generated. Call generate_dataset() first.")
        return len(self.Y)

    def __getitem__(self, idx):
        if self.X is None or self.Y is None:
            raise RuntimeError("Dataset not generated. Call generate_dataset() first.")
        return self.X[idx], self.Y[idx]

    def generate_plaintext_triples(self, n, diff):
        p1 = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(n, 16)
        diff_array = np.array(diff[:16], dtype=np.uint8)
        p1_prime = p1 ^ diff_array
        p2 = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(n, 16)
        return p1, p1_prime, p2

    def encrypt_plaintext_pairs(self, p1, p2, nr):
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if p1.shape[0] != p2.shape[0]:
                raise ValueError("plaintext array length must be the same")
            n = p1.shape[0]
            if n == 0:
                return np.zeros((16, 0), dtype=np.uint8), np.zeros((16, 0), dtype=np.uint8)
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

    def generate_training_data(self, n, nr, diff=None, neg='real_encryption', batch_size=10000):
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
            X = self.cipher.to_bits(
                [c1[i] for i in range(16)] + [c2[i] for i in range(16)])
        else:
            c1, c1_prime = self.encrypt_plaintext_pairs(p1, p1_prime, nr)
            Y = chosen.astype(np.uint8)
            feature_dim = 32 * 8  # 32 bytes * 8 bits
            X = np.zeros((n, feature_dim), dtype=np.uint8)
            pos_mask = Y == 1
            if np.any(pos_mask):
                pos_X = self.cipher.to_bits(
                    [c1[i][pos_mask] for i in range(16)] +
                    [c1_prime[i][pos_mask] for i in range(16)])
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
            key_array = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(16, -1)
        elif key_mode == 'random_fixed':
            single_key = np.frombuffer(urandom(16), dtype=np.uint8)
            key_array = single_key.reshape(16, 1).repeat(n, axis=1)
        elif key_mode == 'input_fixed':
            if key is None:
                raise ValueError("key_mode is 'input_fixed' but no key provided")
            single_key = key
            key_array = np.array(key, dtype=np.uint8)
            if key_array.size != 16:
                raise ValueError("Key must be 16 bytes for AES-128")
            key_array = key_array.reshape(16, 1).repeat(n, axis=1)
        else:
            raise ValueError(f"Invalid key_mode: {key_mode}. Must be 'random', 'random_fixed', or 'input_fixed'")
        return single_key, key_array


class Aes:
    BLOCK_BYTES = 16
    KEY_BYTES = 16

    def __init__(self, word_size=8):
        self.word_size = word_size

    def _pt_to_state(self, pt):
        n = pt.shape[1]
        state = np.zeros((n, 4, 4), dtype=np.uint8)
        for row in range(4):
            for col in range(4):
                state[:, row, col] = pt[row + col * 4, :]
        return state

    def _state_to_pt(self, state):
        n = state.shape[0]
        pt = np.zeros((16, n), dtype=np.uint8)
        for row in range(4):
            for col in range(4):
                pt[row + col * 4, :] = state[:, row, col]
        return pt

    def _key_expansion(self, key, nr):
        n = key.shape[0]
        key = key.reshape(n, 4, 4)
        w = np.zeros((n, 4 * (nr + 1), 4), dtype=np.uint8)
        w[:, :4, :] = key.reshape(n, 4, 4)

        for i in range(4, 4 * (nr + 1)):
            temp = w[:, i - 1, :].copy()
            if i % 4 == 0:
                temp = np.roll(temp, -1, axis=1)
                temp = S_BOX[temp]
                temp[:, 0] ^= R_CON[i // 4]
            w[:, i, :] = w[:, i - 4, :] ^ temp

        return w.reshape(n, nr + 1, 4, 4).transpose(0, 1, 3, 2)

    def _sub_bytes(self, state):
        return S_BOX[state]

    def _shift_rows(self, state):
        state = state.copy()
        state[:, 1, :] = state[:, 1, [1, 2, 3, 0]]
        state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]
        state[:, 3, :] = state[:, 3, [3, 0, 1, 2]]
        return state

    def _mix_columns(self, state):
        return _galois_matrix_mult(MIX_MAT, state)

    def encrypt(self, pt, key, nr, last_round_mc=False):
        n = pt.shape[1]
        state = self._pt_to_state(pt)
        key_n = key.T
        round_keys = self._key_expansion(key_n, nr)

        state = state ^ round_keys[:, 0, :, :]

        for r in range(1, nr):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = state ^ round_keys[:, r, :, :]

        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        if last_round_mc:
            state = self._mix_columns(state)
        state = state ^ round_keys[:, nr, :, :]

        return self._state_to_pt(state)

    def check_testvector(self):
        # NIST FIPS-197 Appendix C.1: AES-128
        key = np.array([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c], dtype=np.uint8).reshape(16, 1)
        pt = np.array([0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
                       0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34], dtype=np.uint8).reshape(16, 1)
        expected = np.array([0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
                            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32], dtype=np.uint8)
        c = self.encrypt(pt, key, 10, last_round_mc=False)
        if not np.all(c.flatten() == expected):
            print("Testvector not verified.")
            return False
        print("Testvector verified.")
        return True

    def to_bits(self, arr):
        block_num = len(arr)
        sample_num = len(arr[0])
        X = np.zeros((block_num * self.word_size, sample_num), dtype=np.uint8)
        for i in range(block_num * self.word_size):
            block_idx = i // self.word_size
            bit_offset = self.word_size - (i % self.word_size) - 1
            X[i] = (arr[block_idx] >> bit_offset) & 1
        return X.transpose()
