from __future__ import annotations

import hashlib
import hmac


class PRG:
    """HMAC-DRBG style pseudo-random generator for stego strategies."""

    def __init__(self, key: bytes, seed: bytes):
        if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
            raise ValueError("key must be non-empty bytes")
        if not isinstance(seed, (bytes, bytearray)):
            raise ValueError("seed must be bytes")

        self.key = bytes(key)
        self.val = b"\x01" * 64
        self._reseed(bytes(seed))
        self.byte_index = 0
        self.bit_index = 0

    @classmethod
    def from_hex(cls, input_key_hex: str, sample_seed_prefix_hex: str = "", input_nonce_hex: str = "") -> "PRG":
        key = bytes.fromhex(input_key_hex)
        seed = bytes.fromhex(sample_seed_prefix_hex) + bytes.fromhex(input_nonce_hex)
        return cls(key=key, seed=seed)

    @classmethod
    def from_int_seed(cls, seed: int) -> "PRG":
        if seed < 0:
            raise ValueError("seed must be >= 0")
        key = hashlib.sha256(f"prg-key:{seed}".encode("utf-8")).digest()
        seed_bytes = hashlib.sha256(f"prg-seed:{seed}".encode("utf-8")).digest()
        return cls(key=key, seed=seed_bytes)

    def _hmac(self, key: bytes, val: bytes) -> bytes:
        return hmac.new(key, val, hashlib.sha512).digest()

    def _reseed(self, data: bytes = b"") -> None:
        self.key = self._hmac(self.key, self.val + b"\x00" + data)
        self.val = self._hmac(self.key, self.val)
        if data:
            self.key = self._hmac(self.key, self.val + b"\x01" + data)
            self.val = self._hmac(self.key, self.val)

    def generate_bits(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError("n must be >= 0")
        bits: list[int] = []
        for _ in range(n):
            bit = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1
            bits.append(int(bit))

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            # Keep behavior consistent with previous implementation.
            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self._hmac(self.key, self.val)

        self._reseed()
        return bits

    def generate_random(self, n: int) -> float:
        bits = self.generate_bits(n)
        if len(bits) == 0:
            return 0.0

        decimal_value = 0
        for bit in bits:
            decimal_value = (decimal_value << 1) | int(bit)

        max_value = (1 << len(bits)) - 1
        return decimal_value / max_value if max_value > 0 else 0.0
