from __future__ import annotations

from core.algorithm_enum import StegoAlgorithm
from core.stego_context import DecodeContext, EncodeContext
from core.stego_dispatcher import StegoDispatcher
from utils import PRG


def run_roundtrip(algorithm: StegoAlgorithm):
    dispatcher = StegoDispatcher(verbose=True)

    prob_table = [0.42, 0.26, 0.18, 0.09, 0.05]
    indices = [10, 20, 30, 40, 50]
    bit_stream = "010111001101011010010111"
    precision = 16
    steps = 4

    enc_prg = PRG.from_int_seed(seed=11)
    dec_prg = PRG.from_int_seed(seed=11)
    bit_index = 0
    tokens = []
    interval = None

    print(f"\n=== Encode: {algorithm.value} ===")
    for _ in range(steps):
        er = dispatcher.dispatch_encode(
            EncodeContext(
                algorithm=algorithm,
                prob_table=prob_table,
                indices=indices,
                bit_stream=bit_stream,
                bit_index=bit_index,
                precision=precision,
                prg=enc_prg,
                cur_interval=interval,
            )
        )
        tokens.append(er.sampled_token_id)
        bit_index = er.metadata.get("next_bit_index", bit_index + er.bits_consumed)
        interval = er.metadata.get("cur_interval")
        print(f"token={er.sampled_token_id}, bits_used={er.bits_consumed}, next_bit_index={bit_index}")

    print(f"\n=== Decode: {algorithm.value} ===")
    recovered = ""
    interval = None
    for token_id in tokens:
        dr = dispatcher.dispatch_decode(
            DecodeContext(
                algorithm=algorithm,
                prob_table=prob_table,
                indices=indices,
                prev_token_id=token_id,
                precision=precision,
                prg=dec_prg,
                cur_interval=interval,
            )
        )
        recovered += dr.bits
        interval = dr.metadata.get("cur_interval")
        print(f"token={token_id}, recovered_bits='{dr.bits}'")

    print("\n=== Result ===")
    print("consumed_bits:", bit_index)
    print("origin_prefix:", bit_stream[:bit_index])
    print("recover_prefix:", recovered[:bit_index])
    print("match:", recovered[:bit_index] == bit_stream[:bit_index])


if __name__ == "__main__":
    run_roundtrip(StegoAlgorithm.AC)
