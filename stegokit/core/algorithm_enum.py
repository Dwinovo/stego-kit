from enum import StrEnum


class StegoAlgorithm(StrEnum):
    AC = "ac"
    ADG = "adg"
    DISCOP = "discop"
    DISCOP_BASE = "discop_base"
    FDPSS_DIFFERENTIAL_BASED = "fdpss_differential_based"
    FDPSS_BINARY_BASED = "fdpss_binary_based"
    FDPSS_STABILITY_BASED = "fdpss_stability_based"
    METEOR = "meteor"
    ARS = "ars"
    HUFFMAN = "huffman"
