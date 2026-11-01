import numpy as np


def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.
    """
    ref = reference_sequence.upper()
    dist = distant_sequence.upper()

    # Ignore positions if seq has indel ('-') or 'N'
    valid_pairs = [(r, d) for r, d in zip(ref, dist) if r in "ACGT" and d in "ACGT"]
    L = len(valid_pairs)
    if L == 0:
        return 0.0

    mismatches = sum(1 for r, d in valid_pairs if r != d)
    if mismatches == 0:
        return 0.0

    p = mismatches / L
    if p >= 0.75:
        p = 0.749999  # log domain errorr fix 
        
    jc_distance = -0.75 * np.log(1.0 - (4.0 / 3.0) * p) * L
    return float(jc_distance)


def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.
    """
    ref = reference_sequence.upper()
    dist = distant_sequence.upper()

    valid_pairs = [(r, d) for r, d in zip(ref, dist) if r in "ACGT" and d in "ACGT"]
    L = len(valid_pairs)
    if L == 0:
        return 0.0

    transitions_set = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    ts = 0
    tv = 0

    for r, d in valid_pairs:
        if r != d:
            if (r, d) in transitions_set:
                ts += 1
            else:
                tv += 1

    if ts == 0 and tv == 0:
        return 0.0

    P = ts / L
    Q = tv / L

    k2p_distance = (-0.5 * np.log(1.0 - 2.0 * P - Q) - 0.25 * np.log(1.0 - 2.0 * Q)) * L
    return float(k2p_distance)