



def combine_pint_arrays(a, b, idx):

    if not a.check(b.units):
        raise ValueError('Incompatible units', a, b)

    b_star = b.to(a.units)
    A = a.magnitude
    B = b_star.magnitude
    A[idx:] = B[idx:]
    return A * a.units
