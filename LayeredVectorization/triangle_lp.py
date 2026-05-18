import numpy as np
from scipy.optimize import linprog


def polygon_to_halfspace_ccw(P):
    """
    Convert a convex polygon into half-space form:

        A x <= b

    Assumption:
        - P is convex
        - vertices are ordered counter-clockwise
    """
    A = []
    b = []

    n = len(P)
    for i in range(n):
        p = P[i]
        q = P[(i + 1) % n]
        edge = q - p
        a = np.array([edge[1], -edge[0]])
        A.append(a)
        b.append(a @ p)

    return np.array(A), np.array(b)


def solve_stack_lp(polygons, top_displacement):
    """
    Solve the translation-only 2D stacking problem.

    Input:
        polygons:
            list of convex polygons [P1, P2, ..., PN]
            P1 is the bottom block
            PN is the top block

        top_displacement:
            np.array([dx, dy])
            prescribed displacement of the top block

    Output:
        displacements:
            array of shape (N, 2)
            displacement of each polygon
    """
    N = len(polygons)

    num_d_vars = 2 * N
    num_s_vars = 2 * (N - 1)
    num_vars = num_d_vars + num_s_vars

    c = np.zeros(num_vars)
    c[num_d_vars:] = 1.0

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    halfspaces = [polygon_to_halfspace_ccw(P) for P in polygons]

    for i in range(1, N):
        child = polygons[i]
        A_parent, b_parent = halfspaces[i - 1]
        for v in child:
            rhs = b_parent - A_parent @ v
            for a, r in zip(A_parent, rhs):
                row = np.zeros(num_vars)
                row[2 * i: 2 * i + 2] += a
                row[2 * (i - 1): 2 * (i - 1) + 2] -= a
                A_ub.append(row)
                b_ub.append(r)

    top_idx = N - 1
    row = np.zeros(num_vars)
    row[2 * top_idx] = 1.0
    A_eq.append(row)
    b_eq.append(top_displacement[0])

    row = np.zeros(num_vars)
    row[2 * top_idx + 1] = 1.0
    A_eq.append(row)
    b_eq.append(top_displacement[1])

    for i in range(N - 1):
        for coord in range(2):
            d_idx = 2 * i + coord
            s_idx = num_d_vars + 2 * i + coord

            row = np.zeros(num_vars)
            row[d_idx] = 1.0
            row[s_idx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(num_vars)
            row[d_idx] = -1.0
            row[s_idx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    bounds = [(None, None)] * num_d_vars + [(0, None)] * num_s_vars

    result = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x[:num_d_vars].reshape(N, 2)
