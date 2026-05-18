import numpy as np
from scipy.optimize import linprog

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def plot_stack_result(
    polygons,
    displacements,
    output_path="triangle_stack_lp_output.png",
):
    """
    Plot original polygons and LP-displaced polygons.

    Dashed lines:
        original polygons

    Solid lines:
        polygons after LP displacement
    """

    moved_polygons = [
        P + d for P, d in zip(polygons, displacements)
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, P in enumerate(polygons, start=1):
        patch = MplPolygon(
            P,
            closed=True,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            label="Original" if i == 1 else None,
        )
        ax.add_patch(patch)

    for i, P in enumerate(moved_polygons, start=1):
        patch = MplPolygon(
            P,
            closed=True,
            fill=False,
            linewidth=2.0,
            label="After LP displacement" if i == 1 else None,
        )
        ax.add_patch(patch)

        center = P.mean(axis=0)
        ax.text(
            center[0],
            center[1],
            f"B{i}",
            ha="center",
            va="center",
        )

    for P, d in zip(polygons, displacements):
        c0 = P.mean(axis=0)
        c1 = c0 + d
        ax.annotate(
            "",
            xy=c1,
            xytext=c0,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.2,
            ),
        )

    all_points = np.vstack(polygons + moved_polygons)
    pad = 1.0
    ax.set_xlim(all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
    ax.set_ylim(all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("2D Polygon Stacking LP")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {output_path}")


def make_equilateral_triangle(width):
    """
    Create an upright equilateral triangle centered at the origin.
    Vertices are in counter-clockwise order.
    """
    h = width * np.sqrt(3) / 2

    return np.array([
        [-width / 2, -h / 3],
        [ width / 2, -h / 3],
        [0.0,        2 * h / 3],
    ], dtype=float)


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
                row[2 * i : 2 * i + 2] += a
                row[2 * (i - 1) : 2 * (i - 1) + 2] -= a
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

    displacements = result.x[:num_d_vars].reshape(N, 2)
    return displacements


if __name__ == "__main__":
    widths = [10, 8, 6, 4, 2]
    polygons = [make_equilateral_triangle(w) for w in widths]
    top_displacement = np.array([3.0, 0.0])
    displacements = solve_stack_lp(polygons, top_displacement)

    print("Solved displacements:")
    for i, d in enumerate(displacements, start=1):
        print(f"Block {i}: dx = {d[0]:.4f}, dy = {d[1]:.4f}")

    plot_stack_result(
        polygons,
        displacements,
        output_path="triangle_stack_lp_output.png",
    )
