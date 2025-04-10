import numpy as np
from scipy.optimize import minimize
from test_data import generate_data

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def loss_fn(flat_xyz, points, tdoa):
    # Разворачиваем из плоского массива
    A = flat_xyz[0:2]
    B = flat_xyz[2:4]
    C = flat_xyz[4:6]

    loss = 0.0
    for label in "DEF":
        D = points[label]
        diffs = tdoa[label]

        pred_ab = distance(A, D) - distance(B, D)
        pred_ac = distance(A, D) - distance(C, D)
        pred_bc = distance(B, D) - distance(C, D)

        loss += (pred_ab - diffs['AB']) ** 2
        loss += (pred_ac - diffs['AC']) ** 2
        loss += (pred_bc - diffs['BC']) ** 2

    return loss

def solve_with_minimize(points, tdoa):
    # Стартовая инициализация (6 переменных: Ax, Ay, Bx, By, Cx, Cy)
    x0 = np.random.randn(6)

    res = minimize(loss_fn, x0, args=(points, tdoa), method='L-BFGS-B')

    A = res.x[0:2]
    B = res.x[2:4]
    C = res.x[4:6]

    return A, B, C, res

if __name__ == "__main__":
    points, tdoa = generate_data()

    A, B, C, res = solve_with_minimize(points, tdoa)

    print("\n[SCIPY] Оценённые координаты:")
    print("A =", A)
    print("B =", B)
    print("C =", C)
    print(f"Loss: {res.fun:.6f}, Success: {res.success}")
