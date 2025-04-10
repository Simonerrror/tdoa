import numpy as np
from test_data import generate_data

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def loss_fn(A, B, C, points, tdoa):
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

def grad_descent(points, tdoa, lr=0.05, epochs=10000, tol=1e-8, decay_every=1000, decay_rate=0.5):
    center = np.mean([points['D'], points['E'], points['F']], axis=0)
    A = center + np.random.randn(2) * 0.5
    B = center + np.random.randn(2) * 0.5
    C = center + np.random.randn(2) * 0.5

    prev_loss = None

    for i in range(epochs):
        grad_A = np.zeros(2)
        grad_B = np.zeros(2)
        grad_C = np.zeros(2)
        loss = 0.0

        for label in "DEF":
            D = points[label]
            diffs = tdoa[label]

            dA = A - D
            dB = B - D
            dC = C - D
            distA = np.linalg.norm(dA)
            distB = np.linalg.norm(dB)
            distC = np.linalg.norm(dC)
            eps = 1e-8

            # AB
            diff = (distA - distB) - diffs['AB']
            grad_A += (diff / (distA + eps)) * dA
            grad_B -= (diff / (distB + eps)) * dB
            loss += diff ** 2

            # AC
            diff = (distA - distC) - diffs['AC']
            grad_A += (diff / (distA + eps)) * dA
            grad_C -= (diff / (distC + eps)) * dC
            loss += diff ** 2

            # BC
            diff = (distB - distC) - diffs['BC']
            grad_B += (diff / (distB + eps)) * dB
            grad_C -= (diff / (distC + eps)) * dC
            loss += diff ** 2

        A -= lr * grad_A
        B -= lr * grad_B
        C -= lr * grad_C

        # Печать каждые 100 итераций
        if i % 100 == 0:
            print(f"iter {i}: loss = {loss:.10f}")

        # Early stopping
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            print(f"Early stopping at iter {i}, Δloss = {abs(prev_loss - loss):.2e}")
            break

        # Decay learning rate
        if i % decay_every == 0 and i > 0:
            lr *= decay_rate
            print(f"[{i}] learning rate decayed to {lr:.5f}")

        prev_loss = loss

    return A, B, C


if __name__ == "__main__":
    points, tdoa = generate_data()
    A_hat, B_hat, C_hat = grad_descent(points, tdoa)

    print("\nОценённые координаты:")
    print("A =", A_hat)
    print("B =", B_hat)
    print("C =", C_hat)

    print("\nНастоящие координаты:")
    print("A =", points['A'])
    print("B =", points['B'])
    print("C =", points['C'])
