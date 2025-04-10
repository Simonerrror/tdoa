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

def grad_descent(points, tdoa, lr=0.01, epochs=1000):
    # Инициализация (случайно)
    A = np.random.randn(2)
    B = np.random.randn(2)
    C = np.random.randn(2)

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

            # avoid division by zero
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

        if i % 100 == 0:
            print(f"iter {i}: loss = {loss:.6f}")

    return A, B, C
