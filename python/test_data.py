import numpy as np

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def generate_data():
    # Истинные координаты приёмников (которые будем восстанавливать потом)
    A = np.array([0.0, 0.0])
    B = np.array([5.0, 0.0])
    C = np.array([2.5, 5.0])

    # Источники сигнала (заданы)
    D = np.array([3.0, 4.0])
    E = np.array([6.0, 1.0])
    F = np.array([1.0, 3.0])

    points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F}

    # Разности расстояний
    tdoa = {
        'D': {
            'AB': distance(A, D) - distance(B, D),
            'AC': distance(A, D) - distance(C, D),
            'BC': distance(B, D) - distance(C, D),
        },
        'E': {
            'AB': distance(A, E) - distance(B, E),
            'AC': distance(A, E) - distance(C, E),
            'BC': distance(B, E) - distance(C, E),
        },
        'F': {
            'AB': distance(A, F) - distance(B, F),
            'AC': distance(A, F) - distance(C, F),
            'BC': distance(B, F) - distance(C, F),
        },
    }

    return points, tdoa

if __name__ == "__main__":
    points, tdoa = generate_data()
    print("Координаты:")
    for k, v in points.items():
        print(f"{k}: {v}")
    print("\nРазности расстояний:")
    for p, vals in tdoa.items():
        print(f"{p}: {vals}")
