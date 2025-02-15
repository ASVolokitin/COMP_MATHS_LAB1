import numpy as np

def generate_and_save_augmented_matrix():
    n = int(input("Введите размер матрицы n: "))
    filename = input("Введите имя файла для сохранения: ")
    matrix = np.random.uniform(-100, 100, (n, n + 1))
    matrix = np.round(matrix, 5)
    np.savetxt(filename, matrix, fmt="%.5f", delimiter=" ")
    print(f"Расширенная матрица сохранена в {filename}")

generate_and_save_augmented_matrix()