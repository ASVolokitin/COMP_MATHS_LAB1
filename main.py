import numpy as np
from decimal import Decimal

def validate_choice(choice):
    if choice != 'K' and choice != 'F':
        while choice != 'K' and choice != 'F': choice = input("Неверный выбор. Введите букву К или F: ")
    return choice

def validate_n(n):
    try:
        if len(n.split()) > 1:
            print("Размер матрицы должен быть единственным числом.")
            return False
        n = int(n)
        if (n <= 0):
            print("Размер матрицы должен быть положительным числом.")
            return False
        if (n > 20):
            print("Такой размер матрицы не поддерживается.")
            return False
    except ValueError:
        print("Данные некорректны. Размер матрицы должен быть целым числом.")
        return False
    except TypeError: return False
    return True

def validate_matrix_row(row):
    try:
        row = list(map(float, row))
        return row
    except ValueError:
        print(f"Введённые данные некорректны.")
        return False

def read_n_f():
    filename = input("Введите имя файла: ")
    try:
        with open(filename, 'r') as file:
            n = file.readline().replace(',', '.')
            return n
    except FileNotFoundError:
        print("Файл не найден.")
        return None
    except PermissionError:
        print("Нет доступа к файлу.")

def read_matrix():
    choice = input("Вы хотите задать размер матрицы с клавиатуры (K) или из файла (F)? ")
    choice = validate_choice(choice)
    if choice == 'K':
        n = input("Введите размер матрицы: ").replace(',', '.')
        while True:
            if validate_n(n): break
            n = input("Введите размер матрицы: ").replace(',', '.')
    else: n = read_n_f()
    if n is None or not validate_n(n): return None
    n = int(n)
    print(f"Установлен размер матрицы: {n}")

    choice = input("Вы хотите ввести матрицу с клавиатуры (K) или из файла (F)? ").strip().upper()
    choice = validate_choice(choice)

    if choice == 'K':
        matrix = []
        print("Введите элементы матрицы построчно:")
        for i in range(n):
            while True:
                row = input(f"Введите {n + 1} элементов {i + 1}-й строки через пробел: ").replace(',', '.').split()
                if not validate_matrix_row(row, n): print(f"Введите {n + 1} чисел через пробел.")
                if len(row) == n + 1: break
                else: print(f"Вы ввели {len(row)} чисел, нужно {n + 1}.")
            matrix.append(list(map(Decimal, row)))

        return matrix

    elif choice == 'F':
        filename = input("Введите имя файла: ")
        try:
            with open(filename, 'r') as file:
                line_counter = 0
                matrix = []
                for line in file:
                    line_counter += 1
                    row = line.strip().replace(',', '.').split()
                    if not validate_matrix_row(row, n):
                        print(f"Строка {line_counter} должна состоять только из чисел, введённых через пробел.")
                        return None
                    if len(row) != n + 1 :
                        print(f"Исходные данные некорректы (строка {line_counter} содержит {len(row)} значений вместо {n + 1}).")
                        return None
                    matrix.append(list(map(Decimal, row)))
            return matrix
        except FileNotFoundError:
            print("Файл не найден.")
            return None

def to_upper_triangular(matrix):
    cp_matrix = [row[:] for row in matrix]
    perm_counter = 0
    n = len(cp_matrix)
    for col in range(n):
        max_row_id = col
        max_row_value = cp_matrix[col][0]
        for row in range(col, n):
            if abs(cp_matrix[row][col]) > abs(max_row_value):
                max_row_id = row
                max_row_value = cp_matrix[row][col]
        swap_rows(cp_matrix, col, max_row_id)
        if max_row_id != col: perm_counter += 1
        for row in range(col + 1, n):
            factor = cp_matrix[row][col] / cp_matrix[col][col]
            for j in range(col, n + 1):
                cp_matrix[row][j] -= factor * cp_matrix[col][j]

    return [cp_matrix, perm_counter]

def determinant(det_matrix):
    n = len(det_matrix)  # Размер квадратной части
    upper_triangular, perm_cnt = to_upper_triangular([row[:] for row in det_matrix])  # Копируем и приводим к треугольному виду

    mat_det = Decimal(1)
    for i in range(n):
        mat_det *= upper_triangular[i][i]  # Перемножаем диагональные элементы

    return mat_det * (-1)**perm_cnt

def numpy_determinant(matrix):
    matrix = np.array([[float(x) for x in row] for row in matrix])
    n = matrix.shape[0]
    square_matrix = matrix[:, :n]
    return float(np.linalg.det(square_matrix))

def print_matrix(pr_matrix):
    formatted_matrix = [[str(Decimal.normalize(x)) for x in row] for row in pr_matrix]
    col_widths = [max(len(formatted_matrix[row][col]) for row in range(len(pr_matrix))) for col in range(len(pr_matrix[0]))]
    for row in formatted_matrix: print("  ".join(f"{num:>{col_widths[i]}}" for i, num in enumerate(row)))


def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]

def gauss_solve(orig_matrix):
    matrix = [row[:] for row in orig_matrix]
    matrix = to_upper_triangular(matrix)[0]
    n = len(matrix)
    solution = [0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = matrix[row][-1]
        for j in range(row + 1, n):
            solution[row] -= matrix[row][j] * solution[j]
        solution[row] /= matrix[row][row]
    return solution

def numpy_solve(matrix):
    try:
        A = np.array([[float(x) for x in row[:-1]] for row in matrix])
        b = np.array([float(row[-1]) for row in matrix])
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("У СЛАУ нет единственного решения")
        return None


def calc_residuals(matrix, x):
    A = [row[:-1] for row in matrix]
    b = [row[-1] for row in matrix]
    n = len(A)
    residuals = [Decimal(0)] * n

    x = [Decimal(val) for val in x]

    for i in range(n):
        Ax_i = sum(Decimal(A[i][j]) * x[j] for j in range(n))
        residuals[i] = Ax_i - Decimal(b[i])

    return residuals

matrix = read_matrix()

if matrix is not None:
    print("Введённая матрица:")
    print_matrix(matrix)

    det = determinant(matrix)
    print("\nОпределитель, вычисленный вручную:", determinant(matrix))
    print("Определитель, вычисленный c использованием сторонней библиотеки", numpy_determinant([row[:len(matrix)] for row in matrix]))
    print("\nМатрица, приведённая к верхнетреугольному виду:")
    print_matrix(to_upper_triangular(matrix)[0])

    if (abs(det) < 1e-9):
        print("Определитель матрицы равен нулю, а значит у системы нет единственного решения")
    else:
        solution = gauss_solve(matrix)
        np_solution = numpy_solve(matrix)

        manual_residuals = calc_residuals(matrix, solution)
        print("\nРучное решение:", [float(x) for x in solution])
        print("Невязка ручного решения:", [float(x) for x in manual_residuals])
        print("Суммарно:", sum(abs(float(x)) for x in manual_residuals))

        np_residuals = list(map(float, calc_residuals(matrix, np_solution)))
        print("\nРешение с помощью библиотеки:", [float(x) for x in np_solution])
        print("Невязка решения с помощью библиотеки: ", np_residuals)
        print("Суммарно:", sum(abs(x) for x in np_residuals))
