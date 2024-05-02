import numpy as np
import unittest


def forward_substitution(L, b):
    """
    Menggunakan substitusi maju untuk menyelesaikan sistem persamaan linier dengan matriks segitiga bawah.

    Args:
        L (numpy.ndarray): Matriks segitiga bawah.
        b (numpy.ndarray): Vektor kolom pada sisi kanan sistem persamaan linier.

    Returns:
        numpy.ndarray: Solusi dari sistem persamaan linier.
    """
    # Membuat vektor y dengan ukuran yang sama dengan vektor b
    y = np.full_like(b, 0)

    for k in range(len(b)):
        y[k] = b[k]

        # Melakukan substitusi maju
        for i in range(k):
            y[k] = y[k] - (L[k, i]*y[i])

        y[k] = y[k] / L[k, k]

    return y


def backward_substitution(U, y):
    """
    Menggunakan substitusi mundur untuk menyelesaikan sistem persamaan linier dengan matriks segitiga atas.

    Args:
        U (numpy.ndarray): Matriks segitiga atas.
        y (numpy.ndarray): Vektor kolom hasil substitusi maju.

    Returns:
        numpy.ndarray: Solusi dari sistem persamaan linier.
    """
    # Membuat vektor x dengan ukuran yang sama dengan vektor y
    x = np.full_like(y, 0)

    for k in range(len(x), 0, -1):
        # Melakukan substitusi mundur
        x[k-1] = (y[k-1] - np.dot(U[k-1, k:], x[k:])) / U[k-1, k-1]

    return x


def crout(A):
    """
    Menggunakan metode Crout untuk melakukan dekomposisi LU pada matriks A.

    Args:
        A (numpy.ndarray): Matriks koefisien.

    Returns:
        tuple: Matriks segitiga bawah (L) dan matriks segitiga atas (U).
    """
    n = len(A)

    # Inisialisasi matriks L dan U dengan nol
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for z in range(n):
        U[z, z] = 1

        for j in range(z, n):
            temporary_L = float(A[j, z])
            for k in range(z):
                temporary_L -= L[j, k]*U[k, z]
            L[j, z] = temporary_L

        for j in range(z+1, n):
            temporary_U = float(A[z, j])
            for k in range(z):
                temporary_U -= L[z, k]*U[k, j]
            U[z, j] = temporary_U / L[z, z]

    return (L, U)


def computing_final_solution(A, b, algorithm_used):
    """
    Menghitung solusi dari sistem persamaan linier Ax = b menggunakan algoritma dekomposisi yang diberikan.

    Args:
        A (numpy.ndarray): Matriks koefisien.
        b (numpy.ndarray): Vektor kolom pada sisi kanan sistem persamaan linier.
        algorithm_used (function): Algoritma dekomposisi yang digunakan (misalnya, crout).

    Returns:
        numpy.ndarray: Solusi dari sistem persamaan linier.
    """
    # Mendapatkan matriks L dan U menggunakan algoritma yang ditentukan
    L, U = algorithm_used(A)

    # Mencetak matriks L dan U (bisa diubah menjadi return jika diperlukan)
    print("L = " + str(L) + "\n")
    print("U = " + str(U) + "\n")

    # Melakukan substitusi maju kemudian substitusi mundur
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


# Matriks koefisien A dan vektor kolom b
A = np.matrix([[1, 1, -1],
              [-1, 1, 1],
              [2, 2, 1]])
b = np.array([1.0, 5.0, 1.0])

# Menggunakan metode Crout untuk mendapatkan solusi sistem persamaan linier
print("\n" + "Solusi menggunakan algoritma Crout:" + "\n")
print("x = " + str(computing_final_solution(A, b, crout)) + "\n")

# acuannya pada source https://clnazalia.wordpress.com/2018/10/27/sistem-persamaan-linier-metode-dekomposisi/

# unit test


class TestLinearSystemSolver(unittest.TestCase):

    def test_computing_final_solution_with_crout(self):
        # Define the matrix A and vector b
        A = np.matrix([[1, 1, -1],
                      [-1, 1, 1],
                      [2, 2, 1]])
        b = np.array([1.0, 5.0, 1.0])

        # Expected solution
        expected_solution = np.array([-2.33333333, 3., -0.33333333])

        # Compute the solution using Crout method
        computed_solution = computing_final_solution(A, b, crout)

        # Check if the computed solution matches the expected solution
        np.testing.assert_allclose(computed_solution, expected_solution)


if __name__ == '__main__':
    unittest.main()
