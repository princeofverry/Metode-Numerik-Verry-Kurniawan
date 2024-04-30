import numpy as np

# Fungsi untuk dekomposisi LU menggunakan algoritma Gauss


def lu_decomposition_gauss(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Mengisi bagian diagonal L dengan 1
        L[i][i] = 1

        # Menghitung elemen-elemen U
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        # Menghitung elemen-elemen L
        for k in range(i + 1, n):
            sum = 0
            for j in range(i):
                sum += (L[k][j] * U[j][i])
            L[k][i] = (matrix[k][i] - sum) / U[i][i]

    return L, U


# Contoh penggunaan
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
L, U = lu_decomposition_gauss(A)
print("Matriks L:")
print(L)
print("Matriks U:")
print(U)
