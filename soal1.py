import numpy as np

# Fungsi untuk mencari matriks balikan menggunakan NumPy


def inverse_matrix(matrix):
    try:
        inverse = np.linalg.inv(matrix)
        return inverse
    except np.linalg.LinAlgError:
        return None


# Contoh penggunaan
A = np.array([[1, 2], [3, 4]])
inverse_A = inverse_matrix(A)
if inverse_A is not None:
    print("Matriks Balikan A:")
    print(inverse_A)
else:
    print("Matriks A tidak memiliki balikan.")