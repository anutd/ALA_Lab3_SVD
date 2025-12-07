import numpy as np


def svd_implementation(A):
    ATA = A.T @ A
    # eigendecomposition of A^T A to get V (right singular vectors)
    eigenvalues, V = np.linalg.eig(ATA)

    # sorting eigenvalues and eigenvectors in descending order. we want the most important patterns first
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    # removing very small singular values (optional but helps stability)
    mask = singular_values > 1e-10
    singular_values = singular_values[mask]
    V = V[:, mask]

    r = len(singular_values)  # rank of matrix
    # computing U (left singular vectors) using σ u = A v  →  u = A v / σ
    U = np.zeros((A.shape[0], r))
    for i in range(r):
        u = A @ V[:, i] / singular_values[i]
        U[:, i] = u / np.linalg.norm(u)  # normalize to unit length

    # (r x r diagonal)
    Σ = np.diag(singular_values)

    VT = V.T

    print("Checking condition σᵢ * uᵢ = A * vᵢ for all singular vectors:")
    for i in range(r):
        left = singular_values[i] * U[:, i]
        right = A @ V[:, i]
        print(f"Vector {i+1}: ", np.allclose(left, right))

    return U, Σ, VT


A = np.array([[3, 1],
              [1, 3],
              [0, 0]], dtype=float)

U, Σ, VT = svd_implementation(A)

A_reconstructed = U @ Σ @ VT

print("\nOriginal A:")
print(A)

print("\nReconstructed A (U @ Σ @ VT):")
print(np.round(A_reconstructed, 6))

print("\nAre they almost equal?", np.allclose(A, A_reconstructed))

print(f"\nU shape: {U.shape}, Σ shape: {Σ.shape}, VT shape: {VT.shape}")
print(f"\nU:\n{np.round(U, 6)}")
print(f"\nΣ:\n{np.round(Σ, 6)}")
print(f"\nVT:\n{np.round(VT, 6)}")
