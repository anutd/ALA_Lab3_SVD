import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


def load_and_prepare_data(file_path='ratings.csv'):
    df = pd.read_csv(file_path)
    ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    print(f"Original data shape: {ratings_matrix.shape}")
    print(f"Number of ratings: {df.shape[0]}")
    return ratings_matrix


def filter_data(ratings_matrix, min_user_ratings=100, min_movie_ratings=100):
    ratings_filtered = ratings_matrix.dropna(thresh=min_user_ratings, axis=0)
    ratings_filtered = ratings_filtered.dropna(thresh=min_movie_ratings, axis=1)

    print(f"Data shape after filtering: {ratings_filtered.shape}")
    print(f"Users remaining: {ratings_filtered.shape[0]}")
    print(f"Movies remaining: {ratings_filtered.shape[1]}")

    return ratings_filtered


def prepare_for_svd(ratings_matrix, fill_value=2.5):
    ratings_matrix_filled = ratings_matrix.fillna(fill_value)
    R = ratings_matrix_filled.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    print(f"Final matrix shape: {R_demeaned.shape}")

    return R_demeaned, user_ratings_mean


def perform_svd(R_demeaned, k=3):
    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma = sigma[::-1]
    U = U[:, ::-1]
    Vt = Vt[::-1, :]

    print(f"U shape: {U.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Vt shape: {Vt.shape}")

    return U, sigma, Vt


def visualize_users(U, num_users=19):
    actual_num_users = min(num_users, U.shape[0])
    U_subset = U[:actual_num_users, :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors_users = plt.cm.tab20(np.linspace(0, 1, actual_num_users))
    ax.scatter(U_subset[:, 0], U_subset[:, 1], U_subset[:, 2], s=50, alpha=0.7, c=colors_users)

    for i in range(min(10, actual_num_users)):
        ax.text(U_subset[i, 0], U_subset[i, 1], U_subset[i, 2],
                f'User {i + 1}', fontsize=8)

    ax.set_title(f'User Preferences (First {actual_num_users} Users)\nCloser points = More similar tastes')

    plt.tight_layout()
    plt.show()


def visualize_movies(Vt, num_movies=20):
    V = Vt.T  # transposing to get movies as rows
    actual_num_movies = min(num_movies, V.shape[0])  # to not ask for more movies than we have
    V_subset = V[:actual_num_movies, :]

    print(f"\nVisualizing {actual_num_movies} movies (out of {V.shape[0]} available)")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors_movies = plt.cm.tab20(np.linspace(0, 1, actual_num_movies))
    ax.scatter(V_subset[:, 0], V_subset[:, 1], V_subset[:, 2], s=50, alpha=0.7, c=colors_movies)

    for i in range(min(10, actual_num_movies)):
        ax.text(V_subset[i, 0], V_subset[i, 1], V_subset[i, 2],
                f'Movie {i + 1}', fontsize=8)

    ax.set_title(f'Movie Characteristics (First {actual_num_movies} Movies)\nCloser points = More similar movies')

    plt.tight_layout()
    plt.show()


def analyze_similarities(U, Vt):
    print("\nStep 7: Analyzing similarities...")
    print("\nDistances between first 5 users (smaller distance = more similar tastes):")
    for i in range(min(5, U.shape[0])):
        for j in range(i + 1, min(5, U.shape[0])):
            distance = np.linalg.norm(U[i] - U[j])
            print(f"User {i + 1} - User {j + 1}: {distance:.4f}")

    print("\nDistances between first 5 movies (smaller distance = more similar movies):")
    V = Vt.T
    for i in range(min(5, V.shape[0])):
        for j in range(i + 1, min(5, V.shape[0])):
            distance = np.linalg.norm(V[i] - V[j])
            print(f"Movie {i + 1} - Movie {j + 1}: {distance:.4f}")


ratings_matrix = load_and_prepare_data('ratings.csv')

ratings_filtered = filter_data(ratings_matrix, min_user_ratings=100, min_movie_ratings=100)

R_demeaned, user_ratings_mean = prepare_for_svd(ratings_filtered, fill_value=2.5)

U, sigma, Vt = perform_svd(R_demeaned, k=3)

visualize_users(U, num_users=19)

visualize_movies(Vt, num_movies=20)

analyze_similarities(U, Vt)


