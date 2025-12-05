import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from collections import Counter

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')  # (users x movies)
print(f"Original shape: {ratings_matrix.shape[0]} users x {ratings_matrix.shape[1]} movies")

min_user_ratings = 50  # users must have rated at least 50 movies
min_movie_ratings = 20  # movies must have at least 20 ratings
ratings_filtered = ratings_matrix.dropna(thresh=min_user_ratings, axis=0)
ratings_filtered = ratings_filtered.dropna(thresh=min_movie_ratings, axis=1)
print(f"After filtering: {ratings_filtered.shape[0]} users x {ratings_filtered.shape[1]} movies")

ratings_filled = ratings_filtered.fillna(2.5)
R = ratings_filled.values
user_avg = np.mean(R, axis=1)  # average rating for each user
R_normalized = R - user_avg.reshape(-1, 1)  # normalizing because some users always give high ratings and some always give low

k_values = [2, 5, 10, 20, 50]  # experiment with different k values
errors = []

for k in k_values:
    print(f"\nTesting with k={k}...")

    U, sigma, Vt = svds(R_normalized, k=k)
    # U (Users × k): How much each user likes each hidden pattern
    # Σ (k values): Importance of each pattern
    # Vᵀ (k × Movies): How much each movie has of each pattern
    sigma = sigma[::-1]
    U = U[:, ::-1]
    Vt = Vt[::-1, :]

    predicted_ratings = U @ np.diag(sigma) @ Vt + user_avg.reshape(-1, 1)

    # calculating error on actual ratings
    # and getting only the cells that had actual ratings
    actual_mask = ~ratings_filtered.isna()
    actual_values = ratings_filtered.values[actual_mask]
    predicted_values = predicted_ratings[actual_mask]

    mae = np.mean(np.abs(actual_values - predicted_values))
    errors.append(mae)

    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"Example: Actual={actual_values[0]:.2f}, Predicted={predicted_values[0]:.2f}")

# Plot errors vs k values
plt.figure(figsize=(8, 5))
plt.plot(k_values, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('k value')
plt.ylabel('MAE (Mean Absolute Error)')
plt.title('How k affects prediction accuracy')
plt.grid(True)
plt.show()

best_k = k_values[np.argmin(errors)]
print(f"\nBest k value - {best_k} (with lowest error)")

print(f"\nFinal predictions with best k value ({best_k}):")
U, sigma, Vt = svds(R_normalized, k=best_k)
sigma = sigma[::-1]
U = U[:, ::-1]
Vt = Vt[::-1, :]

all_predictions = U @ np.diag(sigma) @ Vt + user_avg.reshape(-1, 1)

predictions_df = pd.DataFrame(
    all_predictions,
    columns=ratings_filtered.columns,
    index=ratings_filtered.index
)
print(f"Predictions DataFrame shape: {predictions_df.shape}")
# table with only predicted ratings (remove existing ones) - movies users haven't rated yet
predicted_only = predictions_df.copy()
# removing ratings that already existed (replace with NaN)
existing_mask = ~ratings_filtered.isna()
predicted_only[existing_mask] = np.nan
print(f"Predicted-only: {predicted_only}")
print(f"Predicted-only ratings shape: {predicted_only.shape}")
print(f"Number of new predictions: {predicted_only.notna().sum().sum()}")


def recommend_movies(user_id, num_recommendations=10):
    if user_id not in predictions_df.index:
        print(f"User {user_id} not found!")
        return

    print(f"Recommendations for User {user_id}")

    user_predictions = predictions_df.loc[user_id]

    # movies user has already rated
    if user_id in ratings_filtered.index:
        rated_movies = ratings_filtered.loc[user_id].dropna().index
    else:
        rated_movies = []

    # removing movies user already rated
    new_predictions = user_predictions.drop(rated_movies, errors='ignore')
    top_movies = new_predictions.nlargest(num_recommendations)
    # getting movie details
    recommendations = movies[movies['movieId'].isin(top_movies.index)].copy()
    # adding predicted ratings
    recommendations['predicted_rating'] = recommendations['movieId'].map(top_movies)
    # sorting by predicted rating
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)

    for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
        print(f"{i:2d}. {movie['title']:45} | Rating: {movie['predicted_rating']:.2f} | Genres: {movie['genres']}")

    print(f"\nGenre analysis for User {user_id}:")
    all_genres = []
    for genre_str in recommendations['genres']:
        all_genres.extend(genre_str.split('|'))

    genre_counts = Counter(all_genres)
    for genre, count in genre_counts.most_common(5):
        print(f"  {genre}: {count} movies")

    return recommendations


# recommendations for first 3 users
user_ids = list(predictions_df.index)[:3]

for user_id in user_ids:
    recommend_movies(user_id, num_recommendations=10)
    print()
