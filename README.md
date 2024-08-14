# MovieLens Recommendation System

## Overview

This project demonstrates a simple collaborative filtering model using the MovieLens 100K dataset. The system is built using TensorFlow 1.x, and it trains a collaborative filtering model to recommend movies to users based on their past ratings.

Acknowledgement: this code is part of Google's course on [Recommendation Systems](https://developers.google.com/machine-learning/recommendation/)

## Setup

### Prerequisites

- Python 3.x
- TensorFlow 1.x (`tensorflow.compat.v1`)
- Pandas
- Numpy
- Matplotlib
- Altair (for visualization)
- Scikit-learn (for manifold learning)

### Installation

To set up the environment, you need to install the necessary Python libraries. This code assumes you have access to the Python environment with the required libraries installed.

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow==1.x altair
```

## Data Preparation

### Downloading the Data

The script automatically downloads the MovieLens 100K dataset and extracts the data files.

```python
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
zip_ref = zipfile.ZipFile('movielens.zip', "r")
zip_ref.extractall()
```

### Loading the Data

The dataset contains three files:

- `u.user`: Information about the users (e.g., age, gender, occupation).
- `u.item`: Information about the movies (e.g., title, genre).
- `u.data`: User ratings for movies.

These files are loaded into Pandas DataFrames for further processing.

## Collaborative Filtering Model

The model uses matrix factorization to learn embeddings for users and movies. The embeddings are trained to predict user ratings for movies.

### Key Components

- **Embedding Matrix**: The model learns embedding matrices for users and movies.
- **Loss Function**: Mean Squared Error (MSE) between predicted and actual ratings is used as the loss function.
- **Training**: Gradient Descent is used for optimizing the loss function.

## Training

The model is trained using the MovieLens dataset. The training process involves:

- Splitting the dataset into training and test sets.
- Training the model using the training set and evaluating it on the test set.
- Plotting the training and test errors over iterations.

```python
model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
model.train(num_iterations=1000, learning_rate=10.)
```

## Evaluation

The model's performance is evaluated using Mean Squared Error on the test set. The training and test errors are plotted to visualize the model's learning process.

## Usage

### Movie Recommendations

You can get movie recommendations for a specific user by calling the `user_recommendations` function.

```python
user_recommendations(model, measure=DOT, exclude_rated=True, k=6)
```

### Movie Neighbors

To find movies similar to a given movie, use the `movie_neighbors` function.

```python
movie_neighbors(model, "Aladdin", DOT)
movie_neighbors(model, "Aladdin", COSINE)
```

## Utilities

### DataFrame Enhancements

The code includes additional methods to enhance Pandas DataFrame functionality:

- **`mask`**: Filter DataFrame rows based on a condition.
- **`flatten_cols`**: Flatten multi-level column headers in a DataFrame.

### Convenience Functions

- **`split_dataframe`**: Splits a DataFrame into training and test sets.
- **`build_rating_sparse_tensor`**: Builds a TensorFlow SparseTensor from the ratings DataFrame.

## License

This project is open-source and available under the [MIT License](LICENSE).
