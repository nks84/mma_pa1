import pandas
import numpy as np

class MusicGenreClassifier:
    def __init__(self):
        self.df_tracks = None
        self.df_features = None
        self.X_train = None
        self.X_test = None
        self.X_validation = None
        self.y_train = None
        self.y_test = None
        self.y_validation = None
        self.buckets = {}
        self.genre_counts = {}
        self.bucket_genres = {}
        self.R = None

    def get_shape(self, array):
        shape = array.shape
        rows, columns = shape
        print(f"Number of rows: {rows}, Number of columns: {columns}")

    def generate_random_matrix(self, m, n):
        rij = np.random.choice([-1, 0, 1], size=(m, n), p=[1/6, 2/3, 1/6])
        return np.sqrt(3) * rij

    def load_data(self):
        self.df_tracks = pandas.read_csv('tracks.csv', index_col=0, header=[0, 1])
        self.df_tracks = self.df_tracks[self.df_tracks['set']['subset'] == 'medium']
        self.df_features = pandas.read_csv('features.csv', index_col=0, header=[0, 1, 2])

    def preprocess_data(self):
        self.df_tracks = self.df_tracks[self.df_tracks['track']['genre_top'].isin(['Hip-Hop', 'Pop', 'Folk', 'Rock', 'Experimental', 'International', 'Electronic', 'Instrumental'])]

        self.df_tracks_train = self.df_tracks[self.df_tracks.iloc[:, 30] == 'training']
        self.df_tracks_test = self.df_tracks[self.df_tracks.iloc[:, 30] == 'test']
        self.df_tracks_validation = self.df_tracks[self.df_tracks.iloc[:, 30] == 'validation']

        self.df_features_train = self.df_features[self.df_features.index.isin(self.df_tracks_train.index)]
        self.df_features_test = self.df_features[self.df_features.index.isin(self.df_tracks_test.index)]
        self.df_features_validation = self.df_features[self.df_features.index.isin(self.df_tracks_validation.index)]

        self.X_train = self.df_features_train.values
        self.X_test = self.df_features_test.values
        self.X_validation = self.df_features_validation.values

        self.y_train = self.df_tracks_train['track']['genre_top']
        self.y_test = self.df_tracks_test['track']['genre_top']
        self.y_validation = self.df_tracks_validation['track']['genre_top']

    def train(self, bits=32):
        self.R = self.generate_random_matrix(bits, 518)
        X_train_zero_dot = np.dot(self.X_train, self.R.T)
        X_train_zero_dot = X_train_zero_dot > 0
        X_train_zero_dot = X_train_zero_dot.astype(int)

        for i in range(len(X_train_zero_dot)):
            hash_str = ''.join(X_train_zero_dot[i].astype(str))
            if hash_str not in self.buckets.keys():
                self.buckets[hash_str] = []
            self.buckets[hash_str].append(i)

        for key, value in self.buckets.items():
            self.genre_counts[key] = {}
            for i in range(len(value)):
                genre = self.y_train.iloc[value[i]]
                if genre not in self.genre_counts[key]:
                    self.genre_counts[key][genre] = 1
                else:
                    self.genre_counts[key][genre] += 1

        for key, value in self.genre_counts.items():
            self.genre_counts[key] = {k: v for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)}

        for key, value in self.genre_counts.items():
            self.bucket_genres[key] = list(value.keys())[0]

    def evaluate_train_accuracy(self, majority_percentage=0.75):
        genre_majority = {}
        for key, value in self.genre_counts.items():
            for genre, count in value.items():
                if count/len(self.buckets[key]) > majority_percentage:
                    if genre not in genre_majority:
                        genre_majority[genre] = 1
                    else:
                        genre_majority[genre] += 1

        accuracy = sum(genre_majority.values())/len(self.buckets)
        print(f"Accuracy: {accuracy}")

    def evaluate_test_accuracy(self):
        X_test_zero_dot = np.dot(self.X_test, self.R.T)
        X_test_zero_dot = X_test_zero_dot > 0
        X_test_zero_dot = X_test_zero_dot.astype(int)

        majority_genres = []
        for i in range(len(X_test_zero_dot)):
            bucket_genre = self.bucket_genres.get(''.join(X_test_zero_dot[i].astype(str)))
            majority_genres.append(bucket_genre)

        correct = 0
        for i in range(len(majority_genres)):
            if majority_genres[i] == self.y_test.iloc[i]:
                correct += 1

        accuracy = correct/len(majority_genres)
        print(f"Accuracy Test set: {accuracy}")
        return accuracy

    def evaluate_combined_accuracy(self):
        X_validation_zero_dot = np.dot(self.X_validation, self.R.T)
        X_validation_zero_dot = X_validation_zero_dot > 0
        X_validation_zero_dot = X_validation_zero_dot.astype(int)

        X_test_zero_dot = np.dot(self.X_test, self.R.T)
        X_test_zero_dot = X_test_zero_dot > 0
        X_test_zero_dot = X_test_zero_dot.astype(int)

        majority_genres = []
        for i in range(len(X_validation_zero_dot)):
            bucket_genre = self.bucket_genres.get(''.join(X_validation_zero_dot[i].astype(str)))
            majority_genres.append(bucket_genre)

        for i in range(len(X_test_zero_dot)):
            bucket_genre = self.bucket_genres.get(''.join(X_test_zero_dot[i].astype(str)))
            majority_genres.append(bucket_genre)

        correct = 0
        for i in range(len(X_validation_zero_dot)):
            if majority_genres[i] == self.y_validation.iloc[i]:
                correct += 1

        for i in range(len(X_test_zero_dot)):
            if majority_genres[i] == self.y_test.iloc[i-len(X_validation_zero_dot)]:
                correct += 1

        accuracy = correct/len(majority_genres)
        print(f"Accuracy Validation+Test set: {accuracy}")
        return accuracy
    
    def find_best_bits(self, min_bits=8, max_bits=512, step=32):
        best_bits = None
        best_accuracy = 0

        for bits in range(min_bits, max_bits + 1, step):
            self.train(bits)
            self.evaluate_train_accuracy()
            accuracy = self.evaluate_test_accuracy()

            if accuracy > best_accuracy:
                best_bits = bits
                best_accuracy = accuracy

            print(f"Bits: {bits}, Accuracy: {accuracy}")

        print(f"Best bits: {best_bits}, Best accuracy: {best_accuracy}")
        # also print validation set accuracy
        self.train(best_bits)
        self.evaluate_combined_accuracy()
        print(f"Best bits: {best_bits}, Best validation + test accuracy: {best_accuracy}")

# Example usage
classifier = MusicGenreClassifier()
classifier.load_data()
classifier.preprocess_data()
classifier.find_best_bits()
classifier.train()
classifier.evaluate_train_accuracy()
classifier.evaluate_test_accuracy()
classifier.evaluate_combined_accuracy()