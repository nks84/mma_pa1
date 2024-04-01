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
    #todo
    def set_config(self):
        pass

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
    # Parameter TODO things
    # also parameter l wäre wie ich verstanden habe das hier das ganze quasi öfters läuft mit verschiedenen random matrix
    # k ist jetzt immer der bucket size -> ist also wv ähnliche man suchen soll 
    # m = metric
    # also mabye ist net bucket ne majority sondern man sollte im bucket dann die änhlichsten suchen mit metric m und die dann ausgeben und dann schauen ob die dann auch die gleiche genre haben
    def train(self, bits=32, amount_of_R=10):
        for _ in range(amount_of_R):
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

    def find_matching_songs(self, song_input, metric='euclid', cut=10):
        # find matching bucket for one single song and return the index of the matching songs
        song = np.dot(song_input, self.R.T)
        song = song > 0
        song = song.astype(int)
        hash_str = ''.join(song.astype(str))
        matching_songs = self.buckets.get(hash_str)

        # for all matching songs, calculate the distance to the input song
        # sort the songs by distance and return the index of the songs
        filtered_songs = []
        distance = 0
        if matching_songs is None:
            return []
        for element in matching_songs:
            if metric == "euclid":
                distance = np.linalg.norm(self.X_train[element] - song_input)
            elif metric == "cosine":
                # Calculate cosine similarity
                dot_product = np.dot(self.X_train[element], song_input)
                norm_song = np.linalg.norm(self.X_train[element])
                norm_input = np.linalg.norm(song_input)
                similarity = dot_product / (norm_song * norm_input)
                # Convert similarity to distance (cosine distance)
                distance = 1 - similarity
            
            filtered_songs.append((element, distance))
        sorted_songs = sorted(filtered_songs, key=lambda x: x[1])
        sorted_songs = sorted_songs[:cut]
        return sorted_songs


    
    def test_accuracy_with_find_matching_songs(self,cut = 10):
        correct = 0
        for i in range(len(self.X_test)):
            song = self.X_test[i]
            matching_songs = self.find_matching_songs(song,cut)
            if len(matching_songs) == 0:
                continue
            genres = []
            for element in matching_songs:
                genres.append(self.y_train.iloc[element[0]])
            if max(set(genres), key=genres.count) == self.y_test.iloc[i]:
                correct += 1
        accuracy = correct/len(self.X_test)
        #print(f"Accuracy Test set advanced: {accuracy}")
        return accuracy
    def validate_accuracy_with_find_matching_songs(self,cut = 10):
        correct = 0
        for i in range(len(self.X_validation)):
            song = self.X_validation[i]
            matching_songs = self.find_matching_songs(song,cut)
            if len(matching_songs) == 0:
                continue
            genres = []
            for element in matching_songs:
                genres.append(self.y_train.iloc[element[0]])
            if max(set(genres), key=genres.count) == self.y_validation.iloc[i]:
                correct += 1
        accuracy = correct/len(self.X_validation)
        print(f"Accuracy Validation set advanced: {accuracy}")
        return accuracy
    
    def combined_accuracy_with_find_matching_songs(self,cut = 10):
        correct = 0
        for i in range(len(self.X_validation)):
            song = self.X_validation[i]
            matching_songs = self.find_matching_songs(song,cut)
            if len(matching_songs) == 0:
                continue
            genres = []
            for element in matching_songs:
                genres.append(self.y_train.iloc[element[0]])
            if max(set(genres), key=genres.count) == self.y_validation.iloc[i]:
                correct += 1
        for i in range(len(self.X_test)):
            song = self.X_test[i]
            matching_songs = self.find_matching_songs(song,cut)
            if len(matching_songs) == 0:
                continue
            genres = []
            for element in matching_songs:
                genres.append(self.y_train.iloc[element[0]])
            if max(set(genres), key=genres.count) == self.y_test.iloc[i]:
                correct += 1
        accuracy = correct/(len(self.X_test)+len(self.X_validation))
        print(f"Accuracy Validation+Test set advanced: {accuracy}")
        return accuracy

    
    def find_best_paramters(self):
        classifier = MusicGenreClassifier()
        classifier.load_data()
        classifier.preprocess_data()
        
        best_accuracy = 0
        best_bits = 0
        best_amount_of_R = 0
        best_k = 0
        for bits in range(32, 512,10):
            for amount_of_R in range(1, 35,5):
                for k in range(2, 30,5):
                    classifier.train(bits, amount_of_R)
                    print(f"Bits: {bits}, Amount of R: {amount_of_R}, k: {k}")
                    accuracy = classifier.test_accuracy_with_find_matching_songs(k)
                    print (f"Accuracy: {accuracy}, Bits: {bits}, Amount of R: {amount_of_R}, k: {k}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_bits = bits
                        best_amount_of_R = amount_of_R
                        best_k = k
        print(f"Best accuracy: {best_accuracy}, Best bits: {best_bits}, Best amount of R: {best_amount_of_R}, Best k: {best_k}")

def run():
    classifier = MusicGenreClassifier()
    classifier.load_data()
    classifier.preprocess_data()
    classifier.train(128,5)
    classifier.combined_accuracy_with_find_matching_songs()
    accuracy = classifier.test_accuracy_with_find_matching_songs(20)
    print(f"Accuracy Test set : {accuracy}")
def run_best():
    classifier = MusicGenreClassifier()
    classifier.find_best_paramters()
    
run()