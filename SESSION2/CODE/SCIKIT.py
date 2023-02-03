import numpy as np

#LOAD DATA
def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('C:\\Users\\msthu\\PycharmProjects\\pythonProject1\\Training phase I\\SESSION2\\DATA\\words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)

    return data, labels


def compute_accuracy(predicted_y, expected_y):
  matches = np.equal(predicted_y, expected_y)
  accuracy = np.sum(matches.astype(float)) / len(expected_y)
  return accuracy
#SVM
from sklearn.svm import LinearSVC

train_X, train_Y = load_data(data_path='C:\\Users\\msthu\\PycharmProjects\\pythonProject1\\Training phase I\\SESSION2\\DATA\\20news-train-tfidf.txt')
classifier = LinearSVC(
    C=10.0, # penalty coeff
    tol=0.001, # tolerance for stopping criteria
    verbose=True # whether prints out logs or not
)
classifier.fit(train_X, train_Y)
test_X, test_Y = load_data(data_path='C:\\Users\\msthu\\PycharmProjects\\pythonProject1\\Training phase I\\SESSION2\\DATA\\20news-test-tfidf.txt')
predicted_y = classifier.predict(test_X)
accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_Y)
print("Accuracy: {}".format(accuracy))
#KERNEL SVM
from sklearn.svm import SVC

classifier_ker = SVC(
    C=10.0,  # penalty coeff
    kernel='rbf',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    gamma=0.1,
    tol=0.001,  # tolerance for stopping criteria
    verbose=True  # whether prints out logs or not
)
classifier_ker.fit(train_X, train_Y)
predicted_y_ker = classifier_ker.predict(test_X)
accuracy_ker = compute_accuracy(predicted_y=predicted_y_ker, expected_y=test_Y)
print("Accuracy: {}".format(accuracy_ker))
#KMEANS
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=20,
    init='k-means++',
    n_init=5,  # number of time that kmeans runs with differently initialized centroids
    tol=1e-3,  # threshold for acceptable minimum error decrease
    random_state=6  # set to get deterministic results
)
kmeans.fit(train_X)
predicted_y_kmeans = kmeans.predict(test_X)
accuracy_kmeans = compute_accuracy(predicted_y=predicted_y_kmeans, expected_y=test_Y)
print("Accuracy: {}".format(accuracy_kmeans))

Accuracy: 0.05435743326851688

