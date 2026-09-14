from src.data import load_golden_set
from src.intents import VALID_INTENTS
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

class MajorityClassifier:
    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        # find majority
        c = Counter(y)
        self.majority_class = c.most_common(1)[0][0]

    def predict(self, X):
        return [self.majority_class] * len(X)
        
    def predict_single(self, text):
        return self.majority_class

if __name__ == '__main__':
    # test
    clf = MajorityClassifier()
    clf.fit([], ['payment_issue', 'app_bug', 'app_bug'])
    print(clf.predict(["test"]))
