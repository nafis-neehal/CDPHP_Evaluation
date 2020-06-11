from sklearn.metrics import precision_score, accuracy_score, recall_score, balanced_accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, log_loss, roc_curve, brier_score_loss, confusion_matrix

#for evaluation
class Score:

    def __init__(self, model_name):
        self.model_name = model_name
        self.experimental_samples = -1
        self.precision = {}
        self.recall = {}
        self.accuracy = {}
        self.balanced_acc = {}
        self.f1_score = {}
        self.roc_auc_score = {}
        self.log_loss = {}
        self.brier_score_loss = {}
        self.confusion_matrix = {}

    def get_precision(self, y_true, y_predict, average='binary'):
        return precision_score(y_true, y_predict, average)

    def get_recall(self, y_true, y_predict, average='binary'):
        return recall_score(y_true, y_predict, average)

    def get_accuracy(self, y_true, y_predict):
        return recall_score(y_true, y_predict)

    def get_balanced_acc(self, y_true, y_predict):
        return balanced_accuracy_score(y_true, y_predict)

    def get_f1_score(self, y_true, y_predict, average='binary'):
        return f1_score(y_true, y_predict, average='binary')

    def get_roc_auc_score(self, y_true, y_predict):
        try:
            return roc_auc_score(y_true, y_predict)
        except ValueError:
            return 0.0

    def get_log_loss(self, y_true, y_predict):
        return log_loss(y_true, y_predict, labels=[0.0, 1.0])

    def get_brier_score_loss(self, y_true, y_predict):
        return brier_score_loss(y_true, y_predict)

    def get_confusion_matrix(self, y_true, y_predict):
        return confusion_matrix(y_true, y_predict)
