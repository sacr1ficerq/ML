from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score as precision, recall_score as recall, auc, precision_recall_curve
from sklearn.metrics import accuracy_score as accuracy, roc_auc_score
import pandas as pd

from isoforest import *

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

def auc_pr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def kappa(y_true, y_pred):
    p_o = accuracy(y_true, y_pred)
    a = y_true.mean() * y_pred.mean()
    b = (1 - y_true.mean()) * (1 - y_pred.mean())
    p_e = a + b
    return (p_o - p_e) / (1 - p_e)

def MCC(y_true, y_pred) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if d == 0.0:
        return 0.0
    
    return float(tp * tn - fp * fn) / np.sqrt(d)

X_train, y_train = train.drop(columns='Class'), train['Class']
X_test, y_test = test.drop(columns='Class'), test['Class']
X_val, y_val = val.drop(columns='Class'), val['Class']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_trees_list = [50, 100, 200]
subsample_rates = [0.1, 0.2, 0.5]
contamination = 0.01 

for n_trees in n_trees_list:
    for subsample in subsample_rates:
        print(f'{"N Trees:":<12} {n_trees:<12.4f}')
        print(f'{"AUC-PR:":<12} {subsample:<12.4f}')

        model = ExtendedIsolationForest(
            n_trees=n_trees,
            subsample_rate=subsample,
            contamination=contamination
        )

        model.fit(X_train_scaled)
        scores = model.score_samples(X_test_scaled)
        
        best_threshold = 0
        best_mcc = -1
        thresholds = np.linspace(np.min(scores), np.max(scores), 10)
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            mcc = MCC(y_test, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
        
        y_pred = (scores >= best_threshold).astype(int)

        print(f'{"Threshold:":<12} {best_threshold:<12.4f}')
        print(f'{"AUC-PR:":<12} {auc_pr(y_test, y_pred):<12.4f}')
        print(f'{"AUC-ROC:":<12} {roc_auc_score(y_test, y_pred):<12.4f}')
        print(f'{"Kappa:":<12} {kappa(y_test, y_pred):<12.4f}')
        print(f'{"MCC:":<12} {MCC(y_test, y_pred):<12.4f}')
        print()
