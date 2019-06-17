import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Loading data
path = "../data/Tabelas_Gephi/gephi-allNodes.csv"
df_gephi = pd.read_csv(path).fillna(0)
print(df_gephi.head(10))

df_label = df_gephi[['modularity_class']]
df_gephi.drop(columns=['Id','Label','timeset','modularity_class'],inplace=True)
X_columns = df_gephi.columns
X = df_gephi.values
y = np.squeeze(df_label.values)

class_names = np.unique(y)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = y.astype(np.int)

X[:,-3] = le.fit_transform(X[:,-3])
X = X.astype(np.float)
X = RobustScaler().fit_transform(X)

# Separating traine and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)


# Applying classification model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# Computing metrics
print("="*60)
print("Accuracy: "+str(accuracy_score(y_test,y_pred)))
print("="*60)
print("Precision: "+str(precision_score(y_test,y_pred,average='macro')))
print("="*60)
print("Recall: "+str(recall_score(y_test,y_pred,average='macro')))
print("="*60)

feature_importances = pd.DataFrame(clf.feature_importances_, index = X_columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)
print("="*60)

plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# Compute the correlation matrix
sns.set(style="white")
df_gephi = df_gephi.astype('float64')
corr = df_gephi.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

figure = sns_plot.get_figure()

figure.savefig("output.png")