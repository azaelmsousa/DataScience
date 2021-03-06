import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn

from matplotlib.colors import ListedColormap

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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

# Initialization

names = ["Nearest Neighbors(3)", "Nearest Neighbors(5)", "Nearest Neighbors(10)", "Linear SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# Loading data
path = "../data/Tabelas_Gephi/gephi-allNodes.csv"
df_gephi = pd.read_csv(path).fillna(0)
df_gephi = df_gephi.rename(index=str, columns={"Id": "Município"})

path = "../data/Tabelas_Dengue_2017-2019/Boletim_de_Arboviroses_-_Predominante_-_Outubro_2018_831municipios_1.csv"
df_arboviroses = pd.read_csv(path,decimal=",")

path = "../data/Tabelas_Dengue_2017-2019/LIRAaLIA_Janeiro2019_(2).csv"
df_libralia = pd.read_csv(path,decimal=",")
df_libralia.drop(columns=['IBGE','Regional','STATUS'],inplace=True)


path = "../data/Tabelas_Dengue_2017-2019/Dengue_2018.csv"
df_dengue = pd.read_csv(path,decimal=",")
df_dengue = df_dengue[['Municipio','Total','Incidência','Situação']]
df_dengue = df_dengue.rename(index=str, columns={"Municipio": "Município", "Total": "Total_Casos"})


# Preparing data
df_merge = df_libralia.merge(df_dengue,on=['Município'])
df_data = df_merge.merge(df_gephi,on=['Município'])
df_label = df_data[['Situação']]
print(df_data.head(10))
df_data.drop(columns=['Município','Label','timeset','pageranks','Incidência','Situação'],inplace=True)
X_columns = df_data.columns
X = df_data.values
y = np.squeeze(df_label.values)

print(df_data.head(10))
print(X[0:10,:])
print(y[0:10])

# one hot encoding
class_names = np.unique(y)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = y.astype(np.int)

predominante = df_data.columns.get_loc("Predominante")
index = X.shape[1]-predominante
X[:,-index] = le.fit_transform(X[:,-index])
X = X.astype(np.float)
X = RobustScaler().fit_transform(X)


col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# Separating traine and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Applying classification model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# Computing metrics
print("Random Forest")
print("Accuracy: "+str(accuracy_score(y_test,y_pred)))
print("Precision: "+str(precision_score(y_test,y_pred,average='macro')))
print("Recall: "+str(recall_score(y_test,y_pred,average='macro')))
print("="*60)

feature_importances = pd.DataFrame(clf.feature_importances_, index = X_columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)
print("="*60)

'''
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
'''

#Applying classifications
print("="*60)
for name, clf in zip(names, classifiers):
  print(name)
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)

  print("Accuracy: "+str(accuracy_score(y_test,y_pred)))
  
  print("Precision: "+str(precision_score(y_test,y_pred,average='macro')))
  
  print("Recall: "+str(recall_score(y_test,y_pred,average='macro')))

  #plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
  #                    title='Normalized confusion matrix')

  #plt.show()
  print("="*60)
  




'''
# Compute the correlation matrix
sns.set(style="white")
df_data[['Predominante']] = df_data[['Predominante']].apply(le.fit_transform)
df_data = df_data.astype('float64')
corr = df_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

figure = sns_plot.get_figure()

figure.savefig("output.png")
'''