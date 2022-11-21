import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import keras
from keras.layers import Dense, Dropout, Flatten


def moda(input_array):
    unique_values, counts = np.unique(input_array, return_counts=True)
    index = np.argmax(counts)
    return int(unique_values[index])


class Classifiers:
    '''Classe per allenare e predire il dataset, utilizzando i vari classificatori

    Riceve come input un dizionario contenente i vari classificatori e
    restituisce come output una matrice, il cui numero di colonne dipende dal numero
    di classificatori utilizzati.
    Ogni colonna della matrice contiente tutte le predizioni di X_test fatte con quello specifico classificatore'''
    def __init__(self,clfs):
        self.clfs = clfs

    def fit(self,X_train,y_train):
        for classifier in self.clfs:
            classifier.fit(X_train,y_train)

    def predict(self,X_test):
        y_pred = np.zeros(shape=(X_test.shape[0],len(self.clfs)))
        for n_clfs,classifier in enumerate(self.clfs):
            y_pred[:,n_clfs] = classifier.predict(X_test)
        return y_pred



if __name__ == '__main__':
    path = "../Data/travel_insurance.csv"
    dataset = pd.read_csv(path,index_col=0)

    ''' INFO SUL DATASET'''
    print(f'Shape del Dataset: {dataset.shape} \n')
    print(f'Colonne del Dataset: {dataset.columns} \n')
    print('INFO \n',dataset.info())
    print('DESCRIBE \n',dataset.describe().to_string())
    print(f'Valori nulli di ogni colonna \n {dataset.isna().sum()}')

    ''' FEATURES ENCODING'''
    dataset['Employment Type'].replace('Government Sector',0,inplace=True)
    dataset['Employment Type'].replace('Private Sector/Self Employed', 1, inplace=True)
    dataset['GraduateOrNot'].replace('No', 0, inplace=True)
    dataset['GraduateOrNot'].replace('Yes', 1, inplace=True)
    dataset['FrequentFlyer'].replace('No', 0, inplace=True)
    dataset['FrequentFlyer'].replace('Yes', 1, inplace=True)
    dataset['EverTravelledAbroad'].replace('No', 0, inplace=True)
    dataset['EverTravelledAbroad'].replace('Yes', 1, inplace=True)

    ''' HEATMAP CORRELAZIONI'''
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(), cmap="YlGnBu", annot=True, fmt=".2f",square=True)
    plt.show()

    X = np.array(dataset.iloc[:, :-1])
    y = np.array(dataset.iloc[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #SCALING Normale Standard
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    classifiers = {KNeighborsClassifier(n_neighbors=7):'KNN k=7',
                   DecisionTreeClassifier(criterion='gini'):'Decision Tree gini',
                   DecisionTreeClassifier(criterion='entropy'):'Decision Tree entropy',
                   SVC(kernel='linear'):'SVC linear',
                   SVC(kernel='rbf',C=100, gamma=1):'SVC rbf C=100-gamma=1',
                  RandomForestClassifier(criterion='gini'):'Random Forest gini',
                  RandomForestClassifier(criterion='entropy'):'Random Forest entropy'}

    '''uso la classe creata precedentemente per allenare e fare le predizioni 
    utilizzando i classificatori riportati qui sopra'''
    models = Classifiers(classifiers)
    models.fit(X_train,y_train)
    y_preds = models.predict(X_test)

    nn = keras.Sequential([
        Flatten(input_dim=8),
        Dense(units=400, activation='relu'),
        Dense(units=200, activation='relu'),
        Dense(units=50, activation='relu'),
        Dense(units=10, activation='relu'),
        Dropout(0.10),
        Dense(units=1, activation='sigmoid')])

    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, batch_size=20, epochs=20)
    nn_pred = nn.predict(X_test)

    ''' METRICHE DI VALUTAZIONE '''
    print('-------------------------------------------------------------------')
    for ix,col in enumerate(classifiers):
        acc = accuracy_score(y_test,y_preds[:,ix])
        print(f'Accuratezza {classifiers[col]} >>: {100*acc:.2f}%')

    comb_pred = np.zeros(shape=(X_test.shape[0]))
    for row in range(comb_pred.shape[0]):
        comb_pred[row] = moda(y_preds[row,:])
    comb_acc = accuracy_score(y_test,comb_pred)
    print(f'Accuratezza della Moda delle predizioni >>: {100 * comb_acc:.2f}%')

    nn_pred = np.around(nn_pred,decimals=0)
    nn_pred = np.squeeze(nn_pred)
    print(f'Accuratezza Rete Neurale >>: {np.mean(nn_pred == y_test) * 100: .2f} %')

    print('-------------------------------------------------------------------')
    report = classification_report(y_test,comb_pred)
    print('REPORT \n',report)

    ''' MATRICE DI CONFUSIONE'''
    y_test=np.where(y_test==1,'Buy','Does Not Buy')
    comb_pred=np.where(comb_pred == 1, 'Buy', 'Does Not Buy')
    skplt.metrics.plot_confusion_matrix(y_test, comb_pred)
    plt.title('Matrice di Confusione')
    plt.xlabel('Label Predette')
    plt.ylabel('Label Reali')
    plt.show()
