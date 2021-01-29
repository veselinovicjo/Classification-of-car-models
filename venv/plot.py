import pandas

skup_podataka=pandas.read_csv('auto.txt')

print(skup_podataka.head())

oznake={
    'Kupovina':['vvisoka','visoka', 'srednja', 'niska'],
    'Održavanje':['vvisoka','visoka', 'srednja', 'niska'],
    'Vrata':['2','3','4','5više'],
    'Broj_osoba':['2','4','više'],
    'Prtljažnik':['mali','srednja','veliki'],
    'Bezbednost':['niska','srednja','visoka'],
    'Klasa':['neprihv', 'prihv', 'dobar','vdobar']

}

from sklearn import preprocessing
kodirane_oznake={}
skup_podataka_kodiran=pandas.DataFrame()
for column in skup_podataka:
    if column in oznake:
        kodirane_oznake[column]=preprocessing.LabelEncoder()
        kodirane_oznake[column].fit(oznake[column])
        skup_podataka_kodiran[column]=kodirane_oznake[column].transform(skup_podataka[column])
    else:
        skup_podataka_kodiran[column]=skup_podataka[column]

print(skup_podataka_kodiran.head())


import numpy as np
osobine=np.array(skup_podataka_kodiran.drop(['Klasa'], 1))
oznaka=np.array(skup_podataka_kodiran['Klasa'])

from sklearn import model_selection
param_train, param_test, oznaka_train, oznaka_test=model_selection.train_test_split(
    osobine,
    oznaka,
    test_size=0.1
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

random_forest_classifier=RandomForestClassifier(n_estimators=100, max_depth=6)
random_forest_classifier.fit(param_train,oznaka_train)

extra_trees_classifier=ExtraTreesClassifier(n_estimators=100, max_depth=6)
extra_trees_classifier.fit(param_train,oznaka_train)

from sklearn.metrics import classification_report
print(
    classification_report(
        oznaka_test,
        random_forest_classifier.predict(param_test)
    )
)

print(
    classification_report(
        oznaka_test,
        extra_trees_classifier.predict(param_test)
    )
)

osobine2=np.array(skup_podataka_kodiran.drop(['Klasa','Vrata'],1))
oznaka2=np.array(skup_podataka_kodiran['Klasa'])
param2_train, param2_test, oznaka2_train, oznaka2_test=model_selection.train_test_split(
    osobine2,
    oznaka2,
    test_size=0.1
)

random_forest_classifier2=RandomForestClassifier(n_estimators=150,max_depth=8, criterion='entropy', max_features=5)
random_forest_classifier2.fit(param2_train,oznaka2_train)


from sklearn.metrics import classification_report
print(
    classification_report(
        oznaka2_test, random_forest_classifier2.predict(param2_test)
    )
)

extra_trees_classifier2=ExtraTreesClassifier(n_estimators=150, max_depth=8, criterion='entropy', max_features=5)
extra_trees_classifier2.fit(param2_train,oznaka2_train)

print(
    classification_report(
        oznaka2_test,extra_trees_classifier2.predict(param2_test)
    )
)