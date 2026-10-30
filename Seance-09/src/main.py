#coding:UTF-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import scipy
import scipy.stats
import math

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1
print("Question 1")
temperature = ouvrirUnFichier("data/temperature.csv")

# Question 1 - Premier tiret
print("Question 1 - Premier tiret")
variables = ["Température_en_janvier", "Latitude", "Longitude", "Altitude"]
print(temperature[variables].corr())

# Question 1 - Deuxième tiret
print("Question 1 - Deuxième tiret")
print(temperature[variables].describe())

# Question 1 - Troisième tiret
print("Question 1 - Troisième tiret")
villes = temperature["Ville"]
y = temperature["Température_en_janvier"]
X = temperature[["Latitude", "Longitude", "Altitude"]]
print(villes.head())
print(y.head())
print(X.head())

# Question 1 - Quatrième tiret
print("Question 1 - Quatrième tiret")
X_sm = sm.add_constant(X)
modele = OLS(y, X_sm).fit()
print(modele.params)

# Question 1 - Cinquième tiret
print("Question 1 - Cinquième tiret")
print("Coefficients :")
print(modele.params)
print("Coefficient de détermination R2 :", modele.rsquared)
print(modele.summary())
print("P-values :")
print(modele.pvalues)

# Question 1 - Sixième tiret
print("Question 1 - Sixième tiret")
regression = LinearRegression().fit(X, y)
print("Coefficients (coef_) :", regression.coef_)
print("Constante (intercept_) :", regression.intercept_)

# Question 1 - Septième tiret
print("Question 1 - Septième tiret")

# Question 2
print("Question 2")
geomarketing = ouvrirUnFichier("data/geomarketing.csv")

# Question 2 - Premier tiret
print("Question 2 - Premier tiret")
# L'A.C.P. a déjà été faite : les variables significatives sont données par la consigne.
explicatives = ["surface_totale", "potentiel_Z20", "nb_primaire_Z10", "nb_primaire_Z20",
                "nb_gsa_Z10", "nb_pharmacie_Z5", "nb_conc2_Z10", "nb_conc2_Z20",
                "P10_POP_Z15", "P10_MEN_Z10"]
print(explicatives)

# Question 2 - Deuxième tiret
print("Question 2 - Deuxième tiret")
y2 = geomarketing["ca"]
X2 = geomarketing[explicatives]
print(y2.head())
print(X2.head())

# Question 2 - Troisième tiret
print("Question 2 - Troisième tiret")
modele2 = OLS(y2, sm.add_constant(X2)).fit()
print("Coefficients :")
print(modele2.params)
print("Coefficient de détermination R2 :", modele2.rsquared)
print(modele2.summary())
print("P-values :")
print(modele2.pvalues)

# Question 2 - Quatrième tiret
print("Question 2 - Quatrième tiret")

# Bonus
print("Bonus")
toutes = geomarketing.drop(columns = ["enseigne", "ca"])
modele3 = OLS(y2, sm.add_constant(toutes)).fit()
print("R2 (toutes les variables) :", modele3.rsquared)
print("R2 ajusté (toutes les variables) :", modele3.rsquared_adj)
print("R2 ajusté (variables de l'A.C.P.) :", modele2.rsquared_adj)
print(modele3.summary())
