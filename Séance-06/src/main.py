#coding: UTF-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def couplesComplets(donnees, annee):
    """Retourne deux listes (energie, PIB) ne contenant que les couples complets."""
    pib = pd.to_numeric(donnees["PIB_" + str(annee)], errors="coerce")
    energie = pd.to_numeric(donnees["Utilisation_d_energie_" + str(annee)], errors="coerce")
    x, y = [], []
    for i in range(len(donnees)):
        if not (np.isnan(pib[i]) or np.isnan(energie[i])):
            x.append(energie[i])
            y.append(pib[i])
    return x, y

# Question 1
print("Question 1")
# Visualisation des données
# Pour résoudre le problème posé dans la question 1, il faut utiliser l'opérateur NaN de la bibliothèque Numpy dans une boucle.
#ouvrir un fichier 
donnees = ouvrirUnFichier("./data./pib-vs-energie.csv")
print(donnees[["PIB_2022", "Utilisation_d_energie_2022"]].head())

# Question 2
print("Question 2")
#garder les couples complets
x, y = couplesComplets(donnees, 2022)
print("Couples complets :", len(x), "sur", len(donnees), "territoires")

# Question 3
print("Question 3")
# Variable explicative (x) = énergie ; variable à expliquer (y) = PIB
regression = scipy.stats.linregress(x, y)
print("Pente :", regression.slope)
print("Ordonnée à l'origine :", regression.intercept)
print("R² :", regression.rvalue ** 2)
print("p-value :", regression.pvalue)

# Question 4
print("Question 4")
#corrélation
r, p = scipy.stats.pearsonr(x, y)
print("Corrélation de Pearson :", r, "(p-value :", p, ")")

# Question 5
print("Question 5")
#graphique 
x_arr = np.array(x)
plt.scatter(x, y, s=15, label="Territoires")
plt.plot(x_arr, regression.slope * x_arr + regression.intercept, color="red", label="Droite de régression")
plt.xlabel("Consommation énergétique (kg équivalent pétrole)")
plt.ylabel("PIB (dollars courants)")
plt.title("PIB et consommation énergétique en 2022")
plt.legend()
os.makedirs("./sorties", exist_ok=True)
plt.savefig("./sorties/regression_2022.png", dpi=150)
plt.show()
plt.close()

# Question 6
print("Question 6")

#Bonus
print("Bonus")
os.makedirs("./sorties/annees", exist_ok=True)
resultats = []
for annee in range(1962, 2023):
    x, y = couplesComplets(donnees, annee)
    if len(x) < 3:
        continue
    reg = scipy.stats.linregress(x, y)
    resultats.append([annee, len(x), reg.slope, reg.intercept, reg.rvalue, reg.rvalue ** 2, reg.pvalue])
    plt.scatter(x, y, s=15)
    x_arr = np.array(x)
    plt.plot(x_arr, reg.slope * x_arr + reg.intercept, color="red")
    plt.xlabel("Consommation énergétique")
    plt.ylabel("PIB (dollars courants)")
    plt.title("PIB et énergie en " + str(annee))
    plt.savefig("./sorties/annees/regression_" + str(annee) + ".png", dpi=100)
    plt.close()
tableau = pd.DataFrame(resultats, columns=["annee", "n", "pente", "ordonnee", "r", "r2", "pvalue"])
tableau.to_csv("./sorties/resultats_1962_2022.csv", index=False)
print(tableau.tail())
