#coding:UTF-8

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding = "utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
data = pd.DataFrame(ouvrirUnFichier("./data/resultats-elections-presidentielles-2022-1er-tour.csv"))
# print(data)

# "Code_du_département"
# "Libellé_du_département"
# "Inscrits"
# "Abstentions"
# "Votants"
# "Blancs"
# "Nuls"
# "Exprimés"
# "Nathalie_ARTHAUD"
# "Fabien_ROUSSEL"
# "Emmanuel_MACRON"
# "Jean_LASSALLE"
# "Marine_LE_PEN"
# "Éric_ZEMMOUR"
# "Jean_Luc_MÉLENCHON"
# "Anne_HIDALGO"
# "Yannick_JADOT"
# "Valérie_PÉCRESSE"
# "Philippe_POUTOU"
# "Nicolas_DUPONT_AIGNAN"

# Création du tableau de contingence
# Contrairement à l'usage, vous ne devez pas créer de tableau croisé dynamique, puisque le fichier est déjà un tableau de contingence
nomdescolonnes = ["Libellé_du_département", "Nathalie_ARTHAUD", "Fabien_ROUSSEL", "Emmanuel_MACRON", "Jean_LASSALLE", "Marine_LE_PEN", "Éric_ZEMMOUR", "Jean_Luc_MÉLENCHON", "Anne_HIDALGO", "Yannick_JADOT", "Valérie_PÉCRESSE", "Philippe_POUTOU", "Nicolas_DUPONT_AIGNAN"]
# print(nomdescolonnes)
dictionnaire = {}
for element in range(1, len(nomdescolonnes)):
    dictionnaire[nomdescolonnes[element]] = data[nomdescolonnes[element]]
tableauDeContingence = tableauDeContingence(data[nomdescolonnes[0]], dictionnaire)
# print(tableauDeContingence)

# Question 1
print("Question 1")
# Calculer les marges des lignes et des colonnes
margeColonnes = sommeDesColonnes(tableauDeContingence)
margeLignes = sommeDesLignes(tableauDeContingence)
print("Marges des colonnes :", margeColonnes)
print("Marges des lignes :", margeLignes)

# Question 2
print("Question 2")
#Vérifier les totaux des marges 
if sum(margeLignes) == sum(margeColonnes):
    print("Les totaux sont identiques :", sum(margeLignes))
else:
    print("Les totaux sont différents")

# Question 3
print("Question 3")
# Faire le test du chi2 avec les outils Scipy.stats
chi2, p, ddl, attendu = scipy.stats.chi2_contingency(tableauDeContingence)
print("chi2 =", chi2, "| ddl =", ddl, "| p-value =", p)
if p < 0.05:
    print("Il existe une liaison")
else:
    print("Pas de liaison")

# Question 4
print("Question 4")
# Calcul de l'intensité de liaison de Pearson
n = sum(margeLignes)
phi2 = chi2 / n
print("phi2 =", phi2)

# Question bonus
print("Question bonus")

#Analyse ANOVA
echantillons = ouvrirUnFichier("./data/Echantillonnage-100-Echantillons.csv")
f, pAnova = scipy.stats.f_oneway(echantillons["Pour"], echantillons["Contre"], echantillons["Sans opinion"])
print("ANOVA : F =", f, "| p-value =", pAnova)
print(echantillons.mean())

#Calcul A.F.C
P = tableauDeContingence.values / n
r = P.sum(axis = 1)
c = P.sum(axis = 0)
S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, valeursSingulieres, Vt = np.linalg.svd(S, full_matrices = False)
inertie = valeursSingulieres ** 2
print("Part d'inertie par axe (%) :", (100 * inertie / inertie.sum()).round(1))

coordLignes = U * valeursSingulieres / np.sqrt(r)[:, None]
coordColonnes = Vt.T * valeursSingulieres / np.sqrt(c)[:, None]

plt.scatter(coordLignes[:, 0], coordLignes[:, 1], s = 8, color = "grey")
for i, candidat in enumerate(tableauDeContingence.columns):
    plt.text(coordColonnes[i, 0], coordColonnes[i, 1], candidat, color = "red")
plt.axhline(0, color = "black", linewidth = 0.5)
plt.axvline(0, color = "black", linewidth = 0.5)
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
os.makedirs("./output", exist_ok = True)
plt.savefig("./output/afc.png", dpi = 120)
plt.show()