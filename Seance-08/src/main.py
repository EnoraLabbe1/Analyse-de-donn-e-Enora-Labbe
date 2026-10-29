#coding:utf8

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import scipy
import scipy.stats
import prince
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

os.makedirs("img", exist_ok=True)

# Question 1
print("Question 1 : Analyse en composantes principales")
# Analyse en composantes principales
sipri = ouvrirUnFichier("data/SIPRI-1949-2025.csv")
# Question 1b
print("Question 1b")
# "Code ISO" : code ISO du territoire
# "Country" : territory name
# "État" : nom du territoire
# "Continent" : Continent
# "1949","1950","1951","1952","1953","1954","1955","1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025" : 77 colonnes des différentes années
individus = sipri["Code ISO"]
donnees = sipri.drop(columns = ["Code ISO"])
donnees = donnees.drop(columns = ["Country"])
donnees = donnees.drop(columns = ["État"])
donnees = donnees.drop(columns = ["Continent"])
print(donnees.shape)

# Question 1c
print("Question 1c")
# Remplacement des NaN par 0
donnees = donnees.fillna(0)
# Centrer-réduire
donnees_cr = StandardScaler().fit_transform(donnees)
# ACP
n = donnees_cr.shape[0]
p = donnees_cr.shape[1]
colonnes = ["F" + str(i + 1) for i in range(p)]
pca = PCA(n_components = p)
pca.fit(donnees_cr)

# Question 1d
print("Question 1d")
# Affichage de l'A.C.P.
print(pca)
# Variance expliquée

# Question 1e
print("Question 1e")
print(pca.explained_variance_)
# Variance expliquée en pourcentage
print(pca.explained_variance_ratio_ * 100)
# Calcul des valeurs propres
valeurs_propres = (n - 1) / n * pca.explained_variance_
print(valeurs_propres)

# Question 1f
print("Question 1f")
# Coordonnées des individus
coord = pca.transform(donnees_cr)
print(pd.DataFrame(coord, index = individus, columns = colonnes))

# Question 1g
print("Question 1g")
# Visualisation des individus
plt.figure(figsize = (12, 12))
plt.scatter(coord[:, 0], coord[:, 1], s = 10)
for i, code in enumerate(individus):
    plt.annotate(code, (coord[i, 0], coord[i, 1]), fontsize = 7)
plt.axhline(0, color = "grey")
plt.axvline(0, color = "grey")
plt.xlabel("F1 (" + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + " %)")
plt.ylabel("F2 (" + str(round(pca.explained_variance_ratio_[1] * 100, 2)) + " %)")
plt.title("Mapping des individus : plan factoriel F1-F2")
plt.savefig("img/acp_individus.png", dpi = 150)
plt.close()
# Contribution des individus
contributions = coord ** 2 / (n * valeurs_propres) * 100

# Question 1h
print("Question 1h")
print(pd.DataFrame(contributions, index = individus, columns = colonnes).loc[["usa", "chn"], ["F1", "F2"]])
# Qualité de représentation (cosinus carré) des individus
cos2 = coord ** 2 / (coord ** 2).sum(axis = 1, keepdims = True)
print(pd.DataFrame(cos2, index = individus, columns = colonnes).loc[["usa", "chn"], ["F1", "F2"]])
# Coordonnées des variables : cercle de corrélation

# Question 1i
print("Question 1i")
# Racine carrée des valeurs propres
racine_valeurs_propres = np.sqrt(valeurs_propres)
# Matrice vide pour avoir les coordonnées
coord_var = np.zeros((p, p))
for i in range(p):
    coord_var[:, i] = pca.components_[i] * racine_valeurs_propres[i]
# Création d'un DataFrame
coord_var = pd.DataFrame(coord_var, index = donnees.columns, columns = colonnes)
print(coord_var)
# Visualisation du cercle de corrélation
plt.figure(figsize = (10, 10))
for annee in coord_var.index:
    plt.arrow(0, 0, coord_var.loc[annee, "F1"], coord_var.loc[annee, "F2"], head_width = 0.01, color = "tab:blue")
    plt.annotate(annee, (coord_var.loc[annee, "F1"], coord_var.loc[annee, "F2"]), fontsize = 7)
plt.gca().add_patch(plt.Circle((0, 0), 1, fill = False))
plt.axhline(0, color = "grey")
plt.axvline(0, color = "grey")
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Cercle de corrélation")
plt.gca().set_aspect("equal")
plt.savefig("img/acp_cercle_correlation.png", dpi = 150)
plt.close()

# Question 2
print("Question 2 : Analyse factorielle de correspondances multiples")
# Analyse factorielle de correspondances multiples
# Question 2a
chiens = ouvrirUnFichier("data/chiens.csv")

# Question 2b
races = chiens["Race"]
variables = chiens[["Taille", "Poids", "Vitesse", "Intelligence", "Affection", "Agressivité", "Fonction", "Origine"]]

# Question 2c
print("Question 2c")
# Calcul du tableau disjonctif complet
tdc = pd.get_dummies(variables, dtype = int)
print(tdc)

# Question 2d
print("Question 2d")
mca = prince.MCA(n_components = 8)
mca = mca.fit(tdc)

# Question 2e
print("Question 2e")
# Calcul des valeurs propres
print(mca.eigenvalues_)
inertie_totale = (tdc.shape[1] - variables.shape[1]) / variables.shape[1]
print(np.array(mca.eigenvalues_) / inertie_totale * 100)

# Question 2f
print("Question 2f")
# Coordonnées des lignes et des colonnes
coord_lignes = mca.row_coordinates(tdc)
coord_colonnes = mca.column_coordinates(tdc)
print(coord_lignes)
print(coord_colonnes)

# Question 2g
print("Question 2g")
# Visualiser les deux premiers facteurs
plt.figure(figsize = (12, 12))
plt.scatter(coord_lignes[0], coord_lignes[1], color = "tab:blue", s = 10)
for i, race in enumerate(races):
    plt.annotate(race, (coord_lignes.iloc[i, 0], coord_lignes.iloc[i, 1]), color = "tab:blue", fontsize = 8)
plt.scatter(coord_colonnes[0], coord_colonnes[1], color = "tab:red", marker = "^")
for modalite in coord_colonnes.index:
    plt.annotate(modalite, (coord_colonnes.loc[modalite, 0], coord_colonnes.loc[modalite, 1]), color = "tab:red", fontsize = 8)
plt.axhline(0, color = "grey")
plt.axvline(0, color = "grey")
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Mapping de l'A.C.M. : plan factoriel F1-F2")
plt.savefig("img/acm_mapping.png", dpi = 150)
plt.close()


# Question 2h
print("Question 2h")

# Coordonnées calculées avec Prince
coord_lignes = mca.row_coordinates(tdc)
coord_colonnes = mca.column_coordinates(tdc)

# Valeurs propres calculées avec Prince
valeurs_propres = pd.Series(mca.eigenvalues_)

# Contributions des lignes
masses_lignes = pd.Series(1 / len(tdc), index=coord_lignes.index)
contributions_lignes = coord_lignes.pow(2).multiply(masses_lignes, axis=0)
contributions_lignes = contributions_lignes.div(valeurs_propres, axis=1)
print("Contributions des lignes :")
print(contributions_lignes)

# Contributions des colonnes
masses_colonnes = tdc.sum(axis=0) / (len(tdc) * variables.shape[1])
masses_colonnes.index = tdc.columns
contributions_colonnes = coord_colonnes.pow(2).multiply(masses_colonnes, axis=0)
contributions_colonnes = contributions_colonnes.div(valeurs_propres, axis=1)
print("Contributions des colonnes :")
print(contributions_colonnes)

# Question bonus
print("Question bonus")
# C.A.H. (méthode de Ward) sur les deux premiers facteurs de l'A.C.P.
liaison = sch.linkage(coord[:, :2], method = "ward")
plt.figure(figsize = (16, 8))
sch.dendrogram(liaison, labels = individus.to_list(), leaf_font_size = 6)
plt.title("Dendrogramme de la C.A.H.")
plt.savefig("img/cah_dendrogramme.png", dpi = 150)
plt.close()
# Découpage en 4 classes
classes = sch.fcluster(liaison, t = 4, criterion = "maxclust")
print(pd.Series(classes, index = individus).sort_values())
plt.figure(figsize = (12, 12))
plt.scatter(coord[:, 0], coord[:, 1], c = classes, cmap = "tab10", s = 15)
for i, code in enumerate(individus):
    plt.annotate(code, (coord[i, 0], coord[i, 1]), fontsize = 7)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Classes de la C.A.H. sur le plan factoriel F1-F2")
plt.savefig("img/cah_classes.png", dpi = 150)
plt.close()
