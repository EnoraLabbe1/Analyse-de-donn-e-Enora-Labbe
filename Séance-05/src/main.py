#coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math

def ouvrirUnFichier(nom):
    with open(nom, "r", encoding='utf-8') as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def conversionLog(liste):
    log = []
    for element in liste:
        log.append(math.log(element))
    return log

def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if pd.notna(pop[element]) and pop[element] != '-':
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop


def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element][1]])
    return classement

# Question 2
print("Question 2")

#ouvrir le fichier
depenses = ouvrirUnFichier("data/SIPRI-1949-2025.csv")

print(depenses.head())

# Question 3
print("Question 3")

# "Code ISO" : code ISO du territoire
# "Country" : territory name
# "État" : nom du territoire
# "Continent" : Continent
# "1949","1950","1951","1952","1953","1954","1955","1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970","1971","1972","1973","1974","1975","1976","1977","1978","1979","1980","1981","1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992","1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025" : 77 colonnes des différentes années

#isoler la colonne 2025 et supprimer les valeurs manquantes ou nulles
depenses2025 = list(depenses["2025"])

depenses2025_nettoyees = []

for valeur in depenses2025:
    if pd.notna(valeur) and valeur != 0.0:
        depenses2025_nettoyees.append(valeur)

print(depenses2025_nettoyees)

# Question 4
print("Question 4")

#classer les dépenses par ordre décroissant
depenses2025_ordonnees = ordreDecroissant(
    depenses2025_nettoyees
)

print(depenses2025_ordonnees)

# Question 5
print("Question 5")

#visualiser la loig rang-taille
rangs = list(range(1, len(depenses2025_ordonnees) + 1))

plt.figure()
plt.plot(rangs, depenses2025_ordonnees, marker=".")
plt.xlabel("Rang des États")
plt.ylabel("Dépenses militaires (milliards de dollars)")
plt.title("Loi rang-taille des dépenses militaires en 2025")
plt.grid(True)
plt.savefig("output/question5_loi_rang_taille_2025.png")
plt.show()

# Question 6
print("Question 6")

#convertir les axes en logritme
depenses_log = conversionLog(depenses2025_ordonnees)
rangs_log = conversionLog(rangs)

plt.figure()
plt.plot(rangs_log, depenses_log, marker=".")
plt.xlabel("Logarithme du rang")
plt.ylabel("Logarithme des dépenses militaires")
plt.title("Loi rang-taille en échelle logarithmique, 2025")
plt.grid(True)
plt.savefig("output/question6_loi_rang_taille_log_2025.png")
plt.show()

# Question 7
print("Question 7")
# Mettre ici votre réponse argumentée

# On ne réalise pas un test statistique classique directement sur les rangs.
# On peux étudier la relation entre le logarithme du rang
# et le logarithme des dépenses militaires.
# Une relation approximativement linéaire sur le graphique logarithmique
# peut être compatible avec une loi rang-taille.

# Question 9
print("Question 9")

#ouvrir le fichier 
population = ouvrirUnFichier("data/Le-Monde-HS-Etats-du-monde-2007-2026.csv")

print(population.head())

# Question 10
print("Question 10")

#ouvrir les 5 colonnes
etats = list(population["État"])
pop2007 = list(population["Pop 2007"])
pop2026 = list(population["Pop 2026"])
densite2007 = list(population["Densité 2007"])
densite2026 = list(population["Densité 2026"])

print(etats[:5])
print(pop2007[:5])
print(pop2026[:5])
print(densite2007[:5])
print(densite2026[:5])

# Question 11
print("Question 11")

#classer les population et les densités 
classement_pop2007 = ordrePopulation(pop2007, etats)
classement_pop2026 = ordrePopulation(pop2026, etats)

classement_densite2007 = ordrePopulation(densite2007, etats)
classement_densite2026 = ordrePopulation(densite2026, etats)

print("Population 2007 :", classement_pop2007[:5])
print("Population 2026 :", classement_pop2026[:5])
print("Densité 2007 :", classement_densite2007[:5])
print("Densité 2026 :", classement_densite2026[:5])

# Question 12
print("Question 12")

#comparer les classements 
comparaison_population = classementPays(
    classement_pop2007, classement_pop2026
)
comparaison_population.sort()

comparaison_densite = classementPays(
    classement_densite2007, classement_densite2026
)
comparaison_densite.sort()

print("Comparaison population :", comparaison_population[:5])
print("Comparaison densité :", comparaison_densite[:5])

# Question 13
print("Question 13")

#séparer les deux colonnes de rangs
rangs_population_2007 = []
rangs_population_2026 = []

for ligne in comparaison_population:
    rangs_population_2007.append(ligne[0])
    rangs_population_2026.append(ligne[1])

rangs_densite_2007 = []
rangs_densite_2026 = []

for ligne in comparaison_densite:
    rangs_densite_2007.append(ligne[0])
    rangs_densite_2026.append(ligne[1])

print(rangs_population_2007[:5])
print(rangs_population_2026[:5])
print(rangs_densite_2007[:5])
print(rangs_densite_2026[:5])

# Question 14
print("Question 14")

#Calculer les corrélations de Spearman et Kendall
resultat_spearman_population = scipy.stats.spearmanr(
    rangs_population_2007, rangs_population_2026
)
resultat_kendall_population = scipy.stats.kendalltau(
    rangs_population_2007, rangs_population_2026
)

resultat_spearman_densite = scipy.stats.spearmanr(
    rangs_densite_2007, rangs_densite_2026
)
resultat_kendall_densite = scipy.stats.kendalltau(
    rangs_densite_2007, rangs_densite_2026
)

print("Spearman population :", resultat_spearman_population)
print("Kendall population :", resultat_kendall_population)
print("Spearman densité :", resultat_spearman_densite)
print("Kendall densité :", resultat_kendall_densite)

# BONUS 1 : comparaison des classements militaires
annees_militaires = [str(annee) for annee in range(1949, 2026)]
classements_militaires = {}

for annee in annees_militaires:
    valeurs = depenses[["État", annee]].dropna()
    valeurs = valeurs[valeurs[annee] > 0]
    valeurs = valeurs.sort_values(annee, ascending=False)

    classement = {}
    for rang, etat in enumerate(valeurs["État"], start=1):
        classement[etat] = rang

    classements_militaires[annee] = classement

resultats_militaires = []

for i in range(len(annees_militaires)):
    for j in range(i + 1, len(annees_militaires)):
        annee1 = annees_militaires[i]
        annee2 = annees_militaires[j]

        etats_communs = set(classements_militaires[annee1]) & \
                        set(classements_militaires[annee2])

        if len(etats_communs) >= 2:
            rangs1 = [classements_militaires[annee1][e]
                      for e in etats_communs]
            rangs2 = [classements_militaires[annee2][e]
                      for e in etats_communs]

            spearman = scipy.stats.spearmanr(rangs1, rangs2)
            kendall = scipy.stats.kendalltau(rangs1, rangs2)

            resultats_militaires.append([
                annee1, annee2, len(etats_communs),
                spearman.correlation, kendall.correlation
            ])

table_militaire = pd.DataFrame(
    resultats_militaires,
    columns=["Année 1", "Année 2", "États communs",
             "Spearman", "Kendall"]
)

table_militaire.to_csv(
    "output/bonus_comparaisons_militaires.csv", index=False
)

print(table_militaire.head())
print("Nombre de comparaisons :", len(table_militaire))

# BONUS 2 : comparaison des classements de population
def comparerClassements(rangs1, rangs2):
    spearman = scipy.stats.spearmanr(rangs1, rangs2)
    kendall = scipy.stats.kendalltau(rangs1, rangs2)

    return spearman.correlation, kendall.correlation


annees_population = list(range(2007, 2027))
classements_annuels = {}

for annee in annees_population:
    colonne = "Pop " + str(annee)
    valeurs = population[["État", colonne]].dropna()
    valeurs = valeurs.sort_values(colonne, ascending=False)

    classement = {}
    for rang, etat in enumerate(valeurs["État"], start=1):
        classement[etat] = rang

    classements_annuels[annee] = classement


resultats_population = []

for i in range(len(annees_population)):
    for j in range(i + 1, len(annees_population)):
        annee1 = annees_population[i]
        annee2 = annees_population[j]

        etats_communs = set(classements_annuels[annee1]) & \
                        set(classements_annuels[annee2])

        if len(etats_communs) >= 2:
            rangs1 = [classements_annuels[annee1][e]
                      for e in etats_communs]
            rangs2 = [classements_annuels[annee2][e]
                      for e in etats_communs]

            spearman, kendall = comparerClassements(rangs1, rangs2)

            resultats_population.append([
                annee1, annee2, len(etats_communs),
                spearman, kendall
            ])


table_population = pd.DataFrame(
    resultats_population,
    columns=["Année 1", "Année 2", "États communs",
             "Spearman", "Kendall"]
)

table_population.to_csv(
    "output/bonus_comparaisons_population.csv", index=False
)

print(table_population.head())
print("Nombre de comparaisons :", len(table_population))
