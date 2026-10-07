#coding:utf8

import pandas as pd
import math
import scipy
from scipy.stats import shapiro

# C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
# Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
# Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

# Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#charger le fichier csv
echantillons = ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv")

#afficher les premières lignes
print(echantillons.head())

#afficher les noms des colonnes
print(echantillons.columns)


# Question 1 : Théorie de l'échantillonnage (intervalles de fluctuation)
# L'échantillonnage se base sur la répétitivité.
print("Question 1")
print("Résultat sur le calcul d'un intervalle de fluctuation")

#caclucerla moyenen de chaque colonne
moyennes = echantillons.mean()

#arrondir les moyennes à l'entier
moyennes = moyennes.round(0)

print("Moyennes :")
print(moyennes)

#total des moyennes
total_moyennes = moyennes.sum()

#fréquences observées dans les échantillons
frequences = moyennes / total_moyennes

#arrondir à deux décimales
fréquences = frequences = frequences.round(2)

#fréqeunces de la population mère
population = pd.Series({
    "Pour": 852,
    "Contre": 911,
    "Sans opinion": 422
})

frequences_population = (population / 2185).round(2)

print("Fréquences des échantillons :")
print(fréquences)

print("Fréquences de la population :")
print(frequences_population)

# Taille moyenne d'un échantillon
tailles = echantillons.sum(axis=1)
n = tailles.mean()

# Calcul des intervalles de fluctuation
for opinion in population.index:
    p = population[opinion] / 2185
    marge = 1.96 * math.sqrt(p * (1 - p) / n)

    borne_inf = p - marge
    borne_sup = p + marge

    print(opinion)
    print("Intervalle :", round(borne_inf, 3),
          "-", round(borne_sup, 3))
    

# Question 2 : Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Question 2")
print("Résultat sur le calcul d'un intervalle de confiance")

# Récupérer le premier échantillon
premier_echantillon = echantillons.iloc[0]

# Convertir en liste Python
donnees = list(premier_echantillon)

print(donnees)

# Calculer l'effectif total
total = sum(donnees)

# Calculer les fréquences
frequences_echantillon = [
    valeur / total for valeur in donnees
]

print("Effectif total :", total)
print("Fréquences :", frequences_echantillon)

# Calculer les intervalles de confiance
for i, frequence in enumerate(frequences_echantillon):
    marge = 1.96 * math.sqrt(
    frequence * (1 - frequence) / total
)

    borne_inf = frequence - marge
    borne_sup = frequence + marge

    print("Opinion :", echantillons.columns[i])
    print("Fréquence :", round(frequence, 3))
    print("IC à 95 % :", round(borne_inf, 3),
          "-", round(borne_sup, 3))

# Question 3 : Théorie de la décision (tests d'hypothèse)
# La décision se base sur la notion de risques alpha et bêta.
# Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Question 3")
print("Théorie de la décision")


# Charger les deux distributions
loi1 = ouvrirUnFichier("data/Loi-normale-Test-1.csv")
loi2 = ouvrirUnFichier("data/Loi-normale-Test-2.csv")

print(loi1.head())
print(loi2.head())

# Transformer les données en listes
valeurs1 = loi1.iloc[:, 0].dropna().tolist()
valeurs2 = loi2.iloc[:, 0].dropna().tolist()

print(valeurs1[:10])
print(valeurs2[:10])

# Réaliser les tests de Shapiro-Wilk
test1 = shapiro(valeurs1)
test2 = shapiro(valeurs2)

print("Test 1")
print("Statistique :", test1.statistic)
print("P-value :", test1.pvalue)

print("Test 2")
print("Statistique :", test2.statistic)
print("P-value :", test2.pvalue)

# Question bonus
print("La Loi 2 suit une loi géométrique.")
