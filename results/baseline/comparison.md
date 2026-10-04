# Régression logistique : données brutes et TDA

Même partition stratifiée : 180 train, 60 validation, 60 test (seed 42).
Normalisation, ATOL et classificateurs ajustés sur le train seulement.
C choisi parmi 0.001, 0.01, 0.1, 1, 10, 100 sur la validation.

| Représentation | Variables | Paramètres | C | Train | Validation | Test | F1 macro test |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 600 | 1803 | 0.001 | 100.0% | 88.3% | 91.7% | 0.915 |
| tda_atol | 40 | 123 | 1 | 100.0% | 100.0% | 100.0% | 1.000 |

Sur cette partition, l'écart TDA − brut est de +8.3 points de pourcentage sur le test.
Le résultat concerne l'ensemble du prétraitement TDA + ATOL,
pas une attribution isolée à l'homologie ou à la vectorisation.

## Matrice de confusion test : raw

Lignes = vraies classes ; colonnes = prédictions.

| | A | B | C |
|---|---:|---:|---:|
| A | 20 | 0 | 0 |
| B | 0 | 20 | 0 |
| C | 5 | 0 | 15 |

## Matrice de confusion test : tda_atol

Lignes = vraies classes ; colonnes = prédictions.

| | A | B | C |
|---|---:|---:|---:|
| A | 20 | 0 | 0 |
| B | 0 | 20 | 0 |
| C | 0 | 0 | 20 |

## Interprétation

Cette expérience mesure si la représentation topologique rend les classes
plus faciles à séparer avec une régression logistique. Le résultat doit être
lu tel quel, même si les données brutes obtiennent un score comparable ou supérieur.
Les entrées ont des dimensions différentes : même famille de modèles, mais
pas le même nombre de paramètres. ATOL constitue lui-même un prétraitement appris.
Une partition de 60 exemples test ne démontre pas une supériorité générale.
Les sessions d'acquisition ne sont pas identifiées ; la dépendance entre
fenêtres ne peut donc pas être exclue par cette séparation par échantillon.
Les performances de validation servent à choisir C ; elles ne sont pas
une estimation indépendante des performances finales.
