# Application de Traitement d'Images Astronomiques

Cette application est conçue pour le traitement et l'analyse d'images astronomiques, avec un focus particulier sur la déformation et l'alignement d'images.

## Fonctionnalités Principales

### 1. Interface Utilisateur
- Interface graphique moderne avec thème café
- Mode sombre/clair
- Fenêtres déplaçables
- Barre de progression pour le suivi des traitements

### 2. Méthodes de Traitement

#### Méthode Directe (`v1_direct`)
- Traitement direct des images sans étape intermédiaire
- Options configurables pour le rayon et le nombre de déformations
- Visualisation des résultats en temps réel
- Sauvegarde des images traitées au format FITS

#### Méthode Simple (`v_simple`)
- Alignement basé sur des points de contrôle
- Transformation géométrique simple
- Utile pour les cas où la déformation est minimale

#### Méthode Panchromatique (`process_fits_to_panchromatic`)
- Traitement d'images multi-bandes
- Alignement panchromatique
- Visualisation des cartes de déformation

### 3. Fonctions de Traitement d'Images

#### Chargement et Sauvegarde
- `load_fits`: Charge les images FITS avec redimensionnement automatique
- `save_image_as_fits`: Sauvegarde les images au format FITS
- `save_all_warped_images_as_fits`: Sauvegarde toutes les images déformées

#### Analyse et Traitement
- `mutual_info_matrix`: Calcule la matrice d'information mutuelle
- `find_outliers_2D`: Détecte les points aberrants dans les données 2D
- `apply_ilk`: Applique l'algorithme ILK pour la déformation d'images
- `show_deformation_map_ilk`: Visualise les cartes de déformation ILK
- `show_deformation_map_panchro`: Visualise les cartes de déformation panchromatiques

### 4. Visualisation

#### Affichage des Résultats
- `plot_results`: Visualise les résultats du traitement
- `plot_results_direct`: Visualise les résultats de la méthode directe
- `display_image_sequence`: Affiche une séquence d'images
- `create_embedded_plot`: Crée des graphiques intégrés dans l'interface

#### Contrôles de Navigation
- Navigation entre les images
- Zoom et déplacement
- Changement de mode d'affichage
- Contrôles de visualisation des cartes de déformation

### 5. Interface de Déformation

#### Contrôles de Déformation
- Rotation
- Mise à l'échelle
- Translation
- Points de contrôle ajustables
- Visualisation en temps réel des transformations

### 6. Utilitaires

#### Gestion des Fichiers
- `ouvrir_fichier`: Ouvre les fichiers d'images
- `afficher_chemins`: Affiche les chemins des fichiers
- `enregistrer_fichier`: Sauvegarde les fichiers
- `quitter_application`: Ferme l'application

#### Configuration
- `change_appearance_mode`: Change le mode d'apparence (clair/sombre)
- `change_color_theme`: Change le thème de couleur
- `update_interface`: Met à jour l'interface utilisateur

## Flux de Traitement

1. **Chargement des Images**
   - L'utilisateur sélectionne les images à traiter
   - Les images sont chargées et redimensionnées si nécessaire

2. **Configuration du Traitement**
   - Sélection de la méthode de traitement
   - Configuration des paramètres (rayon, nombre de déformations)
   - Choix des options de visualisation

3. **Exécution du Traitement**
   - Application de l'algorithme sélectionné
   - Suivi de la progression
   - Génération des cartes de déformation

4. **Visualisation des Résultats**
   - Affichage des images traitées
   - Visualisation des cartes de déformation
   - Navigation entre les résultats

5. **Sauvegarde**
   - Export des images traitées
   - Sauvegarde des paramètres de traitement
   - Génération de rapports

## Utilisation

1. Lancez l'application
2. Sélectionnez la méthode de traitement souhaitée
3. Configurez les paramètres
4. Chargez les images à traiter
5. Lancez le traitement
6. Visualisez et analysez les résultats
7. Sauvegardez les résultats si nécessaire

## Notes Techniques

- L'application utilise des algorithmes avancés pour la déformation d'images
- Le traitement est optimisé pour les images astronomiques
- Les résultats peuvent être exportés dans différents formats
- L'interface est conçue pour être intuitive et réactive
