"""
Configuration globale AeroFlow
"""

# -----------------------------
# MODE GÉNÉRAL / RÉSEAU
# -----------------------------
MASTER_IP = "127.0.0.1"
NETWORK_PORT = 5005

MODELS_DIR = "models"

# -----------------------------
# VIDÉO / CAMÉRA
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 10  # utilisé pour limiter la vitesse d'affichage (sleep dans main)

WINDOW_NAME_MASTER = "camera A"
WINDOW_NAME_SLAVE = "camera B"

# -----------------------------
# YOLO / DÉTECTION
# -----------------------------
# Nom du modèle YOLO Ultralytics (à adapter selon ce que tu utilises)
YOLO_MODEL = f"{MODELS_DIR}/yolo11n.pt"  # exemple

# Taille d'image d'inférence (plus petit = plus rapide, moins précis)
YOLO_IMG_SIZE = 640

# Classe COCO "person" = 0
DETECTION_CLASS = 0

# Seuil de confiance pour garder une détection
CONFIDENCE_THRESHOLD = 0.5

# Nombre de frames à sauter dans PeopleTracker.detect_people
# 1 = aucune frame sautée, 2 = une frame sur 2, 3 = une sur 3, etc.
SKIP_FRAMES = 3

# -----------------------------
# SAMPLING / PRÉDICTION
# -----------------------------
# Intervalle entre deux échantillons de comptage (en secondes)
SAMPLING_INTERVAL = 3

# Horizon de prédiction (en secondes)
PREDICTION_HORIZON = 30

# Taille de l'historique gardé par FlowPredictor
HISTORY_SIZE = 200

# Paramètres du modèle de prédiction
# Nombre de points récents pris pour la régression linéaire
PRED_WINDOW_SIZE = 8

# Coefficient de lissage exponentiel (entre 0 et 1)
PRED_SMOOTH_ALPHA = 0.4

# -----------------------------
# DASHBOARD / VISUALISATION
# -----------------------------
# Dimensions de la figure Matplotlib
GRAPH_WIDTH = 13  # en pouces
GRAPH_HEIGHT = 7  # en pouces

# -----------------------------
# LOGS / SESSIONS
# -----------------------------
SESSIONS_DIR = "sessions"

# Tu peux ajouter des options de debug si besoin
DEBUG = False