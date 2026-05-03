"""
Module de prédiction de flux à 30 secondes
Utilise une régression linéaire sur l'historique récent
avec lissage léger et intervalle de confiance.
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque
import config


class FlowPredictor:
    """
    Prédit le flux de personnes à 30 secondes.
    """

    def __init__(self, history_size: int = None):
        self.history_size = history_size or config.HISTORY_SIZE
        self.history = deque(maxlen=self.history_size)

        # Paramètres du modèle
        self.window_size = getattr(config, "PRED_WINDOW_SIZE", 8)  # nb de points récents
        self.smoothing_alpha = getattr(config, "PRED_SMOOTH_ALPHA", 0.4)  # lissage EMA

    def add_measurement(self, count: int):
        """Ajoute une mesure à l'historique."""
        self.history.append(int(count))

    # -------------------------
    # Prédiction principale
    # -------------------------
    def predict(self, method: str = "linear") -> Optional[float]:
        """
        Prédit le flux à 30 secondes.
        Retourne la prédiction ou None si pas assez de données.
        """
        if len(self.history) < 3:
            return None

        if method == "linear":
            pred, _ = self._linear_regression_with_ci()
            return pred
        elif method == "moving_average":
            return self._moving_average()
        else:
            pred, _ = self._linear_regression_with_ci()
            return pred

    def predict_with_ci(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Renvoie (prediction, ecart_type_residus) pour pouvoir tracer
        un intervalle de confiance dans le dashboard.
        """
        if len(self.history) < 3:
            return None, None

        return self._linear_regression_with_ci()

    # -------------------------
    # Modèles internes
    # -------------------------
    def _get_recent_smoothed(self) -> np.ndarray:
        """
        Retourne un vecteur numpy avec les derniers points (fenêtre)
        après lissage exponentiel (EMA).
        """
        data = np.array(self.history, dtype=float)
        if len(data) <= 1:
            return data

        # Fenêtre récente
        window_size = min(self.window_size, len(data))
        recent = data[-window_size:]

        # Lissage EMA simple
        alpha = self.smoothing_alpha
        smoothed = [recent[0]]
        for v in recent[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

        return np.array(smoothed, dtype=float)

    def _linear_regression_with_ci(self) -> Tuple[float, float]:
        """
        Régression linéaire sur les données lissées de la fenêtre récente.
        Extrapole à 30 secondes et renvoie (prediction, std_residus).
        """
        y = self._get_recent_smoothed()
        n = len(y)
        x = np.arange(n)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            prediction = float(y_mean)
            return max(0.0, prediction), 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        steps_ahead = config.PREDICTION_HORIZON / config.SAMPLING_INTERVAL
        next_x = n - 1 + steps_ahead  # extrapole à partir du dernier point
        prediction = slope * next_x + intercept

        # Calcul des résidus pour estimer la variabilité
        y_hat = slope * x + intercept
        residuals = y - y_hat
        if n > 2:
            std_res = float(np.sqrt(np.sum(residuals**2) / (n - 2)))
        else:
            std_res = 0.0

        return max(0.0, prediction), std_res

    def _moving_average(self) -> float:
        """Moyenne mobile simple sur tout l'historique."""
        return float(np.mean(self.history))

    # -------------------------
    # Tendance
    # -------------------------
    def get_trend(self) -> str:
        """
        Analyse la tendance.
        Retourne "hausse", "baisse", ou "stable".
        Basé sur l'évolution des 3 derniers points lissés.
        """
        if len(self.history) < 3:
            return "stable"

        data = np.array(self.history, dtype=float)
        recent = data[-3:]

        if recent[-1] > recent[0] * 1.1:
            return "hausse"
        elif recent[-1] < recent[0] * 0.9:
            return "baisse"
        else:
            return "stable"

    def get_history(self) -> List[int]:
        """Retourne l'historique complet."""
        return list(self.history)