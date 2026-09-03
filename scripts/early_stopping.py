"""Détection de plateau pour l'arrêt anticipé, basée sur la pente d'une régression locale."""
import numpy as np


def detect_plateau(acc_history, window=5, slope_threshold=1e-4):
    """Détecte un plateau via la pente d'une régression linéaire locale sur l'accuracy EMA.
    Préféré à un compteur de patience car la pente s'adapte à l'échelle locale du bruit,
    plutôt que de dépendre d'un seuil absolu sur l'accuracy (cf. justification papier :
    argument tiré de la décroissance du schedule cosine, qui porte sur un TAUX de variation).
    """
    if len(acc_history) < window:
        return False, None
    recent = np.array(acc_history[-window:])
    x = np.arange(window)
    slope = np.polyfit(x, recent, 1)[0]
    return abs(slope) < slope_threshold, slope
