"""
Module de gestion des flux vidéo (webcam ou fichier)
"""

import cv2
import numpy as np
from typing import Optional, Union
import config


class CameraStream:
    """
    Gère la capture vidéo depuis une webcam OU un fichier vidéo.

    - source: int  -> index de webcam (0, 1, ...)
    - source: str  -> chemin vers un fichier vidéo (ex: 'video_test.mp4')
    """

    def __init__(self, source: Union[int, str]):
        self.source = source
        self.cap = None
        self.is_active = False

    def start(self) -> bool:
        """Démarre la capture vidéo."""
        try:
            self.cap = cv2.VideoCapture(self.source)

            # Si c'est une webcam (int), on force la résolution et le FPS
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS,          config.FPS)

            if not self.cap.isOpened():
                print(f"[ERREUR] Impossible d'ouvrir la source vidéo: {self.source}")
                return False

            self.is_active = True
            print(f"[INFO] Source vidéo démarrée: {self.source}")
            return True

        except Exception as e:
            print(f"[ERREUR] Lors du démarrage de la source vidéo: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retourne une frame ou None en cas de fin de vidéo / erreur.
        """
        if not self.is_active or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Fin de fichier si c'est une vidéo
            return None

        return frame

    def stop(self):
        """Arrête la capture et libère les ressources."""
        if self.cap is not None:
            self.cap.release()
        self.is_active = False

    def __del__(self):
        self.stop()