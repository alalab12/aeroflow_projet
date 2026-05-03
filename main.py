"""
AeroFlow - Version ultra-optimisée pour fluidité maximale
"""

import argparse
import time
import cv2
import signal
import os
import json
import datetime
import config

from src.acquisition.camera_stream import CameraStream
from src.acquisition.network_comm import NetworkServer, NetworkClient
from src.processing.tracker import PeopleTracker
from src.prediction.model import FlowPredictor
from src.visualization.dashboard import FlowDashboard, FrameAnnotator


class AeroFlowApp:
    """Application principale AeroFlow"""

    def __init__(self, mode: str, camera_source, master_ip: str = None):
        """
        mode: 'master' ou 'slave'
        camera_source: int (index webcam) OU str (chemin fichier vidéo)
        """
        self.mode = mode
        self.camera_source = camera_source
        self.master_ip = master_ip

        self.camera = CameraStream(camera_source)
        self.tracker = PeopleTracker(use_yolo=True)

        if mode == "master":
            self.predictor = FlowPredictor()
            self.dashboard = FlowDashboard()
            self.server = NetworkServer(callback=self.on_remote_count)
            self.remote_count = 0
        else:
            self.client = NetworkClient(master_ip)

        self.is_running = False
        self.session_start_time = time.time()
        self.session_start_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.session_records = []

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("\nArret demande...")
        self.is_running = False

    def on_remote_count(self, count: int):
        self.remote_count = count

    # ------------------------------------------------------------------
    # MODE MAITRE
    # ------------------------------------------------------------------
    def run_master(self):
        print("Demarrage en mode MAITRE (PC A)")

        if not self.camera.start():
            return

        self.server.start()
        self.is_running = True

        print("AeroFlow actif - Mode optimise")
        print(f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print("Appuyer sur 'q' pour quitter ou Ctrl+C")

        cv2.namedWindow(config.WINDOW_NAME_MASTER, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            config.WINDOW_NAME_MASTER,
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT
        )

        last_sample_time = time.time()
        frame_count = 0

        last_prediction = None
        last_trend = "stable"

        try:
            while self.is_running:
                frame = self.camera.get_frame()
                if frame is None:
                    # Fin de flux vidéo: on sort proprement de la boucle
                    print("Fin du flux vidéo (master), arrêt en cours...")
                    break

                # Détection (avec skip frames intégré dans tracker)
                count_a, bboxes = self.tracker.detect_people(frame)

                # Échantillonnage périodique
                current_time = time.time()
                if current_time - last_sample_time >= config.SAMPLING_INTERVAL:
                    total_count = count_a + self.remote_count
                    self.predictor.add_measurement(total_count)
                    last_sample_time = current_time

                    # Calculer prédiction et tendance seulement lors de l'échantillonnage
                    last_prediction = self.predictor.predict("linear")
                    last_trend = self.predictor.get_trend()

                    self.session_records.append({
                        "timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        ),
                        "local_count": int(count_a),
                        "remote_count": int(self.remote_count),
                        "total_count": int(total_count),
                        "prediction": None if last_prediction is None else float(
                            last_prediction
                        ),
                        "trend": last_trend
                    })

                # Annotation vidéo
                annotated = self.tracker.draw_detections(frame, bboxes, count_a)
                annotated = FrameAnnotator.annotate_frame(
                    annotated, count_a, "A", "Maitre", last_prediction
                )

                cv2.imshow(config.WINDOW_NAME_MASTER, annotated)

                # Petit sleep pour éviter l'effet "vidéo accélérée"
                time.sleep(1.0 / max(getattr(config, "FPS", 10), 1))

                # AUCUNE mise à jour du dashboard ici
                # (il sera affiché une seule fois en fin de session)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Arret demande...")
                    break

                frame_count += 1
                if frame_count > 10:
                    try:
                        if cv2.getWindowProperty(
                            config.WINDOW_NAME_MASTER,
                            cv2.WND_PROP_VISIBLE
                        ) < 1:
                            print("Fenetre fermee...")
                            break
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("\nInterruption Ctrl+C")
        except Exception as e:
            print(f"Erreur: {e}")
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # MODE ESCLAVE
    # ------------------------------------------------------------------
    def run_slave(self):
        print("Demarrage en mode ESCLAVE (PC B)")
        print(f"Connexion au maitre: {self.master_ip}:{config.NETWORK_PORT}")

        if not self.camera.start():
            return

        self.is_running = True

        print("Camera B active - Mode optimise")
        print(f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print("Appuyer sur 'q' pour quitter")

        cv2.namedWindow(config.WINDOW_NAME_SLAVE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            config.WINDOW_NAME_SLAVE,
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT
        )

        last_send_time = time.time()
        frame_count = 0

        try:
            while self.is_running:
                frame = self.camera.get_frame()
                if frame is None:
                    print("Fin du flux vidéo (slave), arrêt en cours...")
                    break

                count_b, bboxes = self.tracker.detect_people(frame)

                current_time = time.time()
                if current_time - last_send_time >= config.SAMPLING_INTERVAL:
                    self.client.send_count(count_b)
                    last_send_time = current_time
                    self.session_records.append({
                        "timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        ),
                        "local_count": int(count_b)
                    })

                annotated = self.tracker.draw_detections(frame, bboxes, count_b)
                annotated = FrameAnnotator.annotate_frame(
                    annotated, count_b, "B", "Esclave"
                )

                cv2.imshow(config.WINDOW_NAME_SLAVE, annotated)

               

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Arret demande...")
                    break

                frame_count += 1
                if frame_count > 10:
                    try:
                        if cv2.getWindowProperty(
                            config.WINDOW_NAME_SLAVE,
                            cv2.WND_PROP_VISIBLE
                        ) < 1:
                            print("Fenetre fermee...")
                            break
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("\nInterruption Ctrl+C")
        except Exception as e:
            print(f"Erreur: {e}")
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # SAUVEGARDE SESSION
    # ------------------------------------------------------------------
    def save_session_data(self):
        """
        Sauvegarde l'historique de la session dans un fichier JSON.
        """
        try:
            if not self.session_records:
                print("[INFO] Aucun enregistrement de session à sauvegarder.")
                return

            os.makedirs("sessions", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("sessions", f"session_{self.mode}_{ts}.json")

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.session_records, f, indent=2, ensure_ascii=False)

            print(f"[INFO] Session sauvegardée dans {filename}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur sauvegarde session: {e}")

    # ------------------------------------------------------------------
    # CLEANUP + DASHBOARD FIN DE SESSION
    # ------------------------------------------------------------------
    def cleanup(self):
        print("\nNettoyage...")
        self.is_running = False
        self.camera.stop()

        # Sauvegarder les données de session
        self.save_session_data()

        if self.mode == "master":
            try:
                # Historique des totaux pour la courbe principale
                history = self.predictor.get_history()  # ex: [12, 15, 18, ...]

                if not history or not self.session_records:
                    print("[INFO] Pas assez de données, dashboard non affiché.")
                else:
                    # Derniers états pour alimenter le panneau info
                    last = self.session_records[-1]
                    last_local = last.get("local_count", 0)
                    last_remote = last.get("remote_count", 0)
                    last_pred = last.get("prediction", None)
                    last_trend = last.get("trend", "stable")

                    print(f"[INFO] Historique total: {len(history)} points")

                    # Dashboard de synthèse, avec history + session_records
                    self.dashboard.show_summary(
                        history=history,
                        prediction=last_pred,
                        current_a=last_local,
                        current_b=last_remote,
                        trend=last_trend,
                        session_records=self.session_records
                    )

            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible d'afficher le résumé : {e}")

            self.server.stop()

        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        print("Programme termine.")


def main():
    parser = argparse.ArgumentParser(
        description="AeroFlow - Version optimisee"
    )
    parser.add_argument("--mode", choices=["master", "slave"], required=True)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la webcam (0 par défaut)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Chemin d'une vidéo à utiliser à la place de la caméra"
    )
    parser.add_argument(
        "--master-ip",
        type=str,
        default=config.MASTER_IP
    )

    args = parser.parse_args()

    if args.mode == "slave" and not args.master_ip:
        print("Erreur: mode slave necessite --master-ip")
        return

    print("=" * 60)
    print("AeroFlow - Version Ultra-Optimisee")
    print("=" * 60)

    # Choix de la source vidéo: fichier (--video) ou webcam (--camera)
    if args.video is not None:
        camera_source = args.video   # fichier vidéo
    else:
        camera_source = args.camera  # index webcam

    app = AeroFlowApp(args.mode, camera_source, args.master_ip)

    if args.mode == "master":
        app.run_master()
    else:
        app.run_slave()


if __name__ == "__main__":
    main()