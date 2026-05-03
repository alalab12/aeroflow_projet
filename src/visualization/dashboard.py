"""
Module de visualisation - Dashboard de synthèse en fin de session
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from datetime import datetime
import config


class FlowDashboard:
    """
    Tableau de bord de synthèse des flux.

    - Graph du haut : flux total + moyenne + médiane + prédiction
    - Zone du bas : tableau (heure, caméra, moyenne 10s)
    - Panneau info à droite avec stats globales et flux A<->B estimés
    """

    def __init__(self):
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def _setup_plot(self):
        """Configure le graphique matplotlib (appelé seulement à la fin)."""
        self.fig = plt.figure(figsize=(config.GRAPH_WIDTH, config.GRAPH_HEIGHT))
        self.fig.suptitle(
            'AeroFlow - Dashboard de synthèse de session',
            fontsize=16,
            fontweight='bold'
        )

        # 2 lignes : courbe en haut, tableau en bas
        self.ax1 = plt.subplot(2, 1, 1)
        self.ax2 = plt.subplot(2, 1, 2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def show_summary(
        self,
        history: List[int],
        prediction: Optional[float],
        current_a: int,
        current_b: int,
        trend: str,
        session_records: List[Dict[str, Any]]
    ):
        """
        Affiche le dashboard UNE SEULE FOIS en fin de session,
        avec tout l'historique disponible.
        """
        if not history or not session_records:
            print("[INFO] Aucun historique, dashboard non affiché.")
            return

        plt.ioff()  # on veut un plt.show() bloquant

        self._setup_plot()
        self._draw(history, prediction, current_a, current_b, trend, session_records)

        print(
            "[INFO] Dashboard de synthèse affiché. "
            "Fermez la fenêtre pour terminer."
        )
        plt.show()

    def _draw(
        self,
        history: List[int],
        prediction: Optional[float],
        current_a: int,
        current_b: int,
        trend: str,
        session_records: List[Dict[str, Any]]
    ):
        """Logique de dessin, appelée uniquement par show_summary()."""

        try:
            self.ax1.clear()
            self.ax2.clear()

            history_arr = np.array(history, dtype=float)
            n = len(history_arr)

            # ---------------------------
            # 1) Graph du haut : courbe + moyenne + médiane + prédiction
            # ---------------------------
            x = np.arange(n)

            self.ax1.plot(
                x, history_arr,
                'b-',
                linewidth=2,
                label='Flux total observé'
            )
            self.ax1.scatter(
                x,
                history_arr,
                s=40,
                c='blue',
                zorder=5,
                alpha=0.7
            )

            mean_val = float(np.mean(history_arr))
            median_val = float(np.median(history_arr))

            self.ax1.axhline(
                mean_val,
                color='green',
                linestyle='--',
                linewidth=2,
                label=f'Moyenne ({mean_val:.1f})'
            )
            self.ax1.axhline(
                median_val,
                color='orange',
                linestyle=':',
                linewidth=2,
                label=f'Médiane ({median_val:.1f})'
            )

            if prediction is not None and n >= 3:
                steps_ahead = config.PREDICTION_HORIZON / config.SAMPLING_INTERVAL
                pred_x = n - 1 + steps_ahead

                self.ax1.plot(
                    [n - 1, pred_x],
                    [history_arr[-1], prediction],
                    'r--',
                    linewidth=2,
                    label=f'Prédiction +{config.PREDICTION_HORIZON}s',
                    alpha=0.8
                )

                self.ax1.scatter(
                    [pred_x],
                    [prediction],
                    s=120,
                    c='red',
                    marker='*',
                    zorder=10,
                    edgecolors='darkred',
                    linewidth=1.5
                )

                recent_window = history_arr[-min(5, n):]
                recent_std = max(np.std(recent_window), 0.5)

                self.ax1.fill_between(
                    [n - 1, pred_x],
                    [history_arr[-1] - recent_std, prediction - recent_std],
                    [history_arr[-1] + recent_std, prediction + recent_std],
                    alpha=0.2,
                    color='red',
                    label='Zone de confiance'
                )

            self.ax1.set_xlabel(
                f'Mesures (tous les {config.SAMPLING_INTERVAL}s)',
                fontsize=11,
                fontweight='bold'
            )
            self.ax1.set_ylabel(
                'Nombre de personnes',
                fontsize=11,
                fontweight='bold'
            )
            self.ax1.set_title(
                'Historique du flux total (moyenne, médiane et prédiction)',
                fontsize=13,
                pad=10
            )

            handles, _ = self.ax1.get_legend_handles_labels()
            if handles:
                self.ax1.legend(loc='upper left', fontsize=9)

            self.ax1.grid(True, alpha=0.3, linestyle='--')

            y_max = max(
                max(history_arr),
                prediction if prediction is not None else 0
            ) * 1.2
            self.ax1.set_ylim(0, max(y_max, 5))

            # ---------------------------
            # 2) Zone du bas : tableau structuré (10 s)
            # ---------------------------
            self._draw_table(session_records)

            # ---------------------------
            # 3) Panneau d'information (stats globales + flux)
            # ---------------------------
            info_text = self._build_info_panel(
                history_arr,
                mean_val,
                median_val,
                trend,
                prediction,
                current_a,
                current_b,
                session_records
            )

            self.fig.text(
                0.99,
                0.99,
                info_text,
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.8',
                    facecolor='whitesmoke',
                    alpha=0.9,
                    edgecolor='black',
                    linewidth=1.0
                ),
                verticalalignment='top',
                horizontalalignment='right',
                family='monospace'
            )

            plt.tight_layout(rect=[0, 0.01, 1, 0.96])

        except Exception as e:
            print(f"Erreur dessin dashboard: {e}")

    def _draw_table(self, session_records: List[Dict[str, Any]]):
        """
        Dessine un tableau dans ax2 :
        Heure | Caméra | Moyenne 10s
        (A, B et Total pour chaque tranche de 10s)
        """
        self.ax2.clear()
        self.ax2.axis('off')

        try:
            times = [
                datetime.fromisoformat(rec["timestamp"])
                for rec in session_records
            ]
            locals_ = [rec.get("local_count", 0) for rec in session_records]
            remotes = [rec.get("remote_count", 0) for rec in session_records]
            totals = [rec.get("total_count", 0) for rec in session_records]

            if not times:
                self.ax2.text(
                    0.5,
                    0.5,
                    "Pas de données pour le tableau 10s",
                    ha='center',
                    va='center',
                    fontsize=11
                )
                return

            rows = []
            start_time = times[0]
            bucket_loc = []
            bucket_rem = []
            bucket_tot = []
            bucket_start = start_time

            for t, loc, rem, tot in zip(times, locals_, remotes, totals):
                # nouvelle tranche de 10s
                if (t - bucket_start).total_seconds() >= 10 and bucket_tot:
                    ts_str = bucket_start.strftime("%H:%M:%S")
                    rows.extend(self._summarize_bucket(ts_str, bucket_loc,
                                                       bucket_rem, bucket_tot))
                    bucket_loc = []
                    bucket_rem = []
                    bucket_tot = []
                    bucket_start = t

                bucket_loc.append(loc)
                bucket_rem.append(rem)
                bucket_tot.append(tot)

            # Dernière tranche
            if bucket_tot:
                ts_str = bucket_start.strftime("%H:%M:%S")
                rows.extend(self._summarize_bucket(ts_str, bucket_loc,
                                                   bucket_rem, bucket_tot))

            if not rows:
                self.ax2.text(
                    0.5,
                    0.5,
                    "Pas assez de données pour le tableau 10s",
                    ha='center',
                    va='center',
                    fontsize=11
                )
                return

            # Construction du tableau structuré
            cell_text = []
            for ts_str, cam_name, mean_str in rows:
                cell_text.append([ts_str, cam_name, mean_str])

            table = self.ax2.table(
                cellText=cell_text,
                colLabels=["Heure", "Caméra", "Moyenne 10s"],
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.3)

        except Exception as e:
            self.ax2.text(
                0.5,
                0.5,
                f"Erreur tableau 10s: {e}",
                ha='center',
                va='center',
                fontsize=11
            )

    def _summarize_bucket(
        self,
        ts_str: str,
        bucket_loc: List[int],
        bucket_rem: List[int],
        bucket_tot: List[int]
    ):
        """
        Construit les lignes (A, B, Total) pour une tranche de 10s.
        Retourne une liste de tuples (heure, camera, moyenne_str).
        """
        rows = []

        if bucket_loc:
            mean_loc = float(np.mean(bucket_loc))
            rows.append((ts_str, "A", f"{mean_loc:5.1f}"))
        if bucket_rem and max(bucket_rem) > 0:  # n'afficher B que s'il a vu qqch
            mean_rem = float(np.mean(bucket_rem))
            rows.append((ts_str, "B", f"{mean_rem:5.1f}"))

        if bucket_tot:
            mean_tot = float(np.mean(bucket_tot))
            rows.append((ts_str, "Total", f"{mean_tot:5.1f}"))

        return rows

    def _estimate_flux_ab(
        self,
        session_records: List[Dict[str, Any]],
        delay_steps: int = 1  # 1 pas * 3s ≈ 2,5s de marche
    ) -> tuple[float, float]:
        """
        Estime les flux A->B et B->A de manière simple, à partir des séries
        local_count (cam A) et remote_count (cam B) avec un décalage temporel.

        On ne compte un flux que si la caméra source et la caméra cible
        voient toutes les deux au moins une personne (valeur > 0).
        """
        local_series = [rec.get("local_count", 0) for rec in session_records]
        remote_series = [rec.get("remote_count", 0) for rec in session_records]

        n = min(len(local_series), len(remote_series))
        if n <= delay_steps:
            return 0.0, 0.0

        flux_a_to_b = 0.0
        flux_b_to_a = 0.0
        count_pairs_ab = 0
        count_pairs_ba = 0

        for i in range(n - delay_steps):
            a_now = local_series[i]
            b_future = remote_series[i + delay_steps]

            b_now = remote_series[i]
            a_future = local_series[i + delay_steps]

            # A -> B : il faut des gens sur A maintenant ET sur B plus tard
            if a_now > 0 and b_future > 0:
                delta_b = b_future - b_now
                if delta_b > 0:
                    flux_a_to_b += delta_b
                count_pairs_ab += 1

            # B -> A : il faut des gens sur B maintenant ET sur A plus tard
            if b_now > 0 and a_future > 0:
                delta_a = a_future - a_now
                if delta_a > 0:
                    flux_b_to_a += delta_a
                count_pairs_ba += 1

        mean_flux_a_to_b = (
            flux_a_to_b / count_pairs_ab if count_pairs_ab > 0 else 0.0
        )
        mean_flux_b_to_a = (
            flux_b_to_a / count_pairs_ba if count_pairs_ba > 0 else 0.0
        )

        return mean_flux_a_to_b, mean_flux_b_to_a

    def _build_info_panel(
        self,
        history: np.ndarray,
        mean_val: float,
        median_val: float,
        trend: str,
        prediction: Optional[float],
        current_a: int,
        current_b: int,
        session_records: List[Dict[str, Any]]
    ) -> str:
        """
        Construit le texte du panneau info (stats globales + flux estimés).
        """
        total_final = current_a + current_b
        std_val = float(np.std(history))

        info = ""
        info += "STATISTIQUES SESSION\n"
        info += "-" * 32 + "\n"
        info += f"Moyenne: {mean_val:5.2f}\n"
        info += f"Médiane: {median_val:5.2f}\n"
        info += f"Écart-type: {std_val:5.2f}\n"
        info += f"Tendance finale: {trend.upper()}\n"

        if prediction is not None:
            info += "-" * 32 + "\n"
            info += f"Prédiction totale +30s: {prediction:5.1f}\n"
            delta = prediction - total_final
            info += f"Delta vs actuel: {delta:+5.1f}\n"

        # Flux estimés A->B / B->A
        flux_a_to_b, flux_b_to_a = self._estimate_flux_ab(
            session_records,
            delay_steps=1  # 1 * 3s ≈ 2,5s
        )

        info += "-" * 32 + "\n"
        info += f"Flux A -> B (moyen): {flux_a_to_b:5.2f}\n"
        info += f"Flux B -> A (moyen): {flux_b_to_a:5.2f}\n"

        return info

    def close(self):
        """Ferme le dashboard"""
        try:
            plt.close(self.fig)
        except Exception:
            pass


class FrameAnnotator:
    """
    Ajoute des annotations sur les frames vidéo
    """

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        count: int,
        camera_name: str,
        mode: str,
        prediction: Optional[float] = None  # gardé pour compat, mais ignoré
    ) -> np.ndarray:
        """Ajoute des informations sur la frame"""
        if frame is None:
            return None

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Fond semi-transparent
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Texte principal (sans accent sur "Camera")
        text1 = f"Camera {camera_name} ({mode})"
        text2 = f"Personnes: {count}"

        cv2.putText(
            annotated,
            text1,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            annotated,
            text2,
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # On n'affiche plus la prédiction

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            annotated,
            timestamp,
            (w - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )

        return annotated