"""
Module de visualisation - Dashboard de synthèse en fin de session
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from typing import List, Optional, Dict, Any
from datetime import datetime
import config
import tkinter as tk
from tkinter import ttk


class FlowDashboard:
    """
    Tableau de bord de synthèse des flux avec scrollbar.

    - Graph du haut : flux total + moyenne + médiane + prédiction
    - Zone du bas : tableau (heure, caméra, mesure 3s) avec scroll
    - Panneau info à droite avec stats globales et flux A<->B estimés
    """

    def __init__(self):
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.root = None

    def _setup_plot(self):
        """Configure le graphique matplotlib avec hauteur adaptée."""
        fig_height = max(config.GRAPH_HEIGHT * 2.5, 18)  # Au moins 18 pouces

        self.fig = plt.figure(figsize=(config.GRAPH_WIDTH, fig_height))
        self.fig.suptitle(
            "AeroFlow - Dashboard de synthèse de session",
            fontsize=16,
            fontweight="bold",
        )

        # 2 sous-graphiques : graphique 30%, tableau 70%
        self.ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=3)  # Graphique
        self.ax2 = plt.subplot2grid((10, 1), (3, 0), rowspan=7)  # Tableau

        plt.tight_layout(rect=[0, 0.01, 0.85, 0.97])  # Laisser place à droite pour stats

    def show_summary(
        self,
        history: List[int],
        prediction_total: Optional[float],
        prediction_a: Optional[float],
        prediction_b: Optional[float],
        current_a: int,
        current_b: int,
        trend: str,
        trend_a: str,
        trend_b: str,
        session_records: List[Dict[str, Any]],
    ):
        """
        Affiche le dashboard UNE SEULE FOIS en fin de session,
        avec scrollbar pour naviguer dans les données.
        """
        if not history or not session_records:
            print("[INFO] Aucun historique, dashboard non affiché.")
            return

        self._setup_plot()
        self._draw(
            history,
            prediction_total,
            prediction_a,
            prediction_b,
            current_a,
            current_b,
            trend,
            trend_a,
            trend_b,
            session_records,
        )

        self._create_scrollable_window()

    def _create_scrollable_window(self):
        """Crée une fenêtre Tkinter avec canvas scrollable."""
        self.root = tk.Tk()
        self.root.title("AeroFlow - Dashboard de synthèse")
        self.root.state("zoomed")  # Fenêtre maximisée

        # Gérer la fermeture proprement
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Graph Matplotlib dans le frame scrollable
        canvas_widget = FigureCanvasTkAgg(self.fig, master=scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas_widget, scrollable_frame)
        toolbar.update()

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Scroll molette
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        print("[INFO] Dashboard affiché. Fermez la fenêtre pour terminer.")
        self.root.mainloop()

    def _on_closing(self):
        """Callback quand l'utilisateur ferme la fenêtre."""
        print("[INFO] Fermeture du dashboard...")
        try:
            plt.close(self.fig)
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur fermeture: {e}")

    def _draw(
        self,
        history: List[int],
        prediction_total: Optional[float],
        prediction_a: Optional[float],
        prediction_b: Optional[float],
        current_a: int,
        current_b: int,
        trend: str,
        trend_a: str,
        trend_b: str,
        session_records: List[Dict[str, Any]],
    ):
        """Logique de dessin."""

        try:
            self.ax1.clear()
            self.ax2.clear()

            history_arr = np.array(history, dtype=float)
            n = len(history_arr)

            # ---------------------------
            # 1) Graph du haut
            # ---------------------------
            x = np.arange(n)

            self.ax1.plot(
                x,
                history_arr,
                "b-",
                linewidth=2,
                label="Flux total observé",
            )
            self.ax1.scatter(
                x,
                history_arr,
                s=40,
                c="blue",
                zorder=5,
                alpha=0.7,
            )

            mean_val = float(np.mean(history_arr))
            median_val = float(np.median(history_arr))

            self.ax1.axhline(
                mean_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Moyenne ({mean_val:.1f})",
            )
            self.ax1.axhline(
                median_val,
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Médiane ({median_val:.1f})",
            )

            if prediction_total is not None and n >= 3:
                steps_ahead = config.PREDICTION_HORIZON / config.SAMPLING_INTERVAL
                pred_x = n - 1 + steps_ahead

                self.ax1.plot(
                    [n - 1, pred_x],
                    [history_arr[-1], prediction_total],
                    "r--",
                    linewidth=2,
                    label=f"Prédiction +{config.PREDICTION_HORIZON}s",
                    alpha=0.8,
                )

                self.ax1.scatter(
                    [pred_x],
                    [prediction_total],
                    s=120,
                    c="red",
                    marker="*",
                    zorder=10,
                    edgecolors="darkred",
                    linewidth=1.5,
                )

                recent_window = history_arr[-min(5, n) :]
                recent_std = max(np.std(recent_window), 0.5)

                self.ax1.fill_between(
                    [n - 1, pred_x],
                    [history_arr[-1] - recent_std, prediction_total - recent_std],
                    [history_arr[-1] + recent_std, prediction_total + recent_std],
                    alpha=0.2,
                    color="red",
                    label="Zone de confiance",
                )

            self.ax1.set_xlabel(
                f"Mesures (tous les {config.SAMPLING_INTERVAL}s)",
                fontsize=11,
                fontweight="bold",
            )
            self.ax1.set_ylabel(
                "Nombre de personnes",
                fontsize=11,
                fontweight="bold",
            )
            self.ax1.set_title(
                "Historique du flux total (moyenne, médiane et prédiction)",
                fontsize=13,
                pad=10,
            )

            handles, _ = self.ax1.get_legend_handles_labels()
            if handles:
                self.ax1.legend(loc="upper left", fontsize=9)

            self.ax1.grid(True, alpha=0.3, linestyle="--")

            y_max = max(
                max(history_arr),
                prediction_total if prediction_total is not None else 0,
            ) * 1.2
            self.ax1.set_ylim(0, max(y_max, 5))

            # ---------------------------
            # 2) Tableau du bas (limité à 40 lignes visibles)
            # ---------------------------
            self._draw_table(session_records)

            # ---------------------------
            # 3) Panneau d'information
            # ---------------------------
            info_text = self._build_info_panel(
                history_arr,
                mean_val,
                median_val,
                trend,
                trend_a,
                trend_b,
                prediction_total,
                prediction_a,
                prediction_b,
                session_records,
            )

            self.fig.text(
                0.99,
                0.99,
                info_text,
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="whitesmoke",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=1.0,
                ),
                verticalalignment="top",
                horizontalalignment="right",
                family="monospace",
            )

            plt.tight_layout(rect=[0, 0.01, 0.85, 0.97])

        except Exception as e:
            print(f"Erreur dessin dashboard: {e}")
            import traceback

            traceback.print_exc()

    def _draw_table(self, session_records: List[Dict[str, Any]]):
        """
        Dessine un tableau dans ax2.
        On limite le nombre de lignes visibles (40) pour ne pas empiéter sur le graphe.
        Le scroll Tkinter permet quand même de tout voir (figure entière).
        """
        self.ax2.clear()
        self.ax2.axis("off")

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
                    "Pas de données pour le tableau",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                return

            # Toutes les lignes possibles
            full_cell_text = []
            for t, loc, rem, tot in zip(times, locals_, remotes, totals):
                ts_str = t.strftime("%H:%M:%S")
                # Caméra A
                full_cell_text.append([ts_str, "A", f"{loc:3.1f}"])
                # Caméra B (toujours affichée, même à 0.0)
                full_cell_text.append([ts_str, "B", f"{rem:3.1f}"])
                # Total
                full_cell_text.append([ts_str, "Total", f"{tot:3.1f}"])

            if not full_cell_text:
                self.ax2.text(
                    0.5,
                    0.5,
                    "Pas de données",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                return

            # Nombre max de lignes visibles
            MAX_VISIBLE_ROWS = 40
            cell_text = full_cell_text[-MAX_VISIBLE_ROWS:]

            table = self.ax2.table(
                cellText=cell_text,
                colLabels=["Heure", "Caméra", "Mesure 3s"],
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)

        except Exception as e:
            self.ax2.text(
                0.5,
                0.5,
                f"Erreur tableau: {e}",
                ha="center",
                va="center",
                fontsize=11,
            )

    def _estimate_flux_ab(
        self,
        session_records: List[Dict[str, Any]],
        delay_steps: int = 2,
        threshold: float = 0.2,
    ) -> tuple[float, float]:
        """Estime les flux A->B et B->A avec délai temporel et seuil adaptatif."""
        local_series = np.array(
            [rec.get("local_count", 0) for rec in session_records], dtype=float
        )
        remote_series = np.array(
            [rec.get("remote_count", 0) for rec in session_records], dtype=float
        )

        n = len(local_series)
        if n <= delay_steps + 1:
            return 0.0, 0.0

        flux_a_to_b_total = 0.0
        flux_b_to_a_total = 0.0
        count_a_to_b = 0
        count_b_to_a = 0

        for i in range(n - delay_steps - 1):
            a_now = local_series[i]
            b_now = remote_series[i]

            a_next = local_series[i + 1]
            b_next = remote_series[i + 1]

            a_delayed = local_series[i + delay_steps]
            b_delayed = remote_series[i + delay_steps]

            # --- Flux A -> B ---
            delta_a_immediate = a_now - a_next
            delta_b_delayed = b_delayed - b_now

            a_variation_significant = (
                delta_a_immediate >= threshold
                or (a_now > 0.1 and delta_a_immediate / a_now >= 0.3)
            )

            b_increase_significant = (
                delta_b_delayed >= threshold or (b_now >= 0 and delta_b_delayed > 0.1)
            )

            if a_variation_significant and b_increase_significant and a_now > 0:
                transferred = min(delta_a_immediate, delta_b_delayed)
                if transferred > 0:
                    flux_a_to_b_total += transferred
                    count_a_to_b += 1

            # --- Flux B -> A ---
            delta_b_immediate = b_now - b_next
            delta_a_delayed = a_delayed - a_now

            b_variation_significant = (
                delta_b_immediate >= threshold
                or (b_now > 0.1 and delta_b_immediate / b_now >= 0.3)
            )

            a_increase_significant = (
                delta_a_delayed >= threshold or (a_now >= 0 and delta_a_delayed > 0.1)
            )

            if b_variation_significant and a_increase_significant and b_now > 0:
                transferred = min(delta_b_immediate, delta_a_delayed)
                if transferred > 0:
                    flux_b_to_a_total += transferred
                    count_b_to_a += 1

        mean_flux_a_to_b = (
            flux_a_to_b_total / count_a_to_b if count_a_to_b > 0 else 0.0
        )
        mean_flux_b_to_a = (
            flux_b_to_a_total / count_b_to_a if count_b_to_a > 0 else 0.0
        )

        return mean_flux_a_to_b, mean_flux_b_to_a

    def _build_info_panel(
        self,
        history: np.ndarray,
        mean_val: float,
        median_val: float,
        trend: str,
        trend_a: str,
        trend_b: str,
        prediction_total: Optional[float],
        prediction_a: Optional[float],
        prediction_b: Optional[float],
        session_records: List[Dict[str, Any]],
    ) -> str:
        """Construit le texte du panneau info avec prédictions par caméra."""
        std_val = float(np.std(history))

        info = ""
        info += "STATISTIQUES SESSION\n"
        info += "-" * 32 + "\n"
        info += f"Moyenne: {mean_val:5.2f}\n"
        info += f"Médiane: {median_val:5.2f}\n"
        info += f"Écart-type: {std_val:5.2f}\n"
        info += f"Tendance: {trend.upper()}\n"

        info += "-" * 32 + "\n"
        info += "PRÉDICTIONS +30s\n"
        info += "-" * 32 + "\n"

        if prediction_total is not None and len(history) > 0:
            info += f"Total: {prediction_total:5.1f}\n"
            delta_total = prediction_total - history[-1]
            info += f"  Delta: {delta_total:+5.1f}\n"

        if prediction_a is not None:
            info += f"Caméra A: {prediction_a:5.1f}\n"
            info += f"  Tendance: {trend_a.upper()}\n"

        if prediction_b is not None:
            info += f"Caméra B: {prediction_b:5.1f}\n"
            info += f"  Tendance: {trend_b.upper()}\n"

        flux_a_to_b, flux_b_to_a = self._estimate_flux_ab(
            session_records, delay_steps=2, threshold=0.2
        )

        info += "-" * 32 + "\n"
        info += "FLUX DIRECTIONNELS\n"
        info += "-" * 32 + "\n"
        info += f"A -> B (moyen): {flux_a_to_b:5.2f}\n"
        info += f"B -> A (moyen): {flux_b_to_a:5.2f}\n"

        return info

    def close(self):
        """Ferme le dashboard."""
        try:
            if self.root:
                self.root.destroy()
            plt.close(self.fig)
        except Exception:
            pass


class FrameAnnotator:
    """Ajoute des annotations sur les frames vidéo"""

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        count: int,
        camera_name: str,
        mode: str,
        prediction: Optional[float] = None,
    ) -> np.ndarray:
        """Ajoute des informations sur la frame"""
        if frame is None:
            return None

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        text1 = f"Camera {camera_name} ({mode})"
        text2 = f"Personnes: {count}"

        cv2.putText(
            annotated,
            text1,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            text2,
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            annotated,
            timestamp,
            (w - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        return annotated
