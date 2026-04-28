import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from scipy.io import wavfile
from scipy.fft import fft, ifft


qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.audio_path = None
        self.audio_rate = None
        self.audio_data = None

        self.video_path = None
        self.video_info = None

        self.radio_fe2.setChecked(True)
        self.list_codec.setCurrentRow(0)

        self.btn_load_wav.clicked.connect(self.handle_load_audio)
        self.btn_validate_resample.clicked.connect(self.handle_resampling)
        self.btn_compress_audio.clicked.connect(self.handle_audio_compression)
        self.btn_load_video.clicked.connect(self.handle_load_video)
        self.btn_compress_video.clicked.connect(self.handle_video_compression)

    # ------------------------------------------------------------------
    # Outils d'affichage
    # ------------------------------------------------------------------
    def make_figure_for_label(self, label):
        """Crée une figure Matplotlib qui a presque le même ratio que la zone QLabel."""
        dpi = 100
        width = max(label.width(), 400)
        height = max(label.height(), 70)
        plt.ioff()
        plt.clf()
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    def save_plot_to_pixmap(self, filename):
        plt.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25)
        plt.savefig(filename, dpi=100)
        plt.close()
        return QPixmap(filename)

    def set_pixmap_in_label(self, label, pixmap):
        if pixmap.isNull():
            return

        pixmap = pixmap.scaled(
            label.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def get_mono_signal(self):
        if self.audio_data is None:
            return None

        if self.audio_data.ndim == 1:
            return self.audio_data

        return self.audio_data[:, 0]

    # ------------------------------------------------------------------
    # 4.1. Analyse du signal audio
    # ------------------------------------------------------------------
    def get_audio_info(self, file_path):
        fe, signal = wavfile.read(file_path)

        if signal.ndim == 1:
            nb_echantillons = len(signal)
            nb_canaux = 1
            type_signal = "Mono"
        else:
            nb_echantillons, nb_canaux = signal.shape
            type_signal = "Stéréo" if nb_canaux == 2 else f"{nb_canaux} canaux"

        duree = nb_echantillons / fe

        return {
            "type": type_signal,
            "frequence": fe,
            "nb_echantillons": nb_echantillons,
            "nb_canaux": nb_canaux,
            "duree": duree,
            "signal": signal
        }

    def plot_to_pixmap(self, signal, sample_rate):
        self.make_figure_for_label(self.label_signal_temporel)

        temps = np.arange(len(signal)) / sample_rate

        plt.plot(temps, signal, color="blue", linewidth=0.6)
        plt.xlabel("Temps (s)", fontsize=7)
        plt.ylabel("Amplitude", fontsize=7)
        plt.title("Signal temporel", fontsize=8)
        plt.tick_params(axis="both", labelsize=7)

        return self.save_plot_to_pixmap("temp_audio_signal.png")

    def handle_load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger le fichier .wav",
            "",
            "Fichiers WAV (*.wav)"
        )

        if not file_path:
            return

        try:
            info = self.get_audio_info(file_path)

            self.audio_path = file_path
            self.audio_rate = info["frequence"]
            self.audio_data = info["signal"]

            self.audio_features.setPlainText(
                f"Type: {info['type']}\n"
                f"Fréquence: {info['frequence']} Hz\n"
                f"Échantillons: {info['nb_echantillons']}\n"
                f"Canaux: {info['nb_canaux']}\n"
                f"Durée: {info['duree']:.2f} s"
            )

            y_canal = self.get_mono_signal()

            pixmap = self.plot_to_pixmap(y_canal, self.audio_rate)
            self.set_pixmap_in_label(self.label_signal_temporel, pixmap)

            self.label_signal_echantillonne.clear()
            self.label_spectre.clear()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur audio",
                f"Impossible de charger le fichier audio.\n{str(e)}"
            )

    # ------------------------------------------------------------------
    # 4.2. Échantillonnage du signal
    # ------------------------------------------------------------------
    def resample_signal(self, signal, factor):
        if signal is None or len(signal) == 0:
            return None

        return signal[::factor]

    def plot_comparison_to_pixmap(self, original, resampled, factor):
        """Génère un graphique comparatif : Original (bleu) vs Échantillonné (rouge)."""
        self.make_figure_for_label(self.label_signal_echantillonne)

        plt.plot(original[:10000], color='blue', linewidth=0.8, label="Original")

        indices = np.arange(0, len(resampled[:10000 // factor])) * factor
        plt.plot(
            indices,
            resampled[:10000 // factor],
            color='red',
            linewidth=0.6,
            label=f"Fe/{factor}"
        )

        plt.title("Original vs signal échantillonné", fontsize=8)
        plt.xlabel("Échantillons", fontsize=7)
        plt.ylabel("Amplitude", fontsize=7)
        plt.tick_params(axis="both", labelsize=7)
        plt.legend(fontsize=7, loc="upper right")

        return self.save_plot_to_pixmap("temp_resample.png")

    def handle_resampling(self):
        if self.audio_data is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger un fichier audio.")
            return

        if self.radio_fe2.isChecked():
            factor = 2
        elif self.radio_fe4.isChecked():
            factor = 4
        elif self.radio_fe8.isChecked():
            factor = 8
        else:
            QMessageBox.warning(self, "Attention", "Veuillez choisir Fe/2, Fe/4 ou Fe/8.")
            return

        original = self.get_mono_signal()
        resampled = self.resample_signal(original, factor)

        if resampled is None:
            QMessageBox.warning(self, "Erreur", "Aucun signal audio disponible.")
            return

        pixmap = self.plot_comparison_to_pixmap(original, resampled, factor)
        self.set_pixmap_in_label(self.label_signal_echantillonne, pixmap)

    # ------------------------------------------------------------------
    # 4.3. Compression du signal
    # ------------------------------------------------------------------
    def compress_audio_logic(self, signal, r=128):
        z = fft(signal)
        N = len(z)

        modules = np.abs(z)
        modules_tries = np.sort(modules)

        indice_seuil = int(N * (1 - 1 / r))
        seuil = modules_tries[indice_seuil]

        z_compresse = z.copy()
        z_compresse[np.abs(z_compresse) < seuil] = 0

        y_reconstruit = ifft(z_compresse).real

        return z, z_compresse, y_reconstruit, seuil

    def plot_spectrum_to_pixmap(self, spectrum):
        self.make_figure_for_label(self.label_spectre)

        modules = np.abs(spectrum)

        plt.plot(modules, color="blue", linewidth=0.8)
        plt.xlabel("Fréquences", fontsize=7)
        plt.ylabel("Module", fontsize=7)
        plt.title("Spectre du signal", fontsize=8)
        plt.tick_params(axis="both", labelsize=7)

        return self.save_plot_to_pixmap("temp_spectrum.png")

    def handle_audio_compression(self):
        if self.audio_data is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger un fichier audio.")
            return

        try:
            signal = self.get_mono_signal()
            z, z_compresse, y_reconstruit, seuil = self.compress_audio_logic(signal, r=128)

            pixmap = self.plot_spectrum_to_pixmap(z_compresse)
            self.set_pixmap_in_label(self.label_spectre, pixmap)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur compression audio",
                f"Impossible de compresser le signal audio.\n{str(e)}"
            )

    # ------------------------------------------------------------------
    # 4.4. Analyse de vidéo
    # ------------------------------------------------------------------
    def display_frame(self, frame):
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        image = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        # Pour correspondre à l'aperçu du TP, l'image remplit toute la zone d'aperçu.
        self.set_pixmap_in_label(self.label_video_preview, pixmap)

    def handle_load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger la vidéo .avi",
            "",
            "Fichiers vidéo (*.avi *.mp4 *.mov)"
        )

        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            QMessageBox.critical(self, "Erreur vidéo", "Impossible d'ouvrir le fichier vidéo.")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            nb_trames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.video_path = file_path
            self.video_info = {
                "fps": fps,
                "width": width,
                "height": height,
                "nb_trames": nb_trames,
                "taille": os.path.getsize(file_path) / (1024 * 1024)
            }

            self.video_features.setPlainText(
                f"Fichier: {os.path.basename(file_path)}\n"
                f"Résolution: {width}x{height}\n"
                f"FPS: {fps:.2f}\n"
                f"Nombre de trames: {nb_trames}"
            )

            ret, frame = cap.read()

            if ret:
                self.display_frame(frame)
            else:
                QMessageBox.warning(self, "Attention", "Impossible de lire la première trame.")

            self.video_results.clear()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur vidéo",
                f"Impossible d'analyser la vidéo.\n{str(e)}"
            )

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # 4.5. Compression de vidéo numérique
    # ------------------------------------------------------------------
    def handle_video_compression(self):
        if self.video_path is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger une vidéo.")
            return

        codec_item = self.list_codec.currentItem()

        if codec_item is None:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner un codec.")
            return

        codec = codec_item.text()

        try:
            new_fps = float(self.textEdit_fps.toPlainText().strip())
            new_width = int(self.textEdit_width.toPlainText().strip())
            new_height = int(self.textEdit_height.toPlainText().strip())
        except ValueError:
            QMessageBox.warning(self, "Attention", "FPS, Width et Height doivent être numériques.")
            return

        if new_fps <= 0 or new_width <= 0 or new_height <= 0:
            QMessageBox.warning(self, "Attention", "FPS, Width et Height doivent être positifs.")
            return

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            QMessageBox.critical(self, "Erreur vidéo", "Impossible de rouvrir la vidéo source.")
            return

        try:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = os.path.join(
                os.path.dirname(self.video_path),
                f"{base_name}_compressed_{codec}.avi"
            )

            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (new_width, new_height))

            if not out.isOpened():
                QMessageBox.critical(self, "Erreur vidéo", "Impossible de créer le fichier compressé.")
                cap.release()
                return

            nb_trames = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_resized = cv2.resize(frame, (new_width, new_height))
                out.write(frame_resized)
                nb_trames += 1

                if nb_trames % 50 == 0:
                    QtWidgets.QApplication.processEvents()

            cap.release()
            out.release()

            ancienne_taille = os.path.getsize(self.video_path) / (1024 * 1024)
            nouvelle_taille = os.path.getsize(output_path) / (1024 * 1024)

            self.video_results.setPlainText(
                f"Ancienne taille: {ancienne_taille:.2f} MB\n"
                f"Nouvelle taille: {nouvelle_taille:.2f} MB\n"
                f"Résolution: {new_width}x{new_height}\n"
                f"FPS: {new_fps:.2f}\n"
                f"Nombre de trames: {nb_trames}"
            )

            QMessageBox.information(
                self,
                "Compression terminée",
                f"Vidéo compressée enregistrée dans:\n{output_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Erreur compression vidéo",
                f"Impossible de compresser la vidéo.\n{str(e)}"
            )

        finally:
            if cap.isOpened():
                cap.release()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())
