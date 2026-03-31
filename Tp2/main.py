from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import sys
import random
import matplotlib.pyplot as plt

qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.gray_image = None

        self.BinaryThreshold.setChecked(True)
        self.GaussianFilter.setChecked(True)
        self.Rotation.setChecked(True)

        self.Browse.clicked.connect(self.get_image)
        self.ShowHistogram.clicked.connect(self.show_HistOriginal)
        self.Apply.clicked.connect(self.show_ImgHistEqualized)
        self.Validate_1.clicked.connect(self.show_ImgThresholding)
        self.Validate_2.clicked.connect(self.show_ImgFiltered)
        self.Validate_3.clicked.connect(self.show_ImgAugmented)

    def clear_widget(self, widget):
        layout = widget.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def makeFigure(self, widget, image_path):
        self.clear_widget(widget)

        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)

        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            label.setText("Erreur affichage")
        else:
            pixmap = pixmap.scaled(
                widget.width() - 10,
                widget.height() - 10,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            label.setPixmap(pixmap)

        widget.layout().addWidget(label)

    def get_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )

        if file_path == "":
            return

        self.image = cv2.imread(file_path)
        self.gray_image = cv2.imread(file_path, 0)

        if self.gray_image is None:
            return

        cv2.imwrite("Original_Image.png", self.gray_image)
        self.makeFigure(self.OriginalImg, "Original_Image.png")
        self.show_HistOriginal()

    def show_HistOriginal(self):
        if self.gray_image is None:
            return

        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])

        plt.figure(figsize=(6, 3))
        plt.plot(hist)
        plt.tight_layout()
        plt.savefig("Original_Histogram.png")
        plt.close()

        self.makeFigure(self.OriginalHist, "Original_Histogram.png")

    def show_ImgHistEqualized(self):
        if self.gray_image is None:
            return

        image_egalisee = cv2.equalizeHist(self.gray_image)

        cv2.imwrite("Equalized_Image.png", image_egalisee)
        self.makeFigure(self.EqualizedImg, "Equalized_Image.png")

        hist = cv2.calcHist([image_egalisee], [0], None, [256], [0, 256])

        plt.figure(figsize=(6, 3))
        plt.plot(hist)
        plt.tight_layout()
        plt.savefig("Equalized_Histogram.png")
        plt.close()

        self.makeFigure(self.EqualizedHist, "Equalized_Histogram.png")

    def show_ImgThresholding(self):
        if self.gray_image is None:
            return

        if self.BinaryThreshold.isChecked():
            ret, result = cv2.threshold(
                self.gray_image, 120, 255, cv2.THRESH_BINARY
            )
        else:
            ret, result = cv2.threshold(
                self.gray_image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        cv2.imwrite("Thresholding_Image.png", result)
        self.makeFigure(self.ThresholdingImg, "Thresholding_Image.png")

    def show_ImgFiltered(self):
        if self.gray_image is None:
            return

        if self.MeanFilter.isChecked():
            result = cv2.blur(self.gray_image, (11, 11))

        elif self.GaussianFilter.isChecked():
            result = cv2.GaussianBlur(self.gray_image, (15, 15), 10)

        else:
            result = cv2.medianBlur(self.gray_image, 13)

        cv2.imwrite("Filtered_Image.png", result)
        self.makeFigure(self.FilteredImg, "Filtered_Image.png")

    def show_ImgAugmented(self):
        if self.gray_image is None:
            return

        image = self.gray_image.copy()
        (h, w) = image.shape[:2]

        if self.Rotation.isChecked():
            centre = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centre, 45, 1.0)
            result = cv2.warpAffine(image, M, (w, h))

        elif self.Extraction.isChecked():
            result = image[0:h // 2, 0:w // 2]

        else:
            s = random.uniform(1.5, 4.0)
            new_w = int(w * s)
            new_h = int(h * s)

            zoom = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )

            x = (new_w - w) // 2
            y = (new_h - h) // 2
            result = zoom[y:y + h, x:x + w]

        cv2.imwrite("Augmented_Image.png", result)
        self.makeFigure(self.AugmentedImg, "Augmented_Image.png")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())