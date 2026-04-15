from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import sys

qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.gray = None
        self.mag_res = None

        self.parcourir.clicked.connect(self.get_image)
        self.valider1.clicked.connect(self.apply_first_derivative)
        self.valider2.clicked.connect(self.compute_gradient_edges)
        self.applaplacien.clicked.connect(self.apply_laplacian)
        self.applog.clicked.connect(self.apply_log)
        self.appcanny.clicked.connect(self.apply_canny)

    def makeFigure(self, img, widget):
        widget.setPixmap(self.cvToPixmap(img))
        widget.setScaledContents(True)

    def get_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png)"
        )

        if path:
            self.image = cv2.imread(path)
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.makeFigure(self.gray, self.GrayImg)

    def cvToPixmap(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
            q_img = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h, w, ch = img.shape
            q_img = QtGui.QImage(img.data, w, h, w * ch, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def apply_first_derivative(self):
        if self.gray is None:
            return

        if self.fprewitt.isChecked():

            Hx = np.array([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ])

            Hy = np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ])

            gx = cv2.filter2D(self.gray, cv2.CV_64F, Hx)
            gy = cv2.filter2D(self.gray, cv2.CV_64F, Hy)

        elif self.fsobel.isChecked():

            gx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1)

        else:
            return

        mag = np.sqrt(gx**2 + gy**2)
        self.mag_res = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.makeFigure(self.mag_res, self.FilteredImg)

    def compute_gradient_edges(self):
        if self.mag_res is None:
            self.apply_first_derivative()

        try:
            seuil_bas = int(self.s1.toPlainText())
        except:
            seuil_bas = 50

        try:
            seuil_haut = int(self.s2.toPlainText())
        except:
            seuil_haut = 150

        _, contours_binaires = cv2.threshold(
            self.mag_res,
            seuil_bas,
            seuil_haut,
            cv2.THRESH_BINARY
        )

        self.makeFigure(contours_binaires, self.SegmentedImg)

    def apply_laplacian(self):
        if self.gray is None:
            return

        laplacian = cv2.convertScaleAbs(
            cv2.Laplacian(self.gray, cv2.CV_64F)
        )

        self.makeFigure(laplacian, self.LaplacienImg)

    def apply_log(self):
        if self.gray is None:
            return

        blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
        log_res = cv2.convertScaleAbs(
            cv2.Laplacian(blur, cv2.CV_64F)
        )

        self.makeFigure(log_res, self.LoGImg)

    def apply_canny(self):
        if self.gray is None:
            return

        image_canny = cv2.Canny(self.gray, 100, 200)
        self.makeFigure(image_canny, self.CannyImg)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())