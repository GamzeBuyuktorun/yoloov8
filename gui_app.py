import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class YOLOv8GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8: Kupa ve Çay Bardağı Tespiti")
        self.setGeometry(100, 100, 1000, 700)
        
        # Modeli yükle (best.pt dosyasının aynı klasörde olduğundan emin ol)
        self.model = YOLO('best.pt') 
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # Görsel Panelleri
        self.lbl_original = QLabel("Orijinal Resim Buraya Gelecek")
        self.lbl_original.setFrameShape(QFrame.StyledPanel)
        self.lbl_original.setAlignment(Qt.AlignCenter)
        
        self.lbl_result = QLabel("Tahmin Sonucu Buraya Gelecek")
        self.lbl_result.setFrameShape(QFrame.StyledPanel)
        self.lbl_result.setAlignment(Qt.AlignCenter)

        image_layout.addWidget(self.lbl_original)
        image_layout.addWidget(self.lbl_result)

        # Bilgi Paneli
        self.info_label = QLabel("Tespit Edilen Nesneler: -")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Butonlar
        btn_layout = QHBoxLayout()
        btn_select = QPushButton("Select Image")
        btn_select.clicked.connect(self.select_image)
        
        btn_test = QPushButton("Test Image")
        btn_test.clicked.connect(self.test_image)
        
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.save_image)

        btn_layout.addWidget(btn_select)
        btn_layout.addWidget(btn_test)
        btn_layout.addWidget(btn_save)

        # Ana Yerleşim
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_image(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Resim Seç', '', 'Image files (*.jpg *.png *.jpeg)')
        if self.fname:
            pixmap = QPixmap(self.fname).scaled(450, 450, Qt.KeepAspectRatio)
            self.lbl_original.setPixmap(pixmap)
            self.lbl_result.clear()
            self.info_label.setText("Resim yüklendi. Test butonuna basın.")

    def test_image(self):
        if hasattr(self, 'fname'):
            results = self.model(self.fname)
            self.res_plotted = results[0].plot() # Bounding box çizilmiş hali
            
            # Nesne sayılarını hesapla
            names = results[0].names
            counts = results[0].boxes.cls.tolist()
            detected_text = "Tespit Edilenler: "
            for i in set(counts):
                detected_text += f"{names[i]}: {counts.count(i)} adet "
            self.info_label.setText(detected_text)

            # Görüntüyü arayüze bas
            rgb_img = cv2.cvtColor(self.res_plotted, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.lbl_result.setPixmap(QPixmap.fromImage(qt_img).scaled(450, 450, Qt.KeepAspectRatio))

    def save_image(self):
        if hasattr(self, 'res_plotted'):
            save_path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "sonuc.jpg", "JPG (*.jpg);;PNG (*.png)")
            if save_path:
                cv2.imwrite(save_path, self.res_plotted)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8GUI()
    window.show()
    sys.exit(app.exec_())