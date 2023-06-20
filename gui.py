# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import time
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from inference import predict_with_model, __VERSION__
import torch


root = dict()


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, options):
        super().__init__()
        self.options = options

    def run(self):
        global root
        # Here we pass the update_progress (uncalled!)
        self.options['update_percent_func'] = self.update_progress
        predict_with_model(self.options)
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        self.finished.emit()

    def update_progress(self, percent):
        self.progress.emit(percent)


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        global root

        Dialog.setObjectName("Settings")
        Dialog.resize(370, 320)

        self.checkbox_cpu = QCheckBox("Use CPU instead of GPU?", Dialog)
        self.checkbox_cpu.move(30, 10)
        self.checkbox_cpu.resize(320, 40)
        if root['cpu']:
            self.checkbox_cpu.setChecked(True)

        self.checkbox_single_onnx = QCheckBox("Use single ONNX?", Dialog)
        self.checkbox_single_onnx.move(30, 40)
        self.checkbox_single_onnx.resize(320, 40)
        if root['single_onnx']:
            self.checkbox_single_onnx.setChecked(True)

        self.checkbox_large_gpu = QCheckBox("Use large GPU?", Dialog)
        self.checkbox_large_gpu.move(30, 70)
        self.checkbox_large_gpu.resize(320, 40)
        if root['large_gpu']:
            self.checkbox_large_gpu.setChecked(True)

        self.checkbox_kim_1 = QCheckBox("Use old Kim Vocal model?", Dialog)
        self.checkbox_kim_1.move(30, 100)
        self.checkbox_kim_1.resize(320, 40)
        if root['use_kim_model_1']:
            self.checkbox_kim_1.setChecked(True)

        self.checkbox_only_vocals = QCheckBox("Generate only vocals/instrumental?", Dialog)
        self.checkbox_only_vocals.move(30, 130)
        self.checkbox_only_vocals.resize(320, 40)
        if root['only_vocals']:
            self.checkbox_only_vocals.setChecked(True)

        self.chunk_size_label = QLabel(Dialog)
        self.chunk_size_label.setText('Chunk size')
        self.chunk_size_label.move(30, 160)
        self.chunk_size_label.resize(320, 40)

        self.chunk_size_valid = QIntValidator(bottom=100000, top=10000000)
        self.chunk_size = QLineEdit(Dialog)
        self.chunk_size.setFixedWidth(140)
        self.chunk_size.move(130, 170)
        self.chunk_size.setValidator(self.chunk_size_valid)
        self.chunk_size.setText(str(root['chunk_size']))

        self.overlap_large_label = QLabel(Dialog)
        self.overlap_large_label.setText('Overlap large')
        self.overlap_large_label.move(30, 190)
        self.overlap_large_label.resize(320, 40)

        self.overlap_large_valid = QDoubleValidator(bottom=0.001, top=0.999, decimals=10)
        self.overlap_large_valid.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.overlap_large = QLineEdit(Dialog)
        self.overlap_large.setFixedWidth(140)
        self.overlap_large.move(130, 200)
        self.overlap_large.setValidator(self.overlap_large_valid)
        self.overlap_large.setText(str(root['overlap_large']))

        self.overlap_small_label = QLabel(Dialog)
        self.overlap_small_label.setText('Overlap small')
        self.overlap_small_label.move(30, 220)
        self.overlap_small_label.resize(320, 40)

        self.overlap_small_valid = QDoubleValidator(0.001, 0.999, 10)
        self.overlap_small_valid.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.overlap_small = QLineEdit(Dialog)
        self.overlap_small.setFixedWidth(140)
        self.overlap_small.move(130, 230)
        self.overlap_small.setValidator(self.overlap_small_valid)
        self.overlap_small.setText(str(root['overlap_small']))

        self.pushButton_save = QPushButton(Dialog)
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_save.move(30, 280)
        self.pushButton_save.resize(150, 35)

        self.pushButton_cancel = QPushButton(Dialog)
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        self.pushButton_cancel.move(190, 280)
        self.pushButton_cancel.resize(150, 35)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.Dialog = Dialog

        # connect the two functions
        self.pushButton_save.clicked.connect(self.return_save)
        self.pushButton_cancel.clicked.connect(self.return_cancel)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Settings", "Settings"))
        self.pushButton_cancel.setText(_translate("Settings", "Cancel"))
        self.pushButton_save.setText(_translate("Settings", "Save settings"))

    def return_save(self):
        global root
        # print("save")
        root['cpu'] = self.checkbox_cpu.isChecked()
        root['single_onnx'] = self.checkbox_single_onnx.isChecked()
        root['large_gpu'] = self.checkbox_large_gpu.isChecked()
        root['use_kim_model_1'] = self.checkbox_kim_1.isChecked()
        root['only_vocals'] = self.checkbox_only_vocals.isChecked()

        chunk_size_text = self.chunk_size.text()
        state = self.chunk_size_valid.validate(chunk_size_text, 0)
        if state[0] == QValidator.State.Acceptable:
            root['chunk_size'] = chunk_size_text

        overlap_large_text = self.overlap_large.text()
        # locale problems... it wants comma instead of dot
        if 0:
            state = self.overlap_large_valid.validate(overlap_large_text, 0)
            if state[0] == QValidator.State.Acceptable:
                root['overlap_large'] = float(overlap_large_text)
        else:
            root['overlap_large'] = float(overlap_large_text)

        overlap_small_text = self.overlap_small.text()
        if 0:
            state = self.overlap_small_valid.validate(overlap_small_text, 0)
            if state[0] == QValidator.State.Acceptable:
                root['overlap_small'] = float(overlap_small_text)
        else:
            root['overlap_small'] = float(overlap_small_text)

        self.Dialog.close()

    def return_cancel(self):
        global root
        # print("cancel")
        self.Dialog.close()


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(560, 360)
        self.move(300, 300)
        self.setWindowTitle('MVSEP music separation model')
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        global root
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)

    def execute_long_task(self):
        global root

        if len(root['input_files']) == 0 and 1:
            QMessageBox.about(root['w'], "Error", "No input files specified!")
            return

        root['progress_bar'].show()
        root['button_start'].setDisabled(True)
        root['button_finish'].setDisabled(False)
        root['start_proc'] = True

        options = {
            'input_audio': root['input_files'],
            'output_folder': root['output_folder'],
            'cpu': root['cpu'],
            'single_onnx': root['single_onnx'],
            'large_gpu': root['large_gpu'],
            'chunk_size': root['chunk_size'],
            'overlap_large': root['overlap_large'],
            'overlap_small': root['overlap_small'],
            'use_kim_model_1': root['use_kim_model_1'],
            'only_vocals': root['only_vocals'],
        }

        self.update_progress(0)
        self.thread = QThread()
        self.worker = Worker(options)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)

        self.thread.start()

    def stop_separation(self):
        global root
        self.thread.terminate()
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        root['progress_bar'].hide()

    def update_progress(self, progress):
        global root
        root['progress_bar'].setValue(progress)

    def open_settings(self):
        global root
        dialog = QDialog()
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)
        dialog.exec_()


def dialog_select_input_files():
    global root
    files, _ = QFileDialog.getOpenFileNames(
        None,
        "QFileDialog.getOpenFileNames()",
        "",
        "All Files (*);;Audio Files (*.wav, *.mp3, *.flac)",
    )
    if files:
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)
    return files


def dialog_select_output_folder():
    global root
    foldername = QFileDialog.getExistingDirectory(
        None,
        "Select Directory"
    )
    root['output_folder'] = foldername + '/'
    root['output_folder_line_edit'].setText(root['output_folder'])
    return foldername


def create_dialog():
    global root
    app = QApplication(sys.argv)

    w = MyWidget()

    root['input_files'] = []
    root['output_folder'] = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    root['cpu'] = False
    root['large_gpu'] = False
    root['single_onnx'] = False
    root['chunk_size'] = 1000000
    root['overlap_large'] = 0.6
    root['overlap_small'] = 0.5
    root['use_kim_model_1'] = False
    root['only_vocals'] = False

    t = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
    if t > 11.5:
        print('You have enough GPU memory ({:.2f} GB), so we set fast GPU mode. You can change in settings!'.format(t))
        root['large_gpu'] = True
        root['single_onnx'] = False
    elif t < 8:
        root['large_gpu'] = False
        root['single_onnx'] = True
        root['chunk_size'] = 500000

    button_select_input_files = QPushButton(w)
    button_select_input_files.setText("Input audio files")
    button_select_input_files.clicked.connect(dialog_select_input_files)
    button_select_input_files.setFixedHeight(35)
    button_select_input_files.setFixedWidth(150)
    button_select_input_files.move(30, 20)

    input_files_list_text_area = QTextEdit(w)
    input_files_list_text_area.setReadOnly(True)
    input_files_list_text_area.setLineWrapMode(QTextEdit.NoWrap)
    font = input_files_list_text_area.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    input_files_list_text_area.move(30, 60)
    input_files_list_text_area.resize(500, 100)

    button_select_output_folder = QPushButton(w)
    button_select_output_folder.setText("Output folder")
    button_select_output_folder.setFixedHeight(35)
    button_select_output_folder.setFixedWidth(150)
    button_select_output_folder.clicked.connect(dialog_select_output_folder)
    button_select_output_folder.move(30, 180)

    output_folder_line_edit = QLineEdit(w)
    output_folder_line_edit.setReadOnly(True)
    font = output_folder_line_edit.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    output_folder_line_edit.move(30, 220)
    output_folder_line_edit.setFixedWidth(500)
    output_folder_line_edit.setText(root['output_folder'])

    progress_bar = QProgressBar(w)
    # progress_bar.move(30, 310)
    progress_bar.setValue(0)
    progress_bar.setGeometry(30, 310, 500, 35)
    progress_bar.setAlignment(QtCore.Qt.AlignCenter)
    progress_bar.hide()
    root['progress_bar'] = progress_bar

    button_start = QPushButton('Start separation', w)
    button_start.clicked.connect(w.execute_long_task)
    button_start.setFixedHeight(35)
    button_start.setFixedWidth(150)
    button_start.move(30, 270)

    button_finish = QPushButton('Stop separation', w)
    button_finish.clicked.connect(w.stop_separation)
    button_finish.setFixedHeight(35)
    button_finish.setFixedWidth(150)
    button_finish.move(200, 270)
    button_finish.setDisabled(True)

    button_settings = QPushButton('⚙', w)
    button_settings.clicked.connect(w.open_settings)
    button_settings.setFixedHeight(35)
    button_settings.setFixedWidth(35)
    button_settings.move(495, 270)
    button_settings.setDisabled(False)

    mvsep_link = QLabel(w)
    mvsep_link.setOpenExternalLinks(True)
    font = mvsep_link.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    mvsep_link.move(415, 30)
    mvsep_link.setText('Powered by <a href="https://mvsep.com">MVSep.com</a>')

    root['w'] = w
    root['input_files_list_text_area'] = input_files_list_text_area
    root['output_folder_line_edit'] = output_folder_line_edit
    root['button_start'] = button_start
    root['button_finish'] = button_finish
    root['button_settings'] = button_settings

    # w.showMaximized()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    print('Version: {}'.format(__VERSION__))
    create_dialog()
