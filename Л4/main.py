# Загружаем библиотеки
import sys
from utils.help_window import *

# Библиотеки GUI интерфейса (pip install PyQt5) (Qt Designer to edit *.ui - https://build-system.fman.io/qt-designer-download)
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

''' -------- Запуск формы ------- '''
if __name__ == '__main__':                                            # Выполнение условия, если запущен этот файл python, а не если он подгружен через import
    app = QApplication(sys.argv)                                      # Объект приложения (экземпляр QApplication)
    win = HelpWindow()                                                # Создание формы
    sys.exit(app.exec_())                                             # Вход в главный цикл приложения и Выход после закрытия приложения