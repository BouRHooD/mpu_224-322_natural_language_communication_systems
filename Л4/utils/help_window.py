# Загружаем библиотеки
import os
import faiss  
import gensim
import random
import datetime
import numpy as np
import pandas as pd
from math import sqrt
import speech_recognition as sr    
from gensim.models import Word2Vec
from utils.generate_sqlite import *

# Библиотеки GUI интерфейса (pip install PyQt5) (Qt Designer to edit *.ui - https://build-system.fman.io/qt-designer-download)
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class SpeechRecognitionThread(QObject):
        user_voice_text_value_signal = pyqtSignal(str)
        end_get_text_from_voice_signal = pyqtSignal(bool)
        
        def __init__(self):
            super().__init__()
            self._isRunning = True
            # Создаем объект на основе библиотеки speech_recognition и вызываем метод для определения данных
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            print(f"{self.microphone.list_working_microphones()=}")

        @pyqtSlot()
        def get_text_from_voice_run(self):
            try:
                if not self._isRunning: self._isRunning = True; 

                # Начинаем прослушивать микрофон и записываем данные в source
                with self.microphone as source:
                    print("Пользователь говорит...")
                    #self.recognizer.pause_threshold = 0.5             # Устанавливаем паузу, чтобы прослушивание началось лишь по прошествию 1 секунды
                    self.recognizer.adjust_for_ambient_noise(source)   # используем adjust_for_ambient_noise для удаления посторонних шумов из аудио дорожки
                    audio = self.recognizer.listen(source)             # Полученные данные записываем в переменную audio пока мы получили лишь mp3 звук
                    if not self._isRunning: return; 
                
                    # Обрабатываем все при помощи исключений
                    try: 
                        print("Рассшифровываем голос пользователя...")
                        zadanie = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                        if not self._isRunning: return; 
                        
                        self.user_voice_text_value_signal.emit(zadanie) 
                    # Если не смогли распознать текст, то будет вызвана эта ошибка
                    except sr.UnknownValueError:
                        print("Не удалось распознать речь recognize_google")
                    except sr.RequestError as e:
                        print(f"Ошибка при запросе к сервису распознавания: {e}")
            except Exception as ex:
                print(ex)
            finally:
                self.end_get_text_from_voice_signal.emit(self._isRunning)

        def stop(self):
            self._isRunning = False

''' -------- Форма хелпа ------- '''   
class HelpWindow(QMainWindow):       
    def __init__(self, *args, **kwargs):
        super(HelpWindow, self).__init__(*args, **kwargs)

        # Настройки и запуск формы
        self.formOpening()    
        self.voice_ico_on = 'resources\_images\on_voice.png'    
        self.voice_ico_off = 'resources\_images\icons_voicecontrol.png'  
        self.ui.pushButton_SendByVoiseHelp.setIcon(QIcon(self.voice_ico_off))  
        
        self.dislike = 'resources\_images\dislike.png'  
        self.ui.pushButton_Dislike.setIcon(QIcon(self.dislike))  

        self.dir_resource = 'resources'
        self.isDoHelp_version = 2 # old = 1
        if self.isDoHelp_version == 1:
            self.db = SqliteInteraction(self.dir_resource + '/w2v_learn_v1/w2v_sapr.db')
            self.w2v_model = Word2Vec.load(self.dir_resource + "/w2v_learn_v1/articles_model.model") 
            self.ui.pushButton_Dislike.setVisible(True) 
        if self.isDoHelp_version == 2:
            self.ui.pushButton_Dislike.setVisible(False)

            # Загружаем данные
            self.db = SqliteInteraction(self.dir_resource + '/w2v_learn_v2/w2v_sapr.db')
            self.w2v_model = Word2Vec.load(self.dir_resource + "/w2v_learn_v2/w2v_sapr_v2.model")  
            self.df_answers = pd.read_csv(self.dir_resource + '/w2v_learn_v2/Question_Answer.csv', sep=";")
            self.vector_npz = np.load(self.dir_resource + '/w2v_learn_v2/vector.npz')
            self.ques_vec = self.vector_npz['x'] 

            # Проверка загруженных данных
            print(self.df_answers.head(10))
            print(f"{self.df_answers.isna().sum()=}")

            # Изменяем БД под таблицу вопросов и ответов (для легкого редактирования вопросов и ответов)
            pass


        self.doHelpStage = 0 
        self.user_vector = None
        self.HelpNotHelpsStage = 0   

        '''------------ ПОТОК -----------------'''
        # Инициализируем связь потока с pyqt
        self.worker_speach = SpeechRecognitionThread()
        self.worker_speach.user_voice_text_value_signal.connect(self.user_voice_text_value_signal)
        self.worker_speach.end_get_text_from_voice_signal.connect(self.end_get_text_from_voice_signal)
        
        # Переменные главного потока
        self.IsOnVoiceRecognition = False

        # Поток для работы генетического алгоритма
        self.qthread_speach = QThread(parent=self)
        self.qthread_speach.started.connect(self.worker_speach.get_text_from_voice_run)
        self.worker_speach.moveToThread(self.qthread_speach)
        '''------------ ПОТОК -----------------'''                  

        # Подписки на события
        self.ui.pushButton_Dislike.clicked.connect(self.pushButton_Dislike_Clicked)
        self.ui.pushButton_ClearHelp.clicked.connect(self.pushButton_ClearHelp_Clicked)
        self.ui.pushButton_SendMsgHelp.clicked.connect(self.pushButton_SendMsgHelp_Clicked)
        self.ui.pushButton_SendByVoiseHelp.clicked.connect(self.pushButton_SendByVoiseHelp_Clicked)

        # Вывод сообщения о готовности ситемы работать
        self.pushButton_ClearHelp_Clicked(ignore_warning=True)
        
    def formOpening(self):
        # Настройки окна главной формы
        file_ui_path = 'GUI_HELP.ui'
        self.file_icon_path = 'resources\_images\surflay.ico'
        self.ui = uic.loadUi(file_ui_path)                   # GUI, должен быть в папке с main.py
        self.ui.setWindowTitle('Леонов Владислав 224-322')   # Название главного окна
        self.ui.setWindowIcon(QIcon(self.file_icon_path))    # Иконка на гланое окно
        self.ui.setWindowFlags(self.ui.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.ui.show()                                       # Открываем окно формы  
    
    def user_voice_text_value_signal(self, _text):
        self.end_get_text_from_voice_signal(True)
        self.doHelp(_text)

    def end_get_text_from_voice_signal(self, _bool):
        if _bool:
            self.IsOnVoiceRecognition = False
            self.ui.pushButton_SendByVoiseHelp.setIcon(QIcon(self.voice_ico_off))
    
    @pyqtSlot()
    def pushButton_SendByVoiseHelp_Clicked(self):
        '''Завершаем поток'''
        if self.IsOnVoiceRecognition: 
            self.closeEvent()
            return
        '''Запускаем поток'''
        self.closeEvent()
        self.IsOnVoiceRecognition = True
        self.ui.pushButton_SendByVoiseHelp.setIcon(QIcon(self.voice_ico_on))  
        self.qthread_speach.start() 
    
    @pyqtSlot()
    def closeEvent(self):
        '''Завершаем поток'''
        self.worker_speach.stop()
        self.qthread_speach.quit()
        #self.qthread_speach.wait()
        self.IsOnVoiceRecognition = False
        self.ui.pushButton_SendByVoiseHelp.setIcon(QIcon(self.voice_ico_off))

    def plainTextEdit_TextDialogHelp_append_with_date_bot(self, text):
        self.ui.plainTextEdit_TextDialogHelp.appendPlainText(f"{datetime.datetime.now().replace(microsecond=0)} [🤖]: {text}")

    def plainTextEdit_TextDialogHelp_append_with_date_user(self, text):
        self.ui.plainTextEdit_TextDialogHelp.appendPlainText(f"{datetime.datetime.now().replace(microsecond=0)} [👤]: {text}")

    def pushButton_Dislike_Clicked(self):
        self.doHelp('Не правильно', False)

    def pushButton_ClearHelp_Clicked(self, ignore_warning = False):
        if not ignore_warning:
            msg = QMessageBox(self) 
            msg.setIcon(QMessageBox.Warning) 
            msg.setText("Вы уверены, что хотите очистить диалог с системой?") 
            msg.setWindowTitle("Внимание!") 
            msg.setWindowIcon(QIcon(self.file_icon_path))
            msg.addButton("Нет", QMessageBox.NoRole)
            yes_button = msg.addButton("Да", QMessageBox.YesRole)
            msg.setWindowModality(0)
            msg.activateWindow()
            msg.show()
            msg.exec_()
            if msg.clickedButton() != yes_button: return; 
        self.ui.plainTextEdit_TextDialogHelp.clear()
        self.plainTextEdit_TextDialogHelp_append_with_date_bot("Система готова к работе!")
        self.plainTextEdit_TextDialogHelp_append_with_date_bot('Тема "Адаптивный интерфейс САПР на основе нейросетевого анализа пользовательских логов в задачах предсказания команд"')
        self.plainTextEdit_TextDialogHelp_append_with_date_bot('Введите запрос и система постарается дать на него ответ')

    def pushButton_SendMsgHelp_Clicked(self):
        user_text = self.ui.plainTextEdit_UserTextHelp.toPlainText()
        self.doHelp(user_text)
        self.ui.plainTextEdit_UserTextHelp.clear()
    
    def getRubricsFromDB(self):
        rubrics = self.db.select('rubrics')
        rubrics = rubrics.fetchall()
        return rubrics

    def doHelp(self, inText: str, printInText: bool = True):
        if self.isDoHelp_version == 1: self.doHelp_v1(inText, printInText);  
        if self.isDoHelp_version == 2: self.doHelp_v2(inText, printInText);  

    def doHelp_v1(self, inText: str, printInText: bool = True):
        if len(inText) <= 0: return; 
        if printInText:
            self.plainTextEdit_TextDialogHelp_append_with_date_user(inText)

        inText = inText.lower()
        if inText == 'не правильно': 
            if self.user_vector == None: return; 
            self.doHelpStage = 1

        if self.doHelpStage == 0:
            try:
                words = 0
                vectors_sum = 0
                for word in inText.split():
                    _word = word.replace("?", "")
                    if self.w2v_model.wv.has_index_for(_word) and len(_word) > 3:
                        vectors_sum+=self.w2v_model.wv.get_vector(_word).sum()
                        words+=1

                if words == 0:
                    self.plainTextEdit_TextDialogHelp_append_with_date_bot("Запрос не обработан, измените написание слова")
                    return
                
                self.user_vector = vectors_sum / words
                rubrics = self.getRubricsFromDB()
                final_vector = float(rubrics[0][0])
                final_rubric = rubrics[0][1]
                for i in rubrics:
                    if abs(float(i[0]) - self.user_vector) > abs(final_vector - self.user_vector):
                        final_vector = float(i[0])
                        final_rubric = i[1]

                runbrics_word = self.db.selectWhere('rubrics_word', {'rubric': final_rubric})
                res = []
                for i in runbrics_word:
                    for j in i:
                        if j[1].lower() in inText.lower():
                            res.append(f"{j[1]} - {j[2]}")

                if len(res) != 0:
                    for i in res:
                        self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"[Рубрика - {final_rubric}]\n{i}")
                else:
                    print("Подтема не определена, выдаем рандомную информацию из базы данных")
                    rand = random.choice(runbrics_word)
                    rand = random.choice(rand)
                    self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"[Рубрика - {final_rubric}]\n{rand[1]} - {rand[2]}")
            except Exception as ex: 
                print(ex)
        elif self.doHelpStage == 1:
            try:
                self.plainTextEdit_TextDialogHelp_append_with_date_bot("Введите правильную рубрику:")
                rubrics = self.getRubricsFromDB()
                count = 1
                for i in rubrics:
                    self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"{count} - {i[1]}")
                    count += 1
                self.doHelpStage += 1; 
            except Exception as ex: 
                print(ex)
                self.doHelpStage = 0; 
        elif self.doHelpStage == 2:
            try:
                real_rubric = int(inText) - 1
                rubrics = self.getRubricsFromDB()
                sing = 1
                if float(rubrics[real_rubric][0]) < float(self.user_vector):
                    refactor = sqrt(abs(float(self.user_vector) - float(rubrics[real_rubric][0])))
                else:
                    sing = -1
                    refactor = sqrt(abs(float(rubrics[real_rubric][0]) - float(self.user_vector)))
                vector = float(rubrics[real_rubric][0]) + (sing * refactor)
                self.db.update('rubrics', {'vector': vector}, 'rubric', rubrics[real_rubric][1])
                self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"Рубрика '{rubrics[real_rubric][1]}' обновлена, новый вектор = {vector}")
            except Exception as ex: 
                print(ex)
                self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"Рубрика не обновлена!\nВведите новый запрос")
            finally:
                self.doHelpStage = 0; 

    def doHelp_v2(self, inText: str, printInText: bool = True):
        if len(inText) <= 0: return; 
        if printInText:
            self.plainTextEdit_TextDialogHelp_append_with_date_user(inText)

        inText = inText.lower()
        if inText == 'не правильно': 
            if self.user_vector == None: return; 
            self.doHelpStage = 1

        if self.doHelpStage == 0:
            try:
                def trained_sentence_vec(sent):
                    # Filter out terms that are not in the vocabulary from the question sentence
                    #qu_voc = [tm for tm in sent if tm in w2v.wv]
                    qu_voc = []
                    for tm in sent:
                        if tm in self.w2v_model.wv:
                            qu_voc.append(tm)

                    # Get the embedding of the characters
                    #emb = np.vstack([w2v.wv[tm] for tm in qu_voc])
                    w2v_list = []
                    for tm in qu_voc:
                        w2v_list.append(self.w2v_model.wv[tm])
                    if len(w2v_list) <= 0: return None; 
                    emb = np.vstack(w2v_list)
                    # Calculate the vectors of each included word to get the vector of the question
                    ave_vec = np.mean(emb, axis=0)
                    return ave_vec

                def find_answer(qr_sentence, ques_vec):
                    # use one query sentence to retrieve answer
                    qr_sentence = gensim.utils.simple_preprocess(qr_sentence)
                    qr_sent_vec = trained_sentence_vec(qr_sentence)
                    if qr_sent_vec is None: return None; 

                    # perform vector search through similarity comparison
                    n_dim = ques_vec.shape[1]
                    x = np.vstack(ques_vec).astype(np.float32)
                    q = qr_sent_vec.reshape(1, -1)
                    index = faiss.index_factory(n_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
                    faiss.normalize_L2(x)
                    index.add(x)
                    faiss.normalize_L2(q)
                    similarity, idx = index.search(q, k=index.ntotal)
                    ans_idx = idx[0][0]
                    return ans_idx
                
                ans_idx = find_answer(inText, self.ques_vec)
                response_rubric = self.df_answers["Рубрика"][ans_idx]
                response_question = self.df_answers["Вопрос"][ans_idx]
                response_answer = self.df_answers["Ответ"][ans_idx]

                print("Запрос: ", inText)
                print("Вопрос: ", response_question)
                print("Ответ: ", response_answer)

                self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"\n[Рубрика - {response_rubric}]\n{response_question} - {response_answer}")
            except Exception as ex: 
                print(ex)
        elif self.doHelpStage == 1:
            try:
                pass
            except Exception as ex: 
                print(ex)
                self.doHelpStage = 0; 
        elif self.doHelpStage == 2:
            try:
                pass
            except Exception as ex: 
                print(ex)
                self.plainTextEdit_TextDialogHelp_append_with_date_bot(f"Рубрика не обновлена!\nВведите новый запрос")
            finally:
                self.doHelpStage = 0;       