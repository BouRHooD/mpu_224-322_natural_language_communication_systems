import random
from math import sqrt
from gensim.models import Word2Vec
from generate_sqlite import SqliteInteraction

dir_path = 'resources'
db = SqliteInteraction(dir_path + '/rd.db')
w2v_model = Word2Vec.load(dir_path + "/model_name.model")

while True:
    user_input = input("Введите запрос (0 для завершения работы)\n")
    if user_input == "0": exit(1); 

    words = 0
    vectors_sum = 0
    for word in user_input.split():
        if w2v_model.wv.has_index_for(word) and len(word) > 3:
            vectors_sum+=w2v_model.wv.get_vector(word).sum()
            words+=1

    if words == 0:
        print("Запрос не обработан, измените написание слова\n")
    else:
        user_vector = vectors_sum / words
        rubrics = db.select('rubrics')
        rubrics = rubrics.fetchall()
        final_vector = float(rubrics[0][0])
        final_rubric = rubrics[0][1]
        for i in rubrics:
            if abs(float(i[0]) - user_vector) > abs(final_vector - user_vector):
                final_vector = float(i[0])
                final_rubric = i[1]

        print(f"Рубрика определена !! {final_rubric} !!\n")

        runbrics_word = db.selectWhere('rubrics_word', {'rubric': final_rubric})
        res = []
        for i in runbrics_word:
            for j in i:
                if j[1].lower() in user_input.lower():
                    res.append(f"{j[1]} - {j[2]}")

        print("Найдено:")
        if len(res) != 0:
            for i in res:
                print(i)
        else:
            print("Подтема не определена, выдаем рандомную информацию из базы данных\n")
            rand = random.choice(runbrics_word)
            rand = random.choice(rand)
            print(f"{rand[1]} - {rand[2]}")

        valid = int(input("Соотвествует ли информация запросу? 1 - да, 0 - нет\n"))
        if not valid:
            count = 0
            for i in rubrics:
                print(f"{count} - {i[1]}")
                count+=1

            real_rubric = int(input("Выберите верную рубрику\n"))
            sing = 1
            if float(rubrics[real_rubric][0]) < float(user_vector):
                refactor = sqrt(abs(float(user_vector) - float(rubrics[real_rubric][0])))
            else:
                sing = -1
                refactor = sqrt(abs(float(rubrics[real_rubric][0]) - float(user_vector)))
            vector = float(rubrics[real_rubric][0]) + (sing * refactor)
            db.update('rubrics', {'vector': vector}, 'rubric', rubrics[real_rubric][1])
