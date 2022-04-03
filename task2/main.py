import data
import numpy as np
import evaluation

# data.extract_data()


training_data = evaluation.eval_training_characterictics_of_all_pics()

# ВОПРОС!!!
# Почему возникает ошибка TypeError: can't pickle module objects при исполнении такого кода
# data.to_pickle(evaluation)

import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(training_data, f)