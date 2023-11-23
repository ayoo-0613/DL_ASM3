import os
import qlib
import pandas as pd
import numpy as np
from qlib.data import D

def retrieve_data(output_path):
    train_output_path = os.path.join(output_path, 'train')
    test_output_path = os.path.join(output_path, 'test')
    
    # 确保输出路径存在
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    start_train_time_step = '2010-01-01'
    end_train_time_step = '2017-12-31'
    start_test_time_step = '2018-01-01'
    end_test_time_step = '2019-01-01'
    


    for instrm in train_list_instruments:
        df_train = D.features([instrm], fields, start_time=start_train_time_step, end_time=end_train_time_step, freq='day')
        df_train = add_derived_features(df_train)
        df_train.to_csv(os.path.join(train_output_path, f'{instrm}.csv'))

    for instrm in test_list_instruments:
        df_test = D.features([instrm], fields, start_time=start_test_time_step, end_time=end_test_time_step, freq='day')
        df_test = add_derived_features(df_test)
        df_test.to_csv(os.path.join(test_output_path, f'{instrm}.csv'))


# 提取数据中的特征