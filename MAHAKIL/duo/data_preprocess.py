import numpy as np
import pandas as pd

path = "prop/prop.csv"
save_path_data = "prop/data.csv"
save_path_label = "prop/label.csv"
source = pd.read_csv(path)
label = source['bug']
source_ = source.drop(['bug'], axis=1)
source_ = pd.DataFrame(source_)
source_.to_csv(save_path_data, encoding='utf-8', header=False, index=False,sep=',')
label.to_csv(save_path_label, encoding='utf-8', header=False, index=False,sep=',')
