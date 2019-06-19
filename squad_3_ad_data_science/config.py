from os import path

import squad_3_ad_data_science

base_path = path.dirname(path.dirname(squad_3_ad_data_science.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
models_path = path.join(workspace_path, 'models')
