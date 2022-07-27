import pandas as pd
from tqdm import tqdm
import numpy as np

# meta = pd.read_csv('/datasets/jianhaoy/celebA/img_align_celeba/styleclipf5.csv',index_col=0)
meta = pd.read_csv('/datasets/jianhaoy/celebA/img_align_celeba/styleclip0.4.csv')
print(meta.shape)
nupsample = 5
up_meta = pd.DataFrame(columns = meta.columns)
blond_count = 0
for i in tqdm(range(meta.shape[0])):
    size = up_meta.index.size
    if meta.loc[i]['y'] == 1 and meta.loc[i]['split'] == 0:
        up_meta.loc[size] = meta.loc[i]
        blond_count += 1


# reap_meta = pd.DataFrame(np.repeat(up_meta.values,nupsample,axis=0))
reap_meta = pd.DataFrame()
for i in tqdm(range(up_meta.shape[0])):
    a=up_meta.loc[i]
    d=pd.DataFrame(a).T
    reap_meta=reap_meta.append([d]*nupsample)  #每行复制5倍


upsampled_metadata = pd.concat([meta, reap_meta], ignore_index=True)
print(upsampled_metadata.tail())
print(upsampled_metadata.shape)
upsampled_metadata.to_csv('/datasets/jianhaoy/celebA/img_align_celeba/upsampled_styleclipf4_metadata.csv',index=False)
