#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, os.path
from pydub import AudioSegment
from tqdm import tqdm


# In[2]:


input_fold="./all_samples"
output_fold="./all_samples_wav"
sr = 22050

for dirname, _, filenames in tqdm(os.walk(input_fold)):
    for filename in filenames:
        # files                                                                         
        src = f'{dirname}/{filename}'
        dst_fold = f'{output_fold}/{dirname[len(input_fold)+1:]}'
        
        # create target folder if not exists
        isExist = os.path.exists(dst_fold)
        if not isExist:
            os.makedirs(dst_fold)
                
        dst = f'{dst_fold}/{os.path.splitext(filename)[0]}.wav'
                
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(src)
        sound = sound.set_frame_rate(sr)
        sound.export(dst, format="wav")

