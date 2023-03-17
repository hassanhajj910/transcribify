import whisper
import pyannote.audio
import pyannote.core
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import numpy as np
from pydub import AudioSegment
import torchaudio
import torch


key_config = dotenv_values('.env')


def time_overlap(x, y):
    res = list(range(max(int(x[0]), int(y[0])), min(int(x[-1]), int(y[-1]))+1))
    if len(res) > 0:
        over = True
    elif len(res) == 0:
        over = False
    return over

# make functions callable instead of running in the object. 
class audio():
    MODELS_AVAILABLE = {
        'pyannote/voice-activity-detection',
        'pyannote/speaker-diarization@2.1', 
        'pyannote/segmentation'
    }

    def __init__(self, filepath:str) -> None:
        self.file = filepath
        self.waveform, self.sample_rate = torchaudio.load(filepath)
        



    def set_pipeline(self): # can add options here
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=key_config['HF_PYANNOTE_DIARIZATION'])
        return pipeline
    
    def get_speech_sections(self):

        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=key_config["HP_PYANNOTE_VOICEACTIVITY"])
        speech_segment = pipeline(self.file)
        speech_segment_tl = speech_segment.get_timeline()
        times = []
        for seg in speech_segment_tl.segments_set_:
            s, e = seg
            times.append([s,e])
        times = np.array(times)
        times = np.sort(times, axis=0)
        
        return np.array(times)

    def get_clean_sections(self, times=None, threshold=None):
        # threshold has to be positive
        if times is None:
            times = self.get_speech_sections().copy()

        if threshold is None:
            return times
        
        if threshold < 0:
            raise ValueError('Threshold must be a positive integer')
         
        # for easier time check. 
        times[:,1] = times[:,1]+int(threshold)
        # to avoid running out of audio-space, last instance is calculated without adding thresh. 
        times[-1][1] = times[-1,1]-int(threshold)
        # get continuous time segments while allocating a threshold buffer.
        init_range = np.array([times[0][0], times[0][1]])
        group = []
        for i in range(len(times)-1):
            res = time_overlap(times[i], times[i+1])
            if res is True:
                init_range[1] = times[i+1][1]
            elif res is False:
                group.append(init_range)
                init_range = np.array([times[i+1][0], times[i+1][1]])
        if res is True:
            group.append(init_range)
        return np.array(group)
    
    def clean_audio(self, times:np.array):
        # add exceptions and errors

        rate = self.sample_rate
        waveform = self.waveform
        
        times = times.astype(int)
        times_sr = times * rate
        waveform_tensor = torch.tensor(waveform)
        indices = []
        for t in times_sr:
            indices.extend(list(range(t[0],t[1])))
        indices = torch.tensor(indices)
        clean_wave = waveform_tensor.index_select(1, indices)
        return clean_wave, rate

    
    def time_mapper(self, times=None):
        if times is None:
            times = self.get_clean_sections()
        times = np.array(times)
        mod_time = []
        for t in times:
            temp = list(range(int(np.floor(t[0])), int(np.floor(t[1]))))
            temp = [int(np.floor(x)) for x in temp]
            mod_time.extend(temp)
        newtime = list(range(len(mod_time)))
        time_mapper = dict(zip(newtime, mod_time))
        return time_mapper
    
    # def speech_diarization(self, )
    

