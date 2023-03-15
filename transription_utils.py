import whisper
import pyannote.audio
import pyannote.core
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import numpy as np

key_config = dotenv_values('.env')

class audio():
    MODELS_AVAILABLE = {
        'pyannote/voice-activity-detection',
        'pyannote/speaker-diarization@2.1', 
        'pyannote/segmentation'
    }

    def __init__(self, filepath:str, threshold=None) -> None:
        self.file = filepath
        self.pipeline = self.set_pipeline()
        self.speechTime = self.get_speech_sections()
        self.cleanedTime = self.remove_silence(thresh=threshold)
    

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
        
        return times

    def time_overlap(x, y):
        res = list(range(max(int(x[0]), int(y[0])), min(int(x[-1]), int(y[-1]))+1))
        if len(res) > 0:
            over = True
        elif len(res) == 0:
            over = False
        return over


    def remove_silence(self, thresh:None):
        # threshold has to be positive
        times = self.speechTime.copy()

        if thresh is None:
            return times
        
        if thresh < 0:
            raise ValueError('Threshold must be a positive integer')
        
        # for easier time check. 
        times[:,1] = times[:,1]+int(thresh)
        # to avoid running out of audio-space, last instance is calculated without adding thresh. 
        times[-1][1] = times[-1,1]-int(thresh)
        # get continuous time segments while allocating a threshold buffer.
        init_range = np.array([times[0][0], times[0][1]])
        group = []
        for i in range(len(group)-1):
            res = time_overlap(times[i], times[i+1])
            if res is True:
                init_range[1] = times[i+1][1]
            elif res is False:
                group.append(init_range)
                init_range = np.array([times[i+1][0], times[i+1][1]])
        return group
            



            

        
        