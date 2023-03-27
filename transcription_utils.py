import whisper
import pyannote.audio
import pyannote.core
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import numpy as np
import torchaudio
import torch
import librosa


key_config = dotenv_values('.env')
SAMPLE_RATE = 16000

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
        self.waveform, self.sample_rate = self.load_file()
        
    def load_file(self):
        waveform, sample_rate = torchaudio.load(self.file)
        waveform = waveform[0,:]    # take one channel in cases of stereo audio
        waveform = waveform.unsqueeze(0)
        waveform = librosa.resample(np.array(waveform), orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        #waveform, sample_rate = librosa.load(self.file, sr = SAMPLE_RATE)
        return waveform, SAMPLE_RATE

    def get_speaker_sections(self): # can add options here
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=key_config['HF_PYANNOTE_DIARIZATION'])
        
        diary = pipeline({'waveform':torch.Tensor(self.waveform), 'sample_rate':self.sample_rate})
        diary = diary.for_json()
        content = diary['content']
        speaker_times, speaker = [[] for x in range(2)]
        for segment in content:
            s = int(np.round(segment['segment']['start']))
            e = int(np.round(segment['segment']['end']))
            speaker = segment['label']
            speaker_times.append([s,e, speaker])
        speaker_times = np.array(speaker_times)
        return speaker_times
    
    def transcribe(self, speaker_times:np.array, outfile:str):
        # load model
        model = whisper.load_model("large")
        text = []
        wav = self.waveform
        wav = torch.Tensor(wav)
        wav = wav.squeeze(0)
        print(wav.shape)
        with open(outfile, 'w+') as f:
            for speech in speaker_times:
                wav_section = wav[int(speech[0])*SAMPLE_RATE:int(speech[1])*SAMPLE_RATE]
                res = model.transcribe(wav_section, language = 'de')
                text.append('{}:{}\n\n'.format(speech[2], res['text']))
                f.writelines(text[-1])
        


    def get_speech_sections(self):

        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=key_config["HF_PYANNOTE_VOICEACTIVITY"])
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
    

