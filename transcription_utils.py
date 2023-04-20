import whisper
#import pyannote.audio
#import pyannote.core
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import numpy as np
import torchaudio
import torch
import librosa


key_config = dotenv_values('.env')
SAMPLE_RATE = 16000

def time_overlap(x:list, y:list):
    """
    Find the overlap between two utterances - x and y are two times ranges as list.
    """
    res = list(range(max(int(x[0]), int(y[0])), min(int(x[-1]), int(y[-1]))+1))
    if len(res) > 0:
        over = True
    elif len(res) == 0:
        over = False
    return over



# make functions callable instead of running in the object. 
class audio():
    """
    Class that contains the different functions to manipulate audio files. 
    """
    MODELS_AVAILABLE = {
        'pyannote/voice-activity-detection',
        'pyannote/speaker-diarization@2.1', 
        #'pyannote/segmentation'
    }


    def __init__(self, filepath:str) -> None:
        self.file = filepath
        self.waveform, self.sample_rate = self.load_file()
        
    def load_file(self):
        """
        Loads audio file and converts it into a default SAMPLE_RATE, set at 16000. 
        """
        print('Loading file')
        waveform, sample_rate = torchaudio.load(self.file)
        waveform = waveform[0,:]    # take one channel in cases of stereo audio
        waveform = waveform.unsqueeze(0)
        print('Resample')
        waveform = librosa.resample(np.array(waveform), orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        #waveform, sample_rate = librosa.load(self.file, sr = SAMPLE_RATE)
        return waveform, SAMPLE_RATE

    def get_speaker_sections(self): # can add options here
        """
        get the pyannote/speaker-diarization@2.1 model to get speaker time segments in the audio file. 
        This function returns an array containing three columns; start time, end time, speaker, for each utterance. 
        
        """
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=key_config['HF_PYANNOTE_DIARIZATION'])
        
        diary = pipeline({'waveform':torch.Tensor(self.waveform), 'sample_rate':self.sample_rate})      # run model
        # TODO JSON serialization is removed in version 5.0. Adjust accordingly
        diary = diary.for_json()        # convert to json
        content = diary['content']      # get content from json
        speaker_times, speaker = [[] for x in range(2)] 

        for segment in content:         # for each segment in the content, save the start and end time and label (SPEAKER_XX)
            s = int(np.round(segment['segment']['start']))
            e = int(np.round(segment['segment']['end']))
            speaker = segment['label']
            speaker_times.append([s,e, speaker])
        speaker_times = np.array(speaker_times)
        return speaker_times
    
    def transcribe(self, speaker_times:np.array, outfile:str):
        """
        Run OpenAI Whisper to transcribe each speech section, based on speaker_times from the diarization process. 
        Retuns a text file in outfile dir. 
        """
        model = whisper.load_model("large")
        text = []
        wav = self.waveform         
        wav = torch.Tensor(wav)     # convert to tensor
        wav = wav.squeeze(0)        # fix tensor shape. 
        print(wav.shape)
        with open(outfile, 'w+') as f:
            for speech in speaker_times:
                wav_section = wav[int(speech[0])*SAMPLE_RATE:int(speech[1])*SAMPLE_RATE]        # NB. convert times from second to waverate by multiplying with SAMPLE_RATE. 
                # TODO add language to arguments. 
                res = model.transcribe(wav_section, language = 'en')        # language setting here 
                text.append('{}:{}\n\n'.format(speech[2], res['text']))     # write line
                f.writelines(text[-1])
        


    def get_speech_sections(self):
        """
        Some audio files contain large portions of silence, music, or non-speech section.
        This function uses pyannote/voice-activity-detection to detect the voice activity within a video. 
        The sections that include speech as time intercals (np.array)
        """
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=key_config["HF_PYANNOTE_VOICEACTIVITY"])
        speech_segment = pipeline(self.file)
        speech_segment_tl = speech_segment.get_timeline()
        times = []

        for seg in speech_segment_tl.segments_set_:     # loop over speech segments
            s, e = seg  
            times.append([s,e])
        times = np.array(times)
        times = np.sort(times, axis=0)                  
        return np.array(times)

    def get_clean_sections(self, times=None, threshold=None):
        """
        This function smoothes the cleanspeech times, by adding a padding to avoid cutting speech too soon and affecting the transcription pipeline. 
        Given the times and threshold (int in seconds), this function merges time intervals that are separated by time (in sec) less than the threshold, and returns an ammended time array.
        if no threshold is given, simply returns times. 
        """

        # threshold has to be positive
        if times is None:           # if times is none, simply run speech_secitons. 
            times = self.get_speech_sections().copy()

        if threshold is None:       # if no threshold, then simply return times. 
            return times
        
        if threshold < 0:           # threhsold has to be positive.
            raise ValueError('Threshold must be a positive integer')
         
        # for easier time check. 
        times[:,1] = times[:,1]+int(threshold)
        # to avoid running out of audio-space, last instance is calculated without adding thresh. 
        times[-1][1] = times[-1,1]-int(threshold)
        # get continuous time segments while allocating a threshold buffer.
        init_range = np.array([times[0][0], times[0][1]])

        group = []
        for i in range(len(times)-1):   # avoid last time interval to avoid exceeding file timespan.
            res = time_overlap(times[i], times[i+1])
            if res is True:
                init_range[1] = times[i+1][1]   # if there is overlap (with threshold), merge times
            elif res is False:
                group.append(init_range)        # if there is not overlap, add to group
                init_range = np.array([times[i+1][0], times[i+1][1]])
        if res is True:
            group.append(init_range)
        return np.array(group)      
    
    def clean_audio(self, times:np.array):
        """
        Uses time intervals to return a clean audio waveform file.
        """
        # add exceptions and errors
        # TODO - CHECK if sample_rate is correct here. 
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
        """
        Given that we have an original audio file and a clean one. this function maps the times (sec) from the clean audio file to the original file. 
        Returns a dict where the keys are the origianl times in the audio file, while the values are the new times in the clean audio file
        """
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
    

