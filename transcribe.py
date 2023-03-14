import whisper
import pyannote.audio
import pyannote.core
from pyannote.audio import Pipeline

class audio():
    def __init__(self, filepath:str) -> None:
        self.file = filepath
        self.pipeline = self.set_pipeline()
    

    def set_pipeline(self): # can add options here
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")
        return pipeline
    
    