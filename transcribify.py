import transcription_utils
import torchaudio

audio_test = transcription_utils.audio('data/weights_1.wav')
times = audio_test.get_clean_sections(threshold=5)
mapper = audio_test.time_mapper(times)
print(mapper[50])
print(mapper[120])
clean = audio_test.clean_audio(times)
#torchaudio.save('weights_test.wav', clean, audio_test.sample_rate)


