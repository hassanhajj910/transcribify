import transcription_utils
import torchaudio

audio_test = transcription_utils.audio('data/weights_1.wav')
print('step 1')
times = audio_test.get_clean_sections(threshold=5)
print('step 2')
mapper = audio_test.time_mapper(times)
print('step 3')
clean, rate = audio_test.clean_audio(times)
print('step 4')
speaker_times = audio_test.get_speaker_sections()
print('step 5')
audio_test.transcribe(speaker_times, 'data/weights_fullres.txt')
print('done')

