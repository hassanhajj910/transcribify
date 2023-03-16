import transcription_utils

audio_test = transcription_utils.audio('data/weights_1.wav')
print(audio_test.file)
times = audio_test.get_clean_sections(threshold=5)
mapper = audio_test.time_mapper(times)
