import transription_utils

audio_test = transription_utils.audio('data/testrec.wav', threshold=5)
times = audio_test.speechTime
clean = audio_test.cleanedTime





print('TEST', audio_test.speechTime)
print('CLEAN', audio_test.cleanedTime)