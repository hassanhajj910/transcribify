import transcription_utils
import torchaudio
import argparse
import time

parser = argparse.ArgumentParser(description="pass files")
parser.add_argument('-i', '--input', type=str,  help='Path to the WAV file', required=True)
parser.add_argument('-o', '--output', type=str, help='Path to output text file', required=True)
args = vars(parser.parse_args())

start_time = time.time()

audio_test = transcription_utils.audio(args['input'])
print('step 1')
times = audio_test.get_clean_sections(threshold=5)
print('step 2')
mapper = audio_test.time_mapper(times)
print('step 3')
clean, rate = audio_test.clean_audio(times)
print('step 4')
speaker_times = audio_test.get_speaker_sections()
print('step 5')
audio_test.transcribe(speaker_times, args['output'])
end_time = time.time()
process_time = end_time - start_time
process_time = process_time/60
print('done - process took {} minutes'.format(process_time))

