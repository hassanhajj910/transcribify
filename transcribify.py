import transcription_utils
#import torchaudio
import argparse
import time
import os
import re

parser = argparse.ArgumentParser(description="pass files")
parser.add_argument('-i', '--input', type=str,  help='Path to input folder with WAV file', required=True)
parser.add_argument('-o', '--output', type=str, help='Path to output folder to store text file', required=True)
args = vars(parser.parse_args())

RE_EXP = '^(.+)\.wav$'
 
start_time = time.time()
allwav = os.listdir(args['input'])

for wavfile in allwav:
    print('-- Processing file: {}'.format(wavfile))
    iter_start = time.time()
    fpath = os.path.join(args['input'], wavfile)
    audio_test = transcription_utils.audio(fpath)
    print('step 1')
    times = audio_test.get_clean_sections(threshold=5)
    print('step 2')
    mapper = audio_test.time_mapper(times)
    print('step 3')
    clean, rate = audio_test.clean_audio(times)
    print('step 4')
    speaker_times = audio_test.get_speaker_sections()
    print('step 5')

    se = re.search(RE_EXP, wavfile)
    textname = str(se.group(1)) + '.txt'
    textname = os.path.join(args['output'], textname)
    audio_test.transcribe(speaker_times, textname)
    iter_end = time.time()
    iter_time = iter_end - iter_start
    iter_time = iter_time/60
    print('------ Processed file: {} in {} minutes'.format(wavfile, iter_time))


end_time = time.time()
process_time = end_time - start_time
process_time = process_time/60
print('Processing completed in {} minutes'.format(process_time))

