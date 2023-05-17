import logging
logging.disable(logging.ERROR)
logging.log(45, "Подгружаем библиотеки")

import numpy as np
import pyaudio as pa
import copy
import wave
import time

import nemo.collections.asr as nemo_asr

import helpers
import vad_init_and_frame


STEP = 0.01
SECOND = 100
WINDOW_SIZE = 0.31
CHANNELS = 1
SAMPLE_RATE = RATE = 16000
FRAME_LEN = STEP
THRESHOLD = 0.575
FILE_NAME_NUMBER = 1
CHUNK_SIZE = int(STEP * RATE)


logging.basicConfig(format='%(message)s')
logging.log(45, "Восстанавливаем ASR модель из памяти и настраиваем")
asr_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
asr_cfg = copy.deepcopy(asr_model._cfg)
asr_model.eval()
asr_model = asr_model.to(asr_model.device)

logging.log(45, "Восстанавливаем VAD модель из памяти и настраиваем")
vad = vad_init_and_frame.get_vad(SAMPLE_RATE, THRESHOLD, FRAME_LEN, WINDOW_SIZE)

known_embeddings = helpers.load_known_embeddings()

p = pa.PyAudio()
empty_counter = 0
speech_frames = 0
background_frames = 0
is_last_frame_background = False
last_record = bytes()


def process_last_frames(transcription, frame):
    """Возвращает True если отрывок речи закончился и помещает его запись в last_record
       False иначе"""
    global speech_frames
    global background_frames
    global last_record
    global is_last_frame_background
    if transcription[1] == "background":
        background_frames += 1
        if speech_frames >= 5 + 3 * SECOND and background_frames >= SECOND or speech_frames >= 60 * SECOND:
            speech_frames = 0
            logging.log(45, "Обрабатываем записанный голос")
            return True
        is_last_frame_background = True
        if background_frames >= SECOND:
            speech_frames = 0
    elif transcription[1] == "speech":
        speech_frames += 1
        is_last_frame_background = False

    if speech_frames >= 5:
        if background_frames != 0 and not is_last_frame_background:
            background_frames = 0
        last_record += frame.tobytes()
    return False


def save_last_record():
    """Сохраняет аудио, находящееся в last_record в виде байтов в .wav файл"""
    global FILE_NAME_NUMBER
    global last_record
    record_name = f"cache/last_speech{FILE_NAME_NUMBER}.wav"
    wf = wave.open(record_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pa.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(last_record)
    wf.close()
    last_record = bytes()
    FILE_NAME_NUMBER += 1
    return record_name


def callback(in_data, frame_count, time_info, status):
    """Метод, вызываемый PyAudio каждый раз, когда есть данные для считывания с микрофона"""
    global empty_counter
    signal = np.frombuffer(in_data, dtype=np.int16)
    text = vad.transcribe(signal)

    if process_last_frames(text, signal):
        record_name = save_last_record()
        current_embedding = asr_model.get_embedding(record_name)
        for known_embedding in known_embeddings:
            if helpers.verify_embeddings(known_embedding, current_embedding, 0.7):
                print("Известный голос")
                break
        else:
            print("Неизвестный голос")

    if len(text):
        empty_counter = vad.offset
    elif empty_counter > 0:
        empty_counter -= 1
        if empty_counter == 0:
            print(' ', end='\n')
    return in_data, pa.paContinue


stream = p.open(format=pa.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                stream_callback=callback,
                frames_per_buffer=CHUNK_SIZE
                )

print('Начинаем слушать...')

try:
    while stream.is_active():
        time.sleep(0.1)
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

    print()
    print("PyAudio stopped")
