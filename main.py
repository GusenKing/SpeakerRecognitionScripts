import pyaudio as pa
import runpy


def main():
    p = pa.PyAudio()
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)

    if len(input_devices):
        runpy.run_path("VAD_streaming_inference.py")
    else:
        print('ERROR: No audio input device found.')


if __name__ == '__main__':
    main()
