import sys
import logging
logging.disable(logging.ERROR)

import nemo.collections.asr as nemo_asr


def main():
    audio_path1 = sys.argv[1]
    audio_path2 = sys.argv[2]

    #asr_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    asr_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(r"C:\Users\Emin\PycharmProjects\speakerRecognition\pretrained\speakerverification_en_titanet_large.nemo")
    if asr_model.verify_speakers(audio_path1, audio_path2):
        print(1)
    else:
        print(0)


if __name__ == "__main__":
    main()