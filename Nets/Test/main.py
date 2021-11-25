from asrecognition import ASREngine

asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian")

audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]
transcriptions = asr.transcribe(audio_paths)
