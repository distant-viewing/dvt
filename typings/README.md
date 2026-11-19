# typings

Stubs for the libraries the annotators call that ship no types: whisperx,
silero-vad, nltk, insightface, onnx2torch, audio-separator, parselmouth,
soundfile, torchaudio and huggingface_hub, whose
two download functions are typed in a way strict mode does not accept. basedpyright reads them in place of the package,
so each declares everything this package uses from its library, and only
that. The signatures are copied from the installed release. When a stage
starts using another name from one of these libraries, add it here.
