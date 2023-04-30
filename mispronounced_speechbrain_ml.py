import os
import pandas as pd
import torch
import soundfile as sf
import torchaudio
from IPython.display import Audio
from speechbrain.pretrained import EncoderDecoderASR, Tacotron2, HIFIGAN
from speechbrain.dataio.dataio import read_audio
from transformers import AutoTokenizer


class MispronouncedWordsDataset:
    def __init__(self, csv_file="mispronounced.csv"):
        self.csv_file = csv_file
        self.asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

    def generate_audio_files(self, output_dir):
        df = pd.read_csv(self.csv_file)
        os.makedirs(output_dir, exist_ok=True)
    
        for i, row in df.iterrows():
            original_word = row["original"]
            output_original_file = os.path.join(output_dir, f"{original_word}.wav")
            self.synthesize_audio(original_word, output_original_file)

            for j in range(1, 30):
                mispronounced_col = f"mispronounced{j}"  # Changed from f"mispronounced_{j}"
                if mispronounced_col in row and isinstance(row[mispronounced_col], str) and len(row[mispronounced_col].strip()) > 0:
                    mispronounced_word = row[mispronounced_col]
                    output_mispronounced_file = os.path.join(output_dir, f"{original_word}/{mispronounced_word}.wav")
                    os.makedirs(os.path.dirname(output_mispronounced_file), exist_ok=True)
                    self.synthesize_audio(mispronounced_word, output_mispronounced_file)


    def synthesize_audio(self, text, output_file):
        if not isinstance(text, str):
            text = str(text)

    # Running the TTS
            mel_output, mel_length, alignment = self.tacotron2.encode_text(text)

    # Adjust dimensions for HiFi-GAN input
            mel_output = mel_output.unsqueeze(0)

    # Pad the input tensor to the required shape
            mel_output_padded = torch.nn.functional.pad(mel_output, (0, self.hifi_gan.gen.config["upsample_scales"][-1] - mel_output.size(-1) % self.hifi_gan.gen.config["upsample_scales"][-1]), "constant", 0)

    # Running Vocoder (spectrogram-to-waveform)
            waveform = self.hifi_gan(mel_output_padded)

            sf.write(output_file, waveform.detach().cpu().squeeze().numpy(), self.tacotron2.hparams.sample_rate)




    def generate_mispronounced_variations(self, text, num_variations=5):
        tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            mel_spec = self.tts_model.generate(tokens_tensor)
            mel_spec_np = mel_spec.squeeze().numpy()
            wav = librosa.feature.inverse.mel_to_audio(mel_spec_np, sr=self.tts_model.hparams.sample_rate)
            transcriptions = self.asr_model.transcribe_multiple(wav, n=num_variations, beam_size=10)
        return transcriptions
        
    def compare_original_and_mispronounced(self, original_word, mispronounced_word):
        original_audio_file = os.path.join("mispronounced_audio", f"{original_word}.wav")
        mispronounced_audio_file = os.path.join("mispronounced_audio", f"{original_word}/{mispronounced_word}.wav")

        original_transcription = self.asr_model.transcribe_file(original_audio_file).strip()
        mispronounced_transcription = self.asr_model.transcribe_file(mispronounced_audio_file).strip()

        if original_transcription != mispronounced_transcription:
            print(f"Mispronounced word detected: {mispronounced_transcription}")
            print(f"Speaking the original word: {original_transcription}")
            self.play_audio(original_audio_file)


    def play_audio(self, audio_file):
        audio_data, sample_rate = sf.read(audio_file)
        return Audio(audio_data, rate=sample_rate)


def main():
    dataset = MispronouncedWordsDataset()
    dataset.generate_audio_files("mispronounced_audio")

    original_word = "example"
    mispronounced_variations = dataset.generate_mispronounced_variations(original_word)
    print(f"Original word: {original_word}")
    print("Generated mispronounced variations:")

    for i, variation in enumerate(mispronounced_variations):
        print(f"{i + 1}. {variation}")
        dataset.compare_original_and_mispronounced(original_word, variation)

if __name__ == "__main__":
    main()
