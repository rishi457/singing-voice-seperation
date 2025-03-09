import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import os

def separate_sources(song_filenames, output_directory='demo'):
    """Loads a song, processes it, and separates vocals from music."""
    
    # Load the trained model
    model = load_model("C:/Users/mishr/OneDrive/Desktop/BigData/abc/Singing-Voice-Separation-RNN/model/svsrnn.keras")

    # Check if output directory exists, if not create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for song_filename in song_filenames:
        print(f"Processing: {song_filename}")
        
        # Load the song
        y, sr = librosa.load(song_filename, sr=22050, mono=True)
        
        # Compute Short-Time Fourier Transform (STFT) with n_fft=1024 to get 513 frequency bins
        stft_mono = librosa.stft(y, n_fft=1024, hop_length=512)
        magnitude, phase = np.abs(stft_mono), np.angle(stft_mono)

        # Normalize magnitude for model input
        stft_mono_magnitude = magnitude / np.max(magnitude)

        # Ensure the correct shape: (1, time_steps, 513)
        stft_mono_magnitude = np.expand_dims(stft_mono_magnitude.T, axis=0)

        # Predict voice and music components
        predictions = model.predict(stft_mono_magnitude)

        # Check if the model returns a single output or two outputs
        if isinstance(predictions, list) and len(predictions) == 2:
            y1_pred, y2_pred = predictions
        elif isinstance(predictions, np.ndarray) and predictions.shape[-1] == magnitude.shape[0]:
            # In case the model returns a single output, split into two components
            y1_pred = predictions
            y2_pred = np.zeros_like(y1_pred)  # If no accompaniment is provided, assume silence
        else:
            raise ValueError(f"Unexpected model output type: {type(predictions)} or shape: {predictions.shape}")

        # Remove the singleton dimension from y1_pred and y2_pred
        y1_pred = np.squeeze(y1_pred)
        y2_pred = np.squeeze(y2_pred)

        # Convert predictions back to waveform using inverse STFT
        voice_stft = y1_pred.T * np.exp(1j * phase)
        music_stft = y2_pred.T * np.exp(1j * phase)

        voice_waveform = librosa.istft(voice_stft, hop_length=512)
        music_waveform = librosa.istft(music_stft, hop_length=512)

        # Save separated audio
        song_name = os.path.splitext(os.path.basename(song_filename))[0]  # Remove file extension
        voice_output_path = os.path.join(output_directory, f"{song_name}_voice.wav")
        music_output_path = os.path.join(output_directory, f"{song_name}_music.wav")
        
        sf.write(voice_output_path, voice_waveform, sr)
        sf.write(music_output_path, music_waveform, sr)

        print(f"Saved: {voice_output_path} and {music_output_path}")

if __name__ == '__main__':
    songs_dir = 'songs'  # Specify the directory containing the song files
    song_filenames = [os.path.join(songs_dir, f) for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    separate_sources(song_filenames=song_filenames, output_directory='demo')
