import yt_dlp
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

#get yt audio 
def audio_download(url, output_path='audio.mp3'):
    mp3_config = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192' #kbps
        }],
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(mp3_config) as ydl:
        ydl.download([url])
    return output_path

#transcribes the audio
def audio_transcribe(audio_path, model_name='small'):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    transcription = result['text']
    os.remove(audio_path)
    return transcription

#summarize in topics (Llama)
def text_summa(text, model_path='./models/llama-2-7b-chat.Q4_0.gguf'):
    from llama_cpp import Llama
    model = Llama(model_path, n_ctx=2048) #text size

    prompt = f"Please resume the next text in principal topics(bullet points), get me the key dots: {text[:4000]}"

    response = model(prompt, max_tokens=500, temperature=0.5)
    summary = response['choices'][0]['text']
    return summary

#main
if __name__ == "__main__":
    