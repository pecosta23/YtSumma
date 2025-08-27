import yt_dlp
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
from fpdf import FPDF
import torch
import os
from datetime import datetime


#get yt audio 
def audio_download(url, output_path='audio'):
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
    return output_path + '.mp3'

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
    model = Llama(model_path, n_ctx=4096) #context size

    prompt = f"Please resume the next text in principal topics(bullet points), get me the key dots: {text[:6000]}"

    response = model(prompt, max_tokens=1000, temperature=0.5)
    summary = response['choices'][0]['text']
    return summary

#save the summary at a pdf
def save_pdf(summary, pdf_path='resumo_video.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    #tittle
    pdf.cell(200, 10, txt="Summary from the YT video", ln=1, align='C')

    #Date
    current_date = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(200, 10, txt=f"Gerado em {current_date}", ln=1, align='L')

    #Content
    pdf.ln(10)
    for line in summary.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    
    pdf.output(pdf_path)
    print("PDF saved")


#main
if __name__ == "__main__":
    video_url = input("URL of the YT video: ")
    audio_file = audio_download(video_url)
    transcription = audio_transcribe(audio_file)
    summary = text_summa(transcription)
    save_pdf(summary)