import pyaudio
import wave
import keyboard
import time
import os
import whisper
import edge_tts
import asyncio


def record_audio(filename):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    print("开始录音...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    while keyboard.is_pressed('a'):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束...")
    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# 主程序
def record():
    print("Press 'a' to start your conversation, release 'a' when you finished your conversation")
    while True:
        if keyboard.is_pressed('a'):
            filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
            record_audio(filename)
            return filename


def voice_to_text(filepath, modelsize= "medium"):
    _model = whisper.load_model(modelsize)
    _result = _model.transcribe(filepath)
    print(_result["text"])
    return _result


async def text_to_speech(text, output_file):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)


if __name__ == "__main__":
    file = record()
    filepath = "E:\\pythonProject\\VLM\\" + file    # 记得改文件夹路径
    # 检查文件是否已经写入
    while not os.path.exists(filepath):
        time.sleep(0.1)
    print(filepath)
    # filepath = "E:\\pythonProject\\VLM\\2024-07-05_00-07-52.wav"
    model = whisper.load_model("medium")
    result = model.transcribe(filepath, language="en")
    print(result["text"])
    text = result["text"]
    output_file = "output_audio.mp3"
    # 运行TTS转换
    asyncio.run(text_to_speech(text, output_file))
    print(f"Audio file saved as {output_file}")
