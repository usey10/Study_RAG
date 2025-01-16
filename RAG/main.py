from dotenv import load_dotenv

load_dotenv()

from graph import app

if __name__ == "__main__":
    inputs = {"question": "iso 설정 방법에 대해 알려줘"}

    for chunk_msg, metadata in app.stream(inputs, stream_mode="messages"):
            print(chunk_msg.content, end="", flush=True)