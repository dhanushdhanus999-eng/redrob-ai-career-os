"""Hugging Face Spaces entry point."""

from app.demo import build_demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_port=7860)

