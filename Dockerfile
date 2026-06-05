FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb-shm0 libxcb-xfixes0 libxcb-shape0 libxcb-xinerama0 \
    libxcb-randr0 libxcb-image0 libxcb-keysyms1 libxcb-icccm4 \
    libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
