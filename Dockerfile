FROM python:3.9
EXPOSE 8501
WORKDIR /app
COPY requirements_streamlit.txt ./requirements_streamlit.txt
RUN python -m pip install --upgrade pip
RUN pip install -r requirements_streamlit.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0"]