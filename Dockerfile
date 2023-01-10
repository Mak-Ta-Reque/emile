FROM python:3.9
EXPOSE 8501
WORKDIR /app

#COPY requirements_streamlit.txt ./requirements_streamlit.txt
#RUN python -m pip install --upgrade pip


RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements_streamlit.txt
#COPY . .
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]