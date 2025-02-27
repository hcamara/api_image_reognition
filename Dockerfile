FROM python:3.9

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y ffmpeg
RUN apt-get install -y python3-sklearn
RUN pip install cmake
RUN pip install cython
COPY requirements .

RUN pip install -r requirements

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]