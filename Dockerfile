FROM ubuntu:20.04

WORKDIR /app

COPY requirements.txt .

ENV TZ=Africa/Johannesburg
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone    


RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install unixodbc -y \
 && apt-get install unixodbc-dev -y \
 && apt-get install freetds-dev -y \
 && apt-get install freetds-bin -y \
 && apt-get install tdsodbc -y \
 && apt-get install --reinstall build-essential -y

RUN apt-get install curl -y
 
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated msodbcsql17
RUN ACCEPT_EULA=Y apt-get install -y --allow-unauthenticated mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc

RUN apt-get update

RUN apt-get update && apt-get install -y tzdata && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir -r requirements.txt

RUN  rm -vf /var/lib/apt/lists/*
RUN apt-get update

RUN apt-get install git -y

 
RUN apt-get install -y libappindicator1 fonts-liberation  

COPY . .

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

RUN mkdir -p ~/.streamlit/
RUN echo "\
  [server]\n\
  headless = true\n\
  port = $PORT\n\
  enableCORS = false\n\
  \n\
  " > ~/.streamlit/config.toml

RUN echo '[general]\nemail = "a@a.a"' > /root/.streamlit/credentials.toml
RUN echo '[server]\nheadless = true\nenableCORS=false\nport = $PORT' > /root/.streamlit/config.toml

RUN apt-get update
RUN apt-get install unoconv -y   

EXPOSE 8501

# Run the app.
CMD streamlit run --server.port $PORT Introduction.py