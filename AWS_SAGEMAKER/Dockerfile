FROM amazonlinux


# Install Python and pip
RUN yum update -y && \
    yum install -y python3 && \
    yum install -y python3-pip && \
    yum install -y python3-setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [ ! -e /usr/bin/python ]; then ln -sf /usr/bin/python3 /usr/bin/python; fi
    
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY --from=jrottenberg/ffmpeg /usr/local /usr/local/

RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y openssl-devel wget && \
    yum update -y && \
    wget https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz && \
    tar -xvzf cmake-3.18.0.tar.gz && \
    cd cmake-3.18.0 && \
    ./bootstrap && \
    make && \
    make install

# COPY ./app /app
# WORKDIR /app
# RUN pip install --no-cache-dir -r /app/requirements.txt

# CMD ["python", "app.py"]




COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install ffmpeg
# RUN yum install -y ffmpeg

COPY . /app/
WORKDIR /app

CMD ["python", "model.py"]
