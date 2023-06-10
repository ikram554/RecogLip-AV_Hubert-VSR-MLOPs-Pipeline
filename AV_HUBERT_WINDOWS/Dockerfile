# Base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the code and data files to the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Set the entry point command
CMD ["python", "av_hubert.py"]
