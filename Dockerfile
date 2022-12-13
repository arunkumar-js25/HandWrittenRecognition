# base image  
FROM python:3

# Sets an environmental variable that ensures output from python is sent straight to the terminal without buffering it first
ENV PYTHONUNBUFFERED 1

# Sets the container's working directory to /app
RUN mkdir /app
WORKDIR /app

# runs the pip install command for all packages listed in the requirements.txt file
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copies all files from our local project into the container
COPY . /app/

# port where the Django app runs  
EXPOSE 8000  

# start server  
CMD python manage.py runserver  