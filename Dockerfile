# Use the official Python image from Docker Hub
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /application

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application into the container
COPY . .

# Expose the port that your Streamlit app will run on
EXPOSE 8501

# Specify the command to run on container start
CMD ["streamlit", "run", "application/malin.py"]
