# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app

# Copy the model files to the container
COPY models/ /app/models/

# Expose the port that the app runs on
EXPOSE 8050

# Run the application
CMD ["python", "app.py"]
