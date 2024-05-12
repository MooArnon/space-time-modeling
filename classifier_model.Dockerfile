# Use a base image that includes Python and other dependencies
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your model file and any other necessary files into the container
COPY model.pkl /app/
COPY api/classifier_model_api.py /app/app.py
COPY api/classifier_requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 80

# Command to run your Flask app
CMD ["python", "app.py"]
