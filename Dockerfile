# Here is an example docker file with common commands
# NOTE: This uses alpine:latest as a minimal example.
# Other examples - Replace with an appropriate base image for your project needs:
# - node:lts-alpine for Node.js applications
# - python:3.11-alpine for Python projects
# - golang:alpine for Go applications
# - etc.

# Specify the base image
FROM docker.io/alpine:latest

# Set the working directory
WORKDIR /app

# Copy files from host to container (remember to add
# an appropriate .dockerignore if you keep this or only copy specific files)
COPY . .

# Execute commands during build (e.g. installing dependencies)
RUN apk add --no-cache git

# Example: Set environment variables (add your application-specific variables here)
# ENV MY_VAR=value

# Example: Set build-time arguments
# ARG BUILD_DATE
ARG VERSION=latest
RUN echo "Building version: $VERSION"

# Example: Expose ports (adjust for your application)
# EXPOSE 3000

# Define default command to run the application (replace this with your desired start script)
CMD ["sh", "-c", "echo 'Container started. Add your application commands here.'"]

# Example: Volume management & creating mount points
# VOLUME ["/data"]
# VOLUME /var/log /var/db
