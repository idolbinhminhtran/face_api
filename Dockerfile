# 1. Base off Miniconda
FROM continuumio/miniconda3

# 2. Install system deps (cmake, compilers, etc)
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libopenblas-dev \
      liblapack-dev \
      libx11-dev \
      libgtk-3-dev \
 && rm -rf /var/lib/apt/lists/*

# 3. Create & activate your Conda env
WORKDIR /app
COPY environment.yml .
# Tell conda-forge to look at dlib/CMake packages
RUN conda config --add channels conda-forge \
 && conda env create -f environment.yml \
 && conda clean -afy

# 4. Make sure all subsequent steps use that env
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-lc"]

# 5. Copy your app code in
COPY . .

# 6. Expose the port Render (or Railway) will set
EXPOSE 5000

# 7. Run with Gunicorn inside the env
CMD ["conda","run","--no-capture-output","-n","faceenv","gunicorn","app:app","--bind","0.0.0.0:$PORT","--log-file","-"]




