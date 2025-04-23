# 1. Base off Miniconda (small‐ish, but with binary packages)
FROM continuumio/miniconda3

# 2. Copy the environment spec and create it
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# 3. Ensure the environment is activated in subsequent RUN/CMD
SHELL ["conda", "run", "-n", "faceenv", "/bin/bash", "-lc"]

# 4. Copy your app code
COPY . .

# 5. Expose the port Render provides (via $PORT)
EXPOSE 5000

# 6. Run using Gunicorn inside the conda env
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--log-file", "-"]


