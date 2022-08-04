FROM python:3.7.6-slim

RUN python -m pip install \
    numpy==1.18.1 \
    Flask \
    pickle4 \
    scikit-learn==0.22.1

WORKDIR /app

COPY . /app

EXPOSE 5001
CMD ["python3", "api.py"]
