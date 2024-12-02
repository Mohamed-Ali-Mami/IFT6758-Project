test game id
```
2022030152
```

streamlit access
```
http://localhost:8888/
```

## TO RUN WEB APPLICATION

1. create/activate conda env
    Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
            or Anaconda: https://www.anaconda.com/products/individual


    conda env create --file environment.yml
    conda activate ift6758-m3-env



2. On a different terminal, run the flask server

   python3 app.py



3. In your editor's terminal, run the streamlit app

   streamlit run st_app.py



NOTE: The available models that can be used are:

   distance_model, angle_model, both_model

   All use the comet workspace: rodafs
   All have only one available version: 1.0.0

## Building Docker serving

```
    pip list --format=freeze > requirements.txt
    ./build.sh
```
OR

```
    pip list --format=freeze > requirements.txt
    docker build -t ift6758/serving:1.0.0 -f Dockerfile.serving .
```

## Running Docker serving
```
docker run -it -p 127.0.0.1:8890:8890/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:1.0.0
```

## How to build with docker-compose
```
docker-compose up
```