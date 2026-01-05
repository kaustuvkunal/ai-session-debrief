Getting started
===============


## Activate Virtual Environment

- Mac :
        ```source .venv/bin/activate```
 
 
 - Windows: 
        ```.venv\Scripts\activate ``` 

## Start ML-flow server 

 `
 mlflow ui \
  --backend-store-uri 'file:./mlruns' \
  --default-artifact-root './mlruns/artifacts' \
  --port 5001
  `
## Python path setup
export PYTHONPATH=".:$PYTHONPATH"

