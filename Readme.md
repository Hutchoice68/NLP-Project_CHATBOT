# Intent project with line and web platform

## Create Virtual Environment

```
python -m venv venv
```
where, `venv` is your virtual environment name

## Activate Virtual Environment
```
venv\Scripts\activate
```

## Install Python Package
To install , run `pip` command to install 
```
pip install -r requirements.txt
```

## Environment Configuration
Create file `.env` and put your token inside.

```
ACCESS_TOKEN=<Your dhannel access token>
CHANNEL_SECRET=<Your Channel secret>
```

## Running FastAPI
Using following command to run FastAPI on port 8000:
```
uvicorn main:app --port 8000 --reload
```
The `reload` flag is for reload everytime that file made change.
