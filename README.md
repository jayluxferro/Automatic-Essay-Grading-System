## Automatic Essay Grading System
*NB:* The code base was developed using python 2.7.17

### Instructions
1. Create a virtualenv
```
python -m virtualenv .venv
```

2. Activate virtualenv
```
source .venv/bin/activate
```

3. Install python dependencies
```
pip install -r requirements.txt
```

4. Train model
```
python train.py
```
You can choose any model; defined in `models.py`.


### Proof-of-Concept
Visit https://aegs-ai.herokuapp.com

You can also test it locally. Use the commands below:
```
cd ui
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
Open http://localhost:8000
