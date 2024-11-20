
[![Build And Test](https://github.com/satyasundar/erav3-s5/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/satyasundar/erav3-s5/actions/workflows/ml-pipeline.yml)


## A5-CICD-Model-Pipeline
This is a simple MNIST model using PyTorch to test the accuracy of 95% in one epoch. Also it will setup a CI/CD pipeline in GitHub Actions.

### Project Structure:
- `model.py`: Defines the MNIST model.
- `train.py`: Trains the model.
- `predict.py`: Predicts the model.
- `requirements.txt`: Lists the dependencies.
- `README.md`: This file.

```
A5-Model-CICD/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── model.py
├── train.py
├── test_model.py
└── requirements.txt
```


### To Run the Project Locally:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#Run the training script
python train.py

#Run the test
pytest test_model.py -v
```

### To deploy to GitHub:
```
git init
git add .
git commit -m "initial commit"
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
``` 
