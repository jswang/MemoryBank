# cs224n_final_project
Final project for CS224N

To pull BeliefBank data:
```
./get_beliefbank.sh
```

Set up virtual environment with dependencies:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run Memory bank on validation dataset
```
python run.py --mode 'val'
```

Run Memory bank on test dataset
```
python run.py --mode 'test'
```

To visualize results, run:
```
tensorboard --logdir=runs
```
And open http://localhost:6006/

