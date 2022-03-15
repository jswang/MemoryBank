# MemoryBank
This is our implementation of MemoryBank, a Question Answering system with improved consistency and accuracy.
This code runs with Python 3.9 on a google cloud A2 compute instance.

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

To adjust batch size:
```
python run.py --batch_size 10
```

To visualize results, run:
```
tensorboard --logdir=runs
```
And open http://localhost:6006/
