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

Sample test of current MemoryBank implementation:
```
python MemoryBank.py
```

Sample test of current util functions translate_text and translate_conllu:
```
python utils.py
```

To visualize results, run:
```
tensorboard --logdir=runs
```
And open http://localhost:6006/

