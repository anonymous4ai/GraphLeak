# Introduction
Code for GraphLeak:Patient Record Leakage through Gradients with
Knowledge Graph (under review in WWW-Web4Good)

# data prepare
 ```bash
    cat data* > data.tar.gz
    tar xzvf data.tar.gz
 ```

# enviroment set up
 ```bash
    conda create -n graphleak python==3.9
    conda activate graphleak
    pip install -r requirements
 ```
## eICU demo
 ```bash
    python main_eicu.py --model mlp --batch-size 32 --matrix-normalize --w1 1e-5 --scale 5
 ```
## MIMIC-III demo
 ```bash
    python main_mimic.py --model mlp --batch-size 32 --matrix-normalize --w1 1e-5 --w2 1e-5 --scale 5
 ```
## config
```bash
    --model :the model you try to attack (mlp, transformer)
    --matrix-normalize :wheather to use graph Loss
    --w1 （--w2） :the coefficient of graph loss
    --scale  :the scale of knowledge graph (5%, 10%, 20%, 50%, 100%)
    --tag   :weather to use TAG Loss
 ```
