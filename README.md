# Self-Refining DFT

## Setup
Define the environment variables in `.env.example` file and rename to `.env`

## Install
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create --file environment.yaml
```

## Train
```
python src/train.py experiment=train/srt/ethanol-100
```

## Evaluation
```
python src/eval.py experiment=eval/srt/ethanol-100

```