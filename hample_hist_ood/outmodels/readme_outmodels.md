### The trained models are saved into this folder.

The first level contains folders with the name of the transcription factors (e.g., GABPA).

The second level's folders in the directory tree's structure are named after the OOD training modifications (e.g., mixin).

The folders' name in the third level are the dates and times for the training runs.

After running training and evaluation, the structure of 'outmodels' is similar to:
```
outmodels/
├── GABPA
│   ├── id
│   │   └── 03_22_21_37_14
│   │       ├── history.csv
│   │       ├── id_0.pth
│   │       ├── id_Indicators.csv
│   │       ├── id_predicted_values.npy
│   │       ├── id.pth
│   │       └── sub_rnd_predicted_values.npy
│   └── mixin
│       └── 03_24_19_30_36
│           ├── history.csv
│           ├── id_predicted_values.npy
│           ├── mixin_.25_0.pth
│           ├── mixin.25_Indicators.csv
│           └── mixin_.25.pth
└── readme.md

```

