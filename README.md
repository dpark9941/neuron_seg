# neuron_seg

## Document Structure
```
project-root/
├── custom/                 # Workspace until model output (LSD or Affinity) is ready
│   ├── funke/              # funke lab training data
│       ├── fib25/          # EM raw data
│           ├── training/   # training data
│   ├── gp/                 # Gunpowder library from Funke lab        
│   ├── log/                # 
│
├── src/                    # Backend and model code
│   ├── data/               # Data loading, preprocessing
│   ├── models/             # UNet, LSD models, etc.
│   ├── training/           # Training scripts
│   └── inference/          # Inference code
│
├── frontend/               # Web UI / visualization
│   ├── public/             # HTML, CSS, assets
│   ├── src/                # React or vanilla JS code
│   ├── package.json        # Node environment configuration
│   └── README.md
│
├── docs/                   # Documentation, meeting minutes, notes
├── References/             # Reference papers, PDFs, etc.
├── misc/                   # Miscellaneous
└── README.md
```
