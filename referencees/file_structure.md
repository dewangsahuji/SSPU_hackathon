fitoproto_web/
├── fitobot.py
├── app.py                      ← main Flask entry point
│
├── core/
│   └── utils.py                ← shared math + CSV logging helpers
│
├── exercises/
│   ├── bicep/
│   │   └── bicep_tracker.py    ← contains detection + visualizer
│   │
│   ├── squat/
│   │   └── squat_tracker.py
│   │
│   ├── pushup/
│   │   └── pushup_tracker.py
│
├── templates/
│   ├── index.html              ← exercise selector page
│   ├── exercise.html           ← live video display page
│
└── static/
    ├── css/
    └── js/




