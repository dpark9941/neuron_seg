# neuron_seg

## Document Structure
```
project-root/
├── custom/                 # 모델 아웃풋 (LSD or Affinity) 나올때까지는 여기서 작업합시다
│   ├── funke/              # funke lab training data
│       ├── fib25/          # EM raw data
│           ├── training/   # training data
│   ├── gp/                 # Gunpowder library from Funke lab        
│   ├── log/                # 
│
├── src/                    # 백엔드 및 모델 코드
│   ├── data/               # 데이터 로딩, 전처리
│   ├── models/             # UNet, LSD 모델 등
│   ├── training/           # 학습 스크립트
│   └── inference/          # 추론 코드
│
├── frontend/               # 웹 UI / 시각화 / dashboard 코드
│   ├── public/             # HTML, CSS, assets
│   ├── src/                # React or vanilla JS 코드
│   ├── package.json        # Node 환경 설정
│   └── README.md
│
├── docs/                   # 문서, 회의록, 정리노트
├── References/             # 참조 논문 PDF 등
├── misc/                   # 기타 파일
└── README.md
```