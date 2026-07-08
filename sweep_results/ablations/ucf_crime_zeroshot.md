# UCF-Crime cross-dataset zero-shot evaluation

Dataset: UCF-Crime (Anomaly subset, partial)
Classes evaluated: ['Burglary', 'Fighting', 'RoadAccidents']  (total 267 clips)

## Overall
- pred dist: {'WalkwayViol': 7, 'UnauthorInt': 42, 'OpenedPanel': 10, 'CarryOverload': 208}
- frac → UnauthorInt: 0.157 (uniform = 0.25)
- mean max-prob: 0.747
- frac high-conf (max-p>0.5): 0.880
- in-domain Unsafe-Net mean max-prob: 0.8042

## Per UCF-Crime class

| UCF class | N | Walk | Unauth | Panel | Forklift | frac→Unauth | mean p | high-conf |
|-----------|--:|-----:|-------:|------:|---------:|------------:|-------:|----------:|
| Burglary | 100 | 3 | 20 | 4 | 73 | 0.200 | 0.757 | 0.880 |
| Fighting | 50 | 1 | 9 | 0 | 40 | 0.180 | 0.764 | 0.880 |
| RoadAccidents | 117 | 3 | 13 | 6 | 95 | 0.111 | 0.732 | 0.880 |
