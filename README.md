# Airplane Dataset Downloader

Бұл репозиторий ұшақтарға (airplane) байланысты датасеттерді жүктеп алуға арналған Python скриптін қамтиды.
This repository contains a Python script to download airplane-related datasets.

---

## Датасеттер / Available Datasets

### 1. OpenFlights Aviation Dataset *(default)*

| Файл | Сипаттама | Жолдар |
|------|-----------|--------|
| `airports.csv` | Дүние жүзіндегі 7,000+ әуежай | ~7,698 |
| `airlines.csv` | 6,000+ авиакомпания | ~6,162 |
| `planes.csv` | 250+ ұшақ моделі | ~246 |
| `routes.csv` | 67,000+ рейс маршруты | ~67,663 |

Лицензия: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Source: <https://openflights.org/data.html>

### 2. CIFAR-10 "Airplane" class (image dataset)

CIFAR-10 датасетінің **airplane** класы (label `0`).  
~5 000 airplane суреті (32×32 px, RGB), пиксель мәндері CSV ретінде сақталады.

Source: <https://www.cs.toronto.edu/~kriz/cifar.html>

---

## Пайдалану / Usage

```bash
# OpenFlights датасетін жүктеу (default)
python download_dataset.py

# CIFAR-10 ұшақ суреттерін жүктеу
python download_dataset.py --dataset cifar10

# Өзіңіздің қалтаңызға жүктеу
python download_dataset.py --output ./my_data

# Барлық параметрлер
python download_dataset.py --help
```

Python 3.8+ талап етіледі. Қосымша кітапхана орнату қажет емес (тек стандартты кітапхана).

---

## Нәтиже / Output

```
data/
├── airports.csv   # airport_id, name, city, country, iata, icao, latitude, longitude, ...
├── airlines.csv   # airline_id, name, alias, iata, icao, callsign, country, active
├── planes.csv     # name, iata, icao
└── routes.csv     # airline, airline_id, source_airport, destination_airport, ...
```
