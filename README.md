# Data Preparation & Pipeline Project

Project ini merupakan implementasi data preparation dan data pipeline sederhana pada dataset otomotif menggunakan Python, Pandas, NumPy, dan Scikit-learn.

Pipeline dibuat untuk membaca dataset mentah, melakukan pemeriksaan kondisi data, data cleaning, data transformation, dan menyimpan hasil pengolahan sebagai dataset baru.

## Dataset

Dataset utama yang digunakan adalah:

`automobileEDA_dirty_training.csv`

Dataset berisi informasi kendaraan seperti manufacturer, body style, engine, horsepower, fuel system, harga, konsumsi bahan bakar, dan atribut kendaraan lainnya.

Sumber dataset:

https://s.id/dataset-sesi-3

## Struktur Project

```text
data-pipeline-assignment/
├── data/
│   ├── raw/
│   │   └── automobileEDA_dirty_training.csv
│   └── processed/
│       └── automobileEDA_processed.csv
├── src/
│   └── pipeline.py
├── documentation/
│   └── data-flow-diagram.png
├── README.md
└── requirements.txt
