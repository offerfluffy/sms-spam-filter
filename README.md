# SMS Spam Filter Project

Aplikacja klasyfikująca wiadomości SMS jako "Spam" lub "Ham" (nie-spam) przy użyciu uczenia maszynowego.

## 📋 Wymagania i Realizacja
Projekt spełnia wszystkie punkty oceny:

1. **Działający program:** Kompletny pipeline od ładowania danych do predykcji.
2. **OOP:** Kod zorganizowany w klasy (`DataLoader`, `TextPreprocessor`, `TextVectorizer`, `ModelTrainer`).
3. **Git:** Historia zmian i repozytorium.
4. **Analiza danych (EDA):** Notebook `notebooks/exploratory_analysis.ipynb` z wykresami i statystykami.
5. **Normalizacja:** Skalowanie cech (`MinMaxScaler`) i kodowanie etykiet (`LabelEncoder`).
6. **Wektoryzacja:** TF-IDF z obsługą n-gramów.
7. **Trenowanie modelu:** Porównanie 3 klasyfikatorów (Naive Bayes, Logistic Regression, Random Forest).
8. **Alternatywne klasyfikatory:** Testowano różne algorytmy i parametry.
9. **Testy jednostkowe:** Pokrycie testami (`pytest`) dla preprocessingu i wektoryzacji.
10. **Analiza wyników:** Generowanie macierzy pomyłek (Confusion Matrix) i krzywych ROC.

## 🚀 Instrukcja Uruchomienia

### 1. Instalacja wymagań
```bash
pip install -r requirements.txt
```

### 2. Uruchomienie głównego programu
Program pobierze dane, przetworzy je, wytrenuje modele i zapisze wyniki.

```Bash
python main.py
```

### 3. Uruchomienie testów
```Bash
python -m pytest
```

### 4. Analiza Danych
Aby zobaczyć wykresy i statystyki:

```Bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 📊 Wyniki
Po uruchomieniu programu, w folderze results/ generowane są wykresy:

- Confusion Matrix: Pokazuje skuteczność wykrywania spamu.
- ROC Curve: Obrazuje jakość klasyfikatora.
- Model Comparison: Porównanie dokładności (Accuracy) wszystkich modeli.

### 📂 Struktura Projektu
```
sms-spam-filter/
├── data/               # Dane surowe i przetworzone
├── notebooks/          # Analiza eksploracyjna (Jupyter)
├── results/            # Wygenerowane wykresy wyników
├── src/                # Kod źródłowy
│   ├── data_loader.py  # Pobieranie i walidacja danych
│   ├── preprocessor.py # Czyszczenie tekstu i inżynieria cech
│   ├── normalizer.py   # Normalizacja i podział na zbiory
│   ├── vectorizer.py   # TF-IDF
│   ├── model_trainer.py# Trenowanie modeli
│   └── evaluator.py    # Wizualizacja wyników
├── tests/              # Testy jednostkowe (pytest)
├── main.py             # Główny plik uruchomieniowy
├── requirements.txt    # Zależności
└── README.md           # Dokumentacja
```

### 🛠 Technologie
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn
- Pytest
