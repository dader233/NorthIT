
# DataPreprocessor

Реализация класса для очистки и преобразования табличных данных (учебное задание).

## Состав

- `data_preprocessor.py` — основной класс
- `test_data_preprocessor.ipynb` — демонстрация работы (ноутбук сгенерирован ИИ)

## Зависимости

```bash
pip install pandas numpy
```

## Пример использования

```python
from data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 30, np.nan, 40],
    'income': [50000, 60000, 55000, np.nan],
    'city': ['Moscow', 'SPB', 'Moscow', 'SPB']
})

prep = DataPreprocessor(df)
result = prep.fit_transform(threshold=0.3, method='minmax')
print(result)
```

## Примечание

Ноутбук для демонстрации был сгенерирован нейросетью. Сам класс `DataPreprocessor` реализован вручную.
