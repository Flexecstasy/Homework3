
# Эксперименты по архитектуре полносвязных сетей

## Структура
- `homework_depth_experiments.py`: сравнение глубин сетей
- `homework_width_experiments.py`: сравнение ширины слоев и grid search
- `homework_regularization_experiments.py`: сравнение техник регуляризации

## Запуск
```bash
python homework_depth_experiments.py --input_size 20 --num_classes 2 --epochs 20 --device cpu
python homework_width_experiments.py --input_size 20 --num_classes 2 --epochs 20 --device cpu
python homework_regularization_experiments.py --input_size 20 --num_classes 2 --epochs 20 --device cpu
```

## Результаты
Графики сохранены в `plots/`, логи в `results/`.

