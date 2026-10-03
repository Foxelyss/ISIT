import math
import random


def weighted_sum(inputs, weights, bias=0):
    """
    Вычисляет взвешенную сумму входных сигналов
    Параметры:
    inputs - список входных значений [x1, x2, ..., xn]
    weights - список весов [w1, w2, ..., wn]
    bias - смещение (по умолчанию 0)
    Возвращает:
    Взвешенную сумму inputs * weights + bias
    """
    if len(inputs) != len(weights):
        raise ValueError("Количество входов и весов должно совпадать")

    total = bias

    for i in range(len(inputs)):
        total += inputs[i] * weights[i]

    return total


def step_function(x, threshold=0):
    """
    Пороговая функция активации
    Параметры:
    x - входное значение
    threshold - порог срабатывания
    Возвращает:
    1 если x >= threshold, иначе 0
    """
    return 1 if x >= threshold else 0


def sigmoid(x):
    """
    Сигмоидальная функция активации
    Параметры:
    x - входное значение
    Возвращает:
    Значение в диапазоне от 0 до 1
    """
    return 1 / (1 + math.exp(-x))


def relu(x):
    """
    Функция активации ReLU (Rectified Linear Unit)
    Параметры:
    x - входное значение
    Возвращает:
    max(0, x)
    """
    return max(0, x)


class Neuron:
    def __init__(self, num_inputs, activation_function="sigmoid"):
        """
        Инициализация нейрона
        Параметры:
        num_inputs - количество входных сигналов
        activation_function - тип функции активации
        """
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        self.activation_function = activation_function

    def forward(self, inputs):
        """
        Прямое распространение сигнала
        Параметры:
        inputs - список входных значений
        Возвращает:
        Выходное значение нейрона после применения функции
        активации
        """
        if len(inputs) != len(self.weights):
            raise ValueError("Количество входов не соответствует количеству весов")
        # Вычисляем взвешенную сумму
        z = weighted_sum(inputs, self.weights, self.bias)
        # Применяем функцию активации
        if self.activation_function == "step":
            return step_function(z)
        elif self.activation_function == "sigmoid":
            return sigmoid(z)
        elif self.activation_function == "relu":
            return relu(z)
        else:
            raise ValueError("Неизвестная функция активации")
