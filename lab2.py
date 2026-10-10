import matplotlib.pyplot as plt
import numpy as np

# Импорт из предыдущей работы
from neuron import sigmoid, relu, step_function


class DataGenerator:
    """Генератор данных для обучения перцептрона"""

    @staticmethod
    def generate_and_data():
        """Генерация данных для логической операции AND"""
        # [input1, input2], target
        training_data = [([0, 0], [0]), ([0, 1], [0]), ([1, 0], [0]), ([1, 1], [1])]
        return training_data

    @staticmethod
    def generate_or_data():
        """Генерация данных для логической операции OR"""
        training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [1])]
        return training_data

    @staticmethod
    def generate_xor_data():
        """Генерация данных для логической операции XOR"""
        training_data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        return training_data


class LossFunctions:
    """Реализация различных функций потерь"""

    @staticmethod
    def mse_loss(predictions, targets):
        """
        Среднеквадратичная ошибка (Mean Squared Error)
        Параметры:
        predictions - предсказания модели
        targets - целевые значения
        Возвращает:
        Значение функции потерь
        """
        return np.mean((predictions - targets) ** 2)

    @staticmethod
    def mse_derivative(predictions, targets):
        """Производная MSE функции потерь"""
        return 2 * (predictions - targets) / len(predictions)

    @staticmethod
    def binary_cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Бинарная перекрестная энтропия
        """
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        return -np.mean(
            targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
        )

    @staticmethod
    def binary_cross_entropy_derivative(predictions, targets):
        """Производная бинарной перекрестной энтропии"""
        return (predictions - targets) / (predictions * (1 - predictions))


class Perceptron:
    """Однослойный перцептрон с различными функциями активации"""

    def __init__(self, num_inputs, activation_function="sigmoid", learning_rate=0.1):
        """
        Инициализация перцептрона
        Параметры:
        num_inputs - количество входных нейронов
        activation_function - тип функции активации
        learning_rate - скорость обучения
        """
        self.weights = np.random.uniform(-0.5, 0.5, num_inputs)
        self.bias = np.random.uniform(-0.1, 0.1)
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.loss_history = []

    def activate(self, x):
        """Применение функции активации"""
        if self.activation_function == "sigmoid":
            return sigmoid(x)
        elif self.activation_function == "relu":
            return relu(x)
        elif self.activation_function == "step":
            return step_function(x)
        elif self.activation_function == "tanh":
            return np.tanh(x)
        else:
            raise ValueError(
                f"Неизвестная функция активации: {self.activation_function}"
            )

    def activate_derivative(self, x):
        """Производная функции активации"""
        if self.activation_function == "sigmoid":
            sig = sigmoid(x)
            return sig * (1 - sig)
        elif self.activation_function == "relu":
            return 1 if x > 0 else 0
        elif self.activation_function == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == "step":
            return 0  # Для пороговой функции производная не определена
        else:
            raise ValueError(
                f"Неизвестная функция активации: {self.activation_function}"
            )

    def forward(self, inputs):
        """Прямой проход"""
        self.last_inputs = np.array(inputs)
        self.weighted_sum = np.dot(self.last_inputs, self.weights) + self.bias
        self.output = self.activate(self.weighted_sum)
        return self.output

    def backward(self, error):
        """Обратное распространение ошибки"""
        # Вычисление градиента
        delta = error * self.activate_derivative(self.weighted_sum)
        # Градиенты для весов и смещения
        grad_weights = delta * self.last_inputs
        grad_bias = delta
        return grad_weights, grad_bias

    def update_weights(self, grad_weights, grad_bias):
        """Обновление весов и смещения"""
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias


class PerceptronTrainer:
    """Класс для обучения перцептрона и визуализации процесса"""

    def __init__(self, perceptron, loss_function="mse"):
        self.perceptron = perceptron
        self.loss_function = loss_function
        self.training_history = []

    def train(self, training_data, epochs=1000, verbose=True):
        """
        Обучение перцептрона
        Параметры:
        training_data - данные для обучения
        epochs - количество эпох
        verbose - вывод прогресса
        """
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            for inputs, target in training_data:
                # Прямой проход
                prediction = self.perceptron.forward(inputs)
                # Вычисление ошибки
                error = prediction - target[0]
                # Вычисление потерь
                if self.loss_function == "mse":
                    loss = LossFunctions.mse_loss(
                        np.array([prediction]), np.array(target)
                    )
                else:
                    loss = LossFunctions.binary_cross_entropy(
                        np.array([prediction]), np.array(target)
                    )
                total_loss += loss
                # Обратное распространение
                grad_weights, grad_bias = self.perceptron.backward(error)
                # Обновление весов
                self.perceptron.update_weights(grad_weights, grad_bias)
                # Подсчет правильных предсказаний
                if round(prediction) == target[0]:
                    correct_predictions += 1
            # Сохранение истории
            accuracy = correct_predictions / len(training_data)
            epoch_history = {
                "epoch": epoch,
                "loss": total_loss / len(training_data),
                "accuracy": accuracy,
                "weights": self.perceptron.weights.copy(),
                "bias": self.perceptron.bias,
            }
            self.training_history.append(epoch_history)
            # Вывод прогресса
            if verbose and epoch % 100 == 0:
                print(
                    f"Эпоха {epoch}: Loss =     {total_loss / len(training_data):.4f}, "
                    f"Accuracy = {accuracy:.2f}"
                )

    def plot_training_progress(self):
        """Визуализация процесса обучения"""
        epochs = [h["epoch"] for h in self.training_history]
        losses = [h["loss"] for h in self.training_history]
        accuracies = [h["accuracy"] for h in self.training_history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # График функции потерь
        ax1.plot(epochs, losses, "b-", linewidth=2)
        ax1.set_title("Функция потерь во время обучения")
        ax1.set_xlabel("Эпоха")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        # График точности
        ax2.plot(epochs, accuracies, "g-", linewidth=2)
        ax2.set_title("Точность во время обучения")
        ax2.set_xlabel("Эпоха")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.show()


def test_perceptron(perceptron, test_data):
    """Тестирование обученного перцептрона"""
    print("\nРезультаты тестирования:")
    print("=" * 40)
    correct = 0
    for inputs, target in test_data:
        prediction = perceptron.forward(inputs)
        predicted_class = 1 if prediction > 0.5 else 0
        is_correct = predicted_class == target[0]

        if is_correct:
            correct += 1
        print(
            f"Вход: {inputs} -> Предсказание: {prediction:.4f} "
            f"({predicted_class}) | Ожидалось: {target[0]} "
            f"{'✓' if is_correct else '✗'}"
        )
    accuracy = correct / len(test_data)
    print(f"\nИтоговая точность: {accuracy:.2f} ({correct}/{len(test_data)})")
    return accuracy


def compare_activation_functions():
    """Сравнение разных функций активации"""
    training_data = DataGenerator.generate_and_data()
    activation_functions = ["sigmoid", "relu", "tanh"]
    results = {}

    for activation in activation_functions:
        print(f"\nОбучение с функцией активации: {activation}")
        print("-" * 40)
        # Создание и обучение перцептрона
        perceptron = Perceptron(
            num_inputs=2, activation_function=activation, learning_rate=0.5
        )
        trainer = PerceptronTrainer(perceptron)
        trainer.train(training_data, epochs=1000, verbose=False)
        # Тестирование
        accuracy = test_perceptron(perceptron, training_data)
        results[activation] = {
            "accuracy": accuracy,
            "final_weights": perceptron.weights,
            "final_bias": perceptron.bias,
            "training_history": trainer.training_history,
        }
        # Визуализация процесса обучения
        trainer.plot_training_progress()

    return results


def main():
    """Основная функция для выполнения лабораторной работы"""

    # 1. Генерация данных
    print("1. Генерация данных для операции AND")
    training_data = DataGenerator.generate_and_data()
    print("Обучающие данные:")
    for inputs, target in training_data:
        print(f" {inputs} -> {target}")

    # 2. Сравнение функций активации
    print("\n2. Сравнение функций активации")
    results = compare_activation_functions()

    # 3. Анализ результатов
    print("\n3. Сравнительный анализ:")
    print("=" * 50)
    for activation, result in results.items():
        print(
            f"{activation:>8}: Accuracy = {result['accuracy']:.2f},",
            f"Weights = {result['final_weights']}, Bias = {result['final_bias']:.4f}",
        )

    # 4. Тестирование на других операциях
    print("\n4. Тестирование на операции OR:")
    or_data = DataGenerator.generate_or_data()
    perceptron_or = Perceptron(
        num_inputs=2, activation_function="sigmoid", learning_rate=0.5
    )
    trainer_or = PerceptronTrainer(perceptron_or)
    trainer_or.train(or_data, epochs=1000, verbose=False)
    test_perceptron(perceptron_or, or_data)

    # 5. Попытка решить XOR (должно не получиться)
    print("\n5. Попытка решить XOR (однослойный перцептрон не может):")
    xor_data = DataGenerator.generate_xor_data()
    perceptron_xor = Perceptron(
        num_inputs=2, activation_function="sigmoid", learning_rate=0.5
    )
    trainer_xor = PerceptronTrainer(perceptron_xor)
    trainer_xor.train(xor_data, epochs=2000, verbose=False)
    test_perceptron(perceptron_xor, xor_data)


if __name__ == "__main__":
    main()
