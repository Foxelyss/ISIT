import numpy as np
import matplotlib.pyplot as plt
from lab2 import LossFunctions, DataGenerator  # Импорт из предыдущих работ

# 1. Реализация класса слоя нейронной сети:


class NeuralLayer:
    """Слой нейронной сети"""

    def __init__(self, num_inputs, num_neurons, activation_function="sigmoid"):
        """
        Инициализация слоя

        Параметры:
        num_inputs - количество входов слоя
        num_neurons - количество нейронов в слое
        activation_function - функция активации
        """
        # Инициализация весов методом Xavier/Glorot
        limit = np.sqrt(6 / (num_inputs + num_neurons))
        self.weights = np.random.uniform(-limit, limit, (num_inputs, num_neurons))
        self.biases = np.random.uniform(-0.1, 0.1, num_neurons)
        self.activation_function = activation_function
        self.last_inputs = None
        self.last_output = None
        self.last_weighted_sum = None

    def activate(self, x):
        """Применение функции активации"""
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "linear":
            return x
        else:
            raise ValueError(
                f"Неизвестная функция активации: {self.activation_function}"
            )

    def activate_derivative(self, x):
        """Производная функции активации"""
        if self.activation_function == "sigmoid":
            sig = self.activate(x)
            return sig * (1 - sig)
        elif self.activation_function == "relu":
            return (x > 0).astype(float)
        elif self.activation_function == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == "linear":
            return 1
        else:
            raise ValueError(
                f"Неизвестная функция активации: {self.activation_function}"
            )

    def forward(self, inputs):
        """Прямой проход через слой"""
        self.last_inputs = inputs
        self.last_weighted_sum = np.dot(inputs, self.weights) + self.biases
        self.last_output = self.activate(self.last_weighted_sum)
        return self.last_output


# 2. Реализация многослойной нейронной сети:


class MultiLayerNetwork:
    """Многослойная нейронная сеть"""

    def __init__(self, layers_config, learning_rate=0.1):
        """
        Инициализация сети

        Параметры:
        layers_config - список конфигурации слоев [input_size, hidden1_size, ..., output_size]
        learning_rate - скорость обучения
        """
        self.layers = []
        self.learning_rate = learning_rate

        # Создание слоев
        for i in range(len(layers_config) - 1):
            layer = NeuralLayer(
                layers_config[i],
                layers_config[i + 1],
                activation_function="sigmoid"
                if i < len(layers_config) - 2
                else "sigmoid",
            )
            self.layers.append(layer)

    def forward(self, inputs):
        """Прямой проход через всю сеть"""
        current_output = inputs
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, error):
        """Обратное распространение ошибки"""
        gradients = []
        current_error = error

        # Проход от выходного слоя к входному
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Вычисление дельты для текущего слоя
            if i == len(self.layers) - 1:  # Выходной слой
                delta = current_error * layer.activate_derivative(
                    layer.last_weighted_sum
                )
            else:  # Скрытые слои
                next_layer = self.layers[i + 1]
                delta = np.dot(delta, next_layer.weights.T) * layer.activate_derivative(
                    layer.last_weighted_sum
                )

            # Вычисление градиентов
            grad_weights = np.outer(layer.last_inputs, delta)
            grad_biases = delta

            gradients.insert(0, (grad_weights, grad_biases))
            current_error = delta

        return gradients

    def update_weights(self, gradients):
        """Обновление весов сети"""
        for i, (grad_weights, grad_biases) in enumerate(gradients):
            self.layers[i].weights -= self.learning_rate * grad_weights
            self.layers[i].biases -= self.learning_rate * grad_biases

    def predict(self, inputs, threshold=0.5):
        """Предсказание с пороговым значением"""
        outputs = self.forward(inputs)
        return (outputs > threshold).astype(int)


# 3. Класс для обучения и мониторинга многослойной сети:


class NetworkTrainer:
    """Класс для обучения многослойной сети"""

    def __init__(self, network, loss_function="mse"):
        self.network = network
        self.loss_function = loss_function
        self.training_history = []

    def train(self, training_data, epochs=1000, verbose=True, validation_data=None):
        """
        Обучение сети

        Параметры:
        training_data - данные для обучения
        epochs - количество эпох
        verbose - вывод прогресса
        validation_data - данные для валидации
        """
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0

            for inputs, target in training_data:
                # Прямой проход
                prediction = self.network.forward(inputs)

                # Вычисление ошибки
                error = prediction - target

                # Вычисление потерь
                if self.loss_function == "mse":
                    loss = LossFunctions.mse_loss(
                        np.array([prediction]), np.array([target])
                    )
                else:
                    loss = LossFunctions.binary_cross_entropy(
                        np.array([prediction]), np.array([target])
                    )

                total_loss += loss

                # Обратное распространение
                gradients = self.network.backward(error)

                # Обновление весов
                self.network.update_weights(gradients)

                # Подсчет правильных предсказаний
                predicted_class = 1 if prediction[0] > 0.5 else 0
                if predicted_class == target[0]:
                    correct_predictions += 1

                # Валидация
                val_accuracy = 0
                if validation_data:
                    val_correct = 0
                    for val_inputs, val_target in validation_data:
                        val_prediction = self.network.forward(val_inputs)
                        val_predicted_class = 1 if val_prediction[0] > 0.5 else 0
                        if val_predicted_class == val_target[0]:
                            val_correct += 1
                    val_accuracy = val_correct / len(validation_data)

            # Сохранение истории
            accuracy = correct_predictions / len(training_data)
            epoch_history = {
                "epoch": epoch,
                "loss": total_loss / len(training_data),
                "accuracy": accuracy,
                "val_accuracy": val_accuracy if validation_data else None,
            }
            self.training_history.append(epoch_history)

            # Вывод прогресса
            if verbose and epoch % 100 == 0:
                val_info = (
                    f", Val Accuracy = {val_accuracy:.2f}" if validation_data else ""
                )
                print(
                    f"Эпоха {epoch}: Loss = {
                        total_loss / len(training_data):.4f}, Train Accuracy = {
                        accuracy:.2f}{val_info}"
                )

    def plot_training_progress(self):
        """Визуализация процесса обучения"""
        epochs = [h["epoch"] for h in self.training_history]
        losses = [h["loss"] for h in self.training_history]
        train_accuracies = [h["accuracy"] for h in self.training_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # График функции потерь
        ax1.plot(epochs, losses, "b-", linewidth=2)
        ax1.set_title("Функция потерь во время обучения")
        ax1.set_xlabel("Эпоха")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # График точности
        ax2.plot(epochs, train_accuracies, "g-", linewidth=2, label="Train Accuracy")

        # Добавление валидационной точности, если есть
        if self.training_history[0]["val_accuracy"] is not None:
            val_accuracies = [h["val_accuracy"] for h in self.training_history]
            ax2.plot(epochs, val_accuracies, "r--", linewidth=2, label="Val Accuracy")

        ax2.set_title("Точность во время обучения")
        ax2.set_xlabel("Эпоха")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        ax2.legend()

        plt.tight_layout()
        plt.show()


# 4. Визуализация границ принятия решений:


class DecisionBoundaryVisualizer:
    """Класс для визуализации границ принятия решений"""

    @staticmethod
    def plot_decision_boundary(network, data, title="Граница принятия решения"):
        """Визуализация границы принятия решений нейронной сети"""
        # Создание сетки для визуализации
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
        )

        # Предсказание для каждой точки сетки
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = []

        for point in grid_points:
            pred = network.forward(point)
            predictions.append(pred[0])

        Z = np.array(predictions).reshape(xx.shape)

        # Построение графика
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu, levels=50)
        plt.colorbar(label="Вероятность класса 1")

        # Отображение обучающих данных
        colors = ["blue" if target == 0 else "red" for _, target in data]
        points = [inputs for inputs, _ in data]
        points = np.array(points)

        plt.scatter(
            points[:, 0], points[:, 1], c=colors, s=100, edgecolors="black", alpha=0.8
        )
        plt.xlabel("Признак 1")
        plt.ylabel("Признак 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def compare_architectures(architectures, data):
        """Сравнение разных архитектур сетей"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, arch in enumerate(architectures):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Создание и обучение сети
            network = MultiLayerNetwork(arch, learning_rate=0.5)
            trainer = NetworkTrainer(network)
            trainer.train(data, epochs=2000, verbose=False)

            # Визуализация границ принятия решений
            x_min, x_max = -0.5, 1.5
            y_min, y_max = -0.5, 1.5
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
            )

            grid_points = np.c_[xx.ravel(), yy.ravel()]
            predictions = []

            for point in grid_points:
                pred = network.forward(point)
                predictions.append(pred[0])

            Z = np.array(predictions).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu, levels=50)

            # Отображение данных
            colors = ["blue" if target == 0 else "red" for _, target in data]
            points = [inputs for inputs, _ in data]
            points = np.array(points)

            ax.scatter(points[:, 0], points[:, 1], c=colors, s=80, edgecolors="black")
            ax.set_title(
                f"Архитектура: {arch}\nТочность: {
                    trainer.training_history[-1]['accuracy']:.2f}"
            )
            ax.set_xlabel("Признак 1")
            ax.set_ylabel("Признак 2")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 5. Эксперименты с различными архитектурами:


def experiment_xor_solution():
    """Эксперимент по решению задачи XOR разными архитектурами"""
    xor_data = DataGenerator.generate_xor_data()

    # Различные архитектуры сетей
    architectures = [
        [2, 1],  # Однослойная сеть (не должна решить XOR)
        [2, 2, 1],  # Один скрытый слой с 2 нейронами
        [2, 4, 1],  # Один скрытый слой с 4 нейронами
        [2, 8, 4, 1],  # Два скрытых слоя
        [2, 16, 8, 4, 1],  # Три скрытых слоя
    ]

    results = {}

    print("Эксперимент: Решение задачи XOR разными архитектурами")
    print("=" * 60)

    for arch in architectures:
        print(f"\nАрхитектура: {arch}")
        print("-" * 30)

        # Создание и обучение сети
        network = MultiLayerNetwork(arch, learning_rate=0.5)
        trainer = NetworkTrainer(network)
        trainer.train(xor_data, epochs=3000, verbose=False)

        # Тестирование
        correct = 0
        for inputs, target in xor_data:
            prediction = network.forward(inputs)
            predicted_class = 1 if prediction[0] > 0.5 else 0
            if predicted_class == target[0]:
                correct += 1

        accuracy = correct / len(xor_data)
        final_loss = trainer.training_history[-1]["loss"]

        print(f"Точность: {accuracy:.2f}")
        print(f"Финальный loss: {final_loss:.4f}")
        print(f"Количество эпох до сходимости: {len(trainer.training_history)}")

        results[str(arch)] = {
            "accuracy": accuracy,
            "final_loss": final_loss,
            "epochs_to_converge": len(trainer.training_history),
            "training_history": trainer.training_history,
        }

        # Визуализация для сетей со скрытыми слоями
        if len(arch) > 2:  # Только для сетей со скрытыми слоями
            DecisionBoundaryVisualizer.plot_decision_boundary(
                network, xor_data, title=f"Граница решений (Архитектура: {arch})"
            )

    return results


def analyze_network_behavior():
    """Анализ поведения сети при разных параметрах"""
    xor_data = DataGenerator.generate_xor_data()

    # Эксперимент с разными learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.0]

    plt.figure(figsize=(12, 8))

    for lr in learning_rates:
        network = MultiLayerNetwork([2, 4, 1], learning_rate=lr)
        trainer = NetworkTrainer(network)
        trainer.train(xor_data, epochs=2000, verbose=False)

        losses = [h["loss"] for h in trainer.training_history]
        plt.plot(losses, label=f"LR = {lr}")

    plt.title("Влияние learning rate на сходимость")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Логарифмическая шкала для лучшей визуализации
    plt.show()


# 6. Практическое задание и анализ:


def main():
    """Основная функция для выполнения лабораторной работы"""
    print("Лабораторная работа 3: Многослойная нейронная сеть")
    print("=" * 50)

    # 1. Решение задачи XOR
    print("\n1. Решение задачи XOR")
    xor_data = DataGenerator.generate_xor_data()
    print("Данные XOR:")
    for inputs, target in xor_data:
        print(f"{inputs} -> {target}")

    # 2. Сравнение архитектур
    print("\n2. Сравнение различных архитектур сетей")
    results = experiment_xor_solution()

    # 3. Анализ влияния learning rate
    print("\n3. Анализ влияния скорости обучения")
    analyze_network_behavior()

    # 4. Визуализация границ решений для разных архитектур
    print("\n4. Сравнение границ принятия решений")
    architectures_to_compare = [[2, 2, 1], [2, 4, 1], [2, 8, 4, 1]]
    DecisionBoundaryVisualizer.compare_architectures(architectures_to_compare, xor_data)

    # 5. Создание отчета
    print("\n5. Итоговый отчет:")
    print("=" * 40)
    for arch, result in results.items():
        print(f"Архитектура {arch}")
        print(f" Точность: {result['accuracy']:.2f}")
        print(f" Финальный loss: {result['final_loss']:.4f}")
        print(f" Эпох до сходимости: {result['epochs_to_converge']}")


if __name__ == "__main__":
    main()
