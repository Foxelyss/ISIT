import matplotlib.pyplot as plt
from neuron import Neuron, relu, sigmoid, step_function

# Создаем и тестируем нейрон
neuron = Neuron(3, "sigmoid")
inputs = [1, 2, 3]
print(f"Веса нейрона: {[f'{w:.3f}' for w in neuron.weights]}")
print(f"Смещение нейрона: {neuron.bias:.3f}")

output = neuron.forward(inputs)
print(f"Вход: {inputs}, Выход: {output:.3f}")

# Тестируем с разными функциями активации
print("\nСравнение функций активации:")
test_inputs = [0.5, -0.2, 1.0]
neuron_step = Neuron(3, "step")
neuron_relu = Neuron(3, "relu")

print(f"Вход: {test_inputs},")
print(f"Step: {neuron_step.forward(test_inputs)}")
print(f"ReLU: {neuron_relu.forward(test_inputs):.3f}")
print(f"Sigmoid: {neuron.forward(test_inputs):.3f}")

test_inputs = [1, 2, 3]

print(f"Вход: {test_inputs},")
print(f"Step: {neuron_step.forward(test_inputs)}")
print(f"ReLU: {neuron_relu.forward(test_inputs):.3f}")
print(f"Sigmoid: {neuron.forward(test_inputs):.3f}")
fig, ax = plt.subplots()
# Тестируем функции активации
test_values = [-2, -1, 0, 1, 2]
print("Пороговая функция:")

a = []

for x in test_values:
    print(f"step({x}) = {step_function(x)}")
    a.append(step_function(x))

ax.plot(test_values, a, label="Шаговая функция")
a = []

print("\nСигмоида:")
for x in test_values:
    print(f"sigmoid({x}) = {sigmoid(x):.3f}")
    a.append(sigmoid(x))

ax.plot(test_values, a.copy(), label="Сигмойд")
a = []

print("\nReLU:")
for x in test_values:
    print(f"relu({x}) = {relu(x)}")
    a.append(relu(x))

ax.plot(test_values, a, label="ReLu")

plt.legend()
plt.show()
