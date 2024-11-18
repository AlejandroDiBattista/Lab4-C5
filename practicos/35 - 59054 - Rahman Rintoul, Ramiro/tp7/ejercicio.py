import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

data = pd.read_csv('ventas.csv')

ventas = data['ventas'].values
ventas_min = np.min(ventas)
ventas_max = np.max(ventas)
ventas_normalizadas = (ventas - ventas_min) / (ventas_max - ventas_min)
data['ventas_normalizadas'] = ventas_normalizadas

x = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas_normalizadas'].values, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurones en la capa oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

if train_button:
    model = SimpleNN(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    progress_bar = st.progress(0)
    loss_history = []

    for epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    st.success("Entrenamiento finalizado")

    torch.save(model.state_dict(), 'modelo_ventas.pth')


    fig, ax = plt.subplots()
    ax.plot(loss_history, label="Pérdida")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Pérdida")
    ax.set_title("Evolución de la función de costo")
    st.sidebar.pyplot(fig)  

    with torch.no_grad():
        predictions = model(x).numpy()
        predictions = predictions * (ventas_max - ventas_min) + ventas_min  
        actual_sales = data['ventas'].values  

    fig, ax = plt.subplots()
    ax.plot(data['dia'], actual_sales, label="Ventas Reales", color="blue")
    ax.plot(data['dia'], predictions, label="Predicción de la Red", color="red")
    ax.set_xlabel("Día del mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Ventas diarias y Predicción de la Red Neuronal")
    ax.legend()
    st.pyplot(fig)
