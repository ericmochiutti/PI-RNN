---
# try also 'default' to start simple
theme: default
# apply UnoCSS classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
colorSchema: light
---
<style>
.full-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;        /* cobre o slide inteiro */
  z-index: -1;              /* fica atrás do conteúdo */
}
</style>

<img src="./background/90941.jpg" alt="background" class="full-bg" />

# Physics Informed Echo State Networks
<br>
<br>

## Eric Mochiutti  
<br>
<br>
<div class="footer-info">
Universidade de São Paulo  
<br>
<br>
<br>
Novembro de 2025
</div>

---
transition: fade-out
---

# Para que serve uma Rede Neural Recorrente (RNN)?

<v-click>

- **Aproximadores de sistemas dinâmicos**, capazes de representar fenômenos que evoluem ao longo do tempo, como <span v-mark.underline.orange = "4"> **séries temporais**</span>. Cada nova saída depende não apenas da entrada atual,   mas também do que foi apresentado anteriormente.
</v-click>

<v-click>

- **A recorrência** ocorre quando a saída de um passo é realimentada na própria rede, permitindo a formação de uma <span v-mark.circle.green= "5">**memória interna**</span> e o aprendizado de **dependências temporais** (estados ocultos).

</v-click>

<v-click>

<!-- imagem adicionada no final -->
<div class="flex justify-center mt-8">
  <img src="/images/redes.jpg" alt="Diagrama de RNN" class="rounded-2xl shadow-md w-3.5/5">
</div>

</v-click>

<style>

ul {
  font-size: 1.1rem;
  line-height: 1.7rem;
  margin-top: 1.8rem;
}

li {
  margin-bottom: 0.8rem;
  text-align: justify;
}
</style>

---
transition: fade-out
---

# O que são Redes de Estado de Eco (ESN)?

<v-click>

- As ESN são um tipo de RNN, pertencente à família dos modelos de computação de reservatório. O **reservatório** é uma rede recorrente **fixa**, que projeta as entradas em um espaço dinâmico de estados. Na ESN <span v-mark.underline.red = "3">apenas a camada de saída é treinada </span> enquanto as outras são mantidas fixas.
</v-click>

<v-click>
<!-- imagem da arquitetura -->
<div class="flex justify-center mt-6">
  <img src="/images/echostate_architeture.png" alt="Arquitetura de uma Echo State Network" class="rounded-2xl shadow-md w-3/5">
</div>
</v-click>

<style>
ul {
  font-size: 1.1rem;
  line-height: 1.7rem;
  margin-top: 1.8rem;
}

li {
  margin-bottom: 0.8rem;
  text-align: justify;
}
</style>

---
transition: fade-out
---

# E qual a ideia por trás do reservatório?

<v-click>

<div class="flex justify-center items-start gap-8 mt-8">

  <div class="flex flex-col items-center">
    <span class="text-lg font-semibold mb-2">Entrada</span>
    <img src="/images/rocks.jpg" alt="Entrada" class="rounded-xl shadow-md w-36 h-36 object-cover">
  </div>

  <div class="flex flex-col items-center">
    <span class="text-lg font-semibold mb-2">Reservatório</span>
    <img src="/images/lake.gif" alt="Reservatório dinâmico" class="rounded-xl shadow-md w-48 h-36 object-cover">
  </div>

  <div class="flex flex-col items-center">
    <span class="text-lg font-semibold mb-2">Camada de saída</span>
    <img src="/images/camera.jpg" alt="Camada de saída" class="rounded-xl shadow-md w-36 h-36 object-cover scale-x-[-1]">
  </div>

</div>

</v-click>

<v-click>
<!-- imagem da arquitetura -->
<div class="flex justify-center mt-6">
  <img src="/images/echostate_architeture.png" alt="Arquitetura de uma Echo State Network" class="rounded-2xl shadow-md w-2.5/5">
</div>
</v-click>

---
transition: fade-out
---

# Dinâmica do Reservatório

<v-click>

#### Atualização dos estados

$$
\mathbf{x}[n + 1] =
(1-\alpha)\mathbf{x}[n]
+ \alpha f(
\mathbf{W}^{in}\mathbf{u}[n+1]
+ \mathbf{W}\mathbf{x}[n]
+ \mathbf{W}^{fb}\mathbf{y}[n]
+ \mathbf{W}^{b})
$$

</v-click>

<v-click>

<div class="flex justify-center mt-6">
  <img src="/images/projections.jpg" alt="Atualização do estado do reservatório" class="rounded-2xl shadow-lg w-3/4">
</div>

</v-click>

<v-click>

<div class="text-center mt-4 text-sm text-gray-600 italic">
A cada passo temporal, o novo estado <b>x[n+1]</b> combina o estado anterior e a entrada atual,
passando por uma função de ativação não linear.
</div>
</v-click>

<style>
img:hover { transform: scale(1.05); transition: 0.3s; }
</style>

---
transition: fade-out
---

# Espaço de Estados e Dinâmica Interna

<div class="flex flex-row items-center justify-center gap-10 mt-8">

  <!-- Coluna da imagem -->
  <div class="flex justify-center w-1/2">
    <img src="/images/reservoir_projection.png"
         alt="Projeção no espaço de estados"
         class="rounded-2xl shadow-lg w-full max-w-md">
  </div>

  <!-- Coluna do texto -->
  <v-click>
  <div class="text-left text-lg leading-relaxed w-1/2">

  #### Como o reservatório é calculado:
  - O vetor de estados $\mathbf{x}[n] \in \mathbb{R}^{N_r}$ representa a **memória dinâmica** da rede.  
  - O reservatório projeta a entrada $\mathbf{u}[n]$ e a saída $\mathbf{y}[n]$ em um espaço de alta dimensão via $\mathbf{W}^{in}$, $\mathbf{W}$, $\mathbf{Wf}$.  
  - A não-linearidade $f(\cdot)$ (geralmente tanh) cria uma **dinâmica rica**, sensível às entradas passadas.  
  - O parâmetro $\alpha$ controla a **taxa de atualização** (*leaky integration*).

  </div>
  </v-click>

</div>

<style>
img:hover {
  transform: scale(1.02);
  transition: 0.3s;
}
</style>

---
transition: fade-out
---

# Equação de saída e Treinamento da Rede

<v-click>

#### Saída da rede:

$$
\mathbf{y}[n+1] = \mathbf{W}^{out}\mathbf{x}[n+1]
$$

</v-click>

<v-click>

#### Estrutura das matrizes coletadas no treinamento

$$
\mathbf{X} =
\begin{bmatrix}
x_1[1] & \dots & x_1[N_t] \\
\vdots & \ddots & \vdots \\
x_{N_x}[1] & \dots & x_{N_x}[N_t]
\end{bmatrix}
\in \mathbb{R}^{N_x \times N_t}
$$

<v-click>

<div class="text-center text-gray-600 italic mt-4">
Cada coluna representa o estado ou saída em um instante de tempo.
</div>


</v-click>

</v-click>

<v-click>

#### Treinamento do Readout

$$
\mathbf{W}^{out} = \mathbf{\hat Y} \mathbf{X}^\top 
\left( \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I} \right)^{-1}
$$

onde $\lambda$ é o parâmetro de regularização (**Tikhonov**).

</v-click>


<style>
mjx-container {
  font-size: 0.95em;
}
</style>



---
transition: fade-out
---

# Dimensões das Matrizes

::: columns

| Símbolo | Descrição | Dimensão |
|:--|:--|:--|
| $\mathbf{u}[n]$ | Entrada em $n$ | $(N_u, 1)$ |
| $\mathbf{x}[n]$ | Estado do reservatório | $(N_x, 1)$ |
| $\mathbf{y}[n]$ | Saída da rede | $(N_y, 1)$ |
| $\mathbf{W}^{in}$ | Pesos de entrada | $(N_x, N_u)$ |
| $\mathbf{W}$ | Conexões internas | $(N_x, N_x)$ |
| $\mathbf{W}^{fb}$ | Feedback da saída | $(N_x, N_y)$ |
| $\mathbf{W}^{out}$ | Pesos do readout | $(N_y, N_x)$ |
| $\mathbf{X}$ | Estados coletados | $(N_x, N_{t})$ |
| $\mathbf{\hat Y}$ | Saídas desejadas | $(N_y, N_{t})$ |

:::

<style>
table { font-size: 0.85rem; }
mjx-container { font-size: 0.95em; }
</style>


---
transition: fade-out
---

# Modos de Operação da Echo State Network

<div class="grid grid-cols-2 gap-8 mt-6 items-center">

  <!-- Figura 1: Modo de treinamento -->
  <v-click>
  <div class="flex flex-col items-center">
    <span class="text-lg font-semibold mb-2">Modo de Treinamento</span>
    <img src="/images/echostate_trainingmode.png" alt="Modo de Treinamento da ESN" class="rounded-2xl shadow-md w-4/5 hover:scale-105 transition-transform duration-300">
    <p class="text-sm text-gray-600 text-center mt-2 italic">
      Durante o treinamento, a rede utiliza o sinal de saída alvo <b>ŷ[n]</b> (do sistema real) como feedback.
    </p>
  </div>
  </v-click>

  <!-- Figura 2: Modo de execução livre -->
  <v-click>
  <div class="flex flex-col items-center">
    <span class="text-lg font-semibold mb-2">Modo de Execução Livre</span>
    <img src="/images/echostate_freerunmode.png" alt="Modo de Execução Livre da ESN" class="rounded-2xl shadow-md w-4/5 hover:scale-105 transition-transform duration-300">
    <p class="text-sm text-gray-600 text-center mt-2 italic">
      Após o treinamento, a saída prevista <b>y[n]</b> é realimentada ao reservatório — a rede torna-se autônoma.
    </p>
  </div>
  </v-click>

</div>

<style>
img {
  transition: transform 0.3s ease;
}
img:hover {
  transform: scale(1.05);
}
</style>

---
transition: fade-out
---

# Hiperparâmetros da ESN

<br>

<v-click>

- **$N_x$**: Número de neurônios no reservatório  
- **$\rho(\mathbf{W})$**: Raio espectral da matriz de conexões internas  
- **$\alpha$**: Fator de vazamento (taxa de atualização do estado)  
- **$\sigma_{in}$**: Escala dos pesos de entrada $\mathbf{W}^{in}$  
- **$\sigma_{fb}$**: Escala dos pesos de feedback $\mathbf{W}^{fb}$
- **$\sigma_{b}$**: Escala dos pesos de bias $\mathbf{W}^{b}$  
- **$\lambda$**: Parâmetro de regularização do readout (Tikhonov / Ridge)  
- **Densidade do reservatório**: Percentual de conexões não nulas em $\mathbf{W}$  

</v-click>

<v-click>

<div class="text-center text-gray-600 italic mt-4">
Esses hiperparâmetros controlam a dinâmica do reservatório e a capacidade de generalização da rede.
</div>

</v-click>

<style>
mjx-container {
  font-size: 0.95em;
}
</style>

---
transition: fade-out
---
```python
import numpy as np

class ESN:
    """
    This implementation uses fixed recurrent reservoir weights and
    trains only the linear output layer via ridge regression.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of output features.
    train_size : int
        Number of time steps used for training.
    test_size : int
        Number of time steps used for testing.
    warm_up : int
        Number of initial steps discarded during training.
    random_seed : int
        Random seed for reproducibility.
    reservoir_size : int
        Number of reservoir neurons.
    """
```

---
transition: fade-out
---

```python
    def __init__(
        self,
        input_size,
        output_size,
        train_size,
        test_size,
        warm_up,
        random_seed,
        reservoir_size,
    ):
        np.random.seed(random_seed)

        self.input_size = input_size
        self.output_size = output_size
        self.train_size = train_size
        self.test_size = test_size
        self.warm_up = warm_up
        self.reservoir_size = reservoir_size

        # Weight matrices (initialized later)
        self.W = None
        self.Win = None
        self.Wf = None
        self.Wb = None
        self.Wout = None
```

---
transition: fade-out
---

```python
        # Hyperparams
        self.leak_rate = None
        self.reg = None

        # Data placeholders
        self.input_data_train = None
        self.input_data_test = None
        self.output_data_train = None
        self.output_data_test = None
        self.Yt = None

        # Internal states
        self.X_train = None
        self.X_test = None
        self.last_state = None
        self.output_predict = None
```

---
transition: fade-out
---

```python
    def set_data(self, u_train, u_test, train_data, test_data, Yt):

        self.input_data_train = np.asarray(u_train, dtype=np.float64)
        self.input_data_test = np.asarray(u_test, dtype=np.float64)
        self.output_data_train = np.asarray(train_data, dtype=np.float64)
        self.output_data_test = np.asarray(test_data, dtype=np.float64)
        self.Yt = np.asarray(Yt, dtype=np.float64)

    def generate_weights(self, spectral_radius, Win_scaling, Wf_scaling, Wb_scaling):

        # Recurrent weights
        W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        W *= spectral_radius / max_eig
        self.W = W
        # Input weights 
        self.Win = np.random.choice(
            [0, -Win_scaling, Win_scaling],
            size=(self.reservoir_size, self.input_size),
            p=[0.5, 0.25, 0.25],)
        # Feedback and bias
        self.Wf = np.random.uniform(
            -Wf_scaling, Wf_scaling, (self.reservoir_size, self.output_size))
        self.Wb = np.random.uniform(-Wb_scaling, Wb_scaling, self.reservoir_size)
```

---
transition: fade-out
---

```python
    def train_wout(self, leak_rate, reg):

        self.leak_rate = leak_rate
        self.reg = reg
        X = np.zeros((self.reservoir_size, self.train_size))
        x = np.zeros(self.reservoir_size)

        # Run reservoir (teacher forcing)
        for t in range(self.train_size):
            u_t = self.input_data_train[:, t]
            y_t = self.output_data_train[:, t]

            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y_t + self.Wb
            x = (1 - leak_rate) * x + leak_rate * np.tanh(pre_activation)
            X[:, t] = x

        # Discard warm-up
        X_eff = X[:, self.warm_up :]
        Yt_eff = self.Yt

        # Ridge regression: Wout = Y Xᵀ (X Xᵀ + reg I)⁻¹
        A = X_eff @ X_eff.T + reg * np.eye(self.reservoir_size)
        B = Yt_eff @ X_eff.T
        self.Wout = np.linalg.solve(A.T, B.T).T

        self.X_train = X_eff
        self.last_state = x
```

---
transition: fade-out
---

```python
    def predict(self, Wout):
        """
        Run the ESN in free-running (testing) mode.
        """
        x = self.last_state.copy()
        y = self.output_data_train[:, -1].copy()

        Y = np.zeros((self.output_size, self.test_size))
        X_test = np.zeros((self.reservoir_size, self.test_size))

        for t in range(self.test_size):
            u_t = self.input_data_test[:, t]

            pre_activation = self.Win @ u_t + self.W @ x + self.Wf @ y + self.Wb
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
            y = Wout @ x

            X_test[:, t] = x
            Y[:, t] = y

        self.X_test = X_test
        self.output_predict = Y
        return Y
```
---
transition: fade-out
layout: default
---

# Performance da ESN no Sistema Van der Pol

<div class="grid grid-cols-2 gap-4 mt-4 h-full">

  <div class="flex flex-col items-center">
    <h3 class="text-xl font-semibold mb-2 text-center">Dados de Treinamento e Teste</h3>
    <img src="/images/PI_ESN_02/train_test_data.jpg" alt="Dados de Treinamento e Teste Van der Pol" class="rounded-lg shadow-xl w-full" style="max-height: 70vh;">
    <p class="text-sm text-gray-500 mt-2 italic"></p>
  </div>
  
  <div class="flex flex-col items-center">
    <h3 class="text-xl font-semibold mb-2 text-center">Predição da ESN </h3>
    <img src="/images/PI_ESN_02/esn_train_test_vanderpol.jpg" alt="Predição ESN vs Saída do Sistema" class="rounded-lg shadow-xl w-full" style="max-height: 70vh;">
    <p class="text-sm text-gray-500 mt-2 italic"></p>
  </div>

</div>

---
transition: fade-out
layout: default
---

# Physics-Informed Echo State Network (PI-ESN)

A **ESN** é estendida para incorporar **conhecimento físico (ODEs)** na sua função de custo.

## 1. Arquitetura Physics-Informed Echo State Network

A PI-ESN é uma **rede recorrente de tempo discreto** que minimiza:

$$J = \lambda_{data} J_{data} + \lambda_{phy} J_{phy}$$

<div class="grid grid-cols-1 place-items-center mt-6">
  <img src="/images/PI_ESN.jpg" alt="Diagrama da Physics-Informed Echo State Network (PI-ESN)" class="w-1.5/3 rounded-lg shadow-xl">
  <p class="text-sm text-gray-500 mt-3 italic">A ESN é treinada minimizando o custo total ($J$).</p>
</div>


---
transition: fade-out
---
## Cálculo da Perda Física

A lei física é discretizada para calcular a perda em cada, formando a função de resíduo $\mathcal{F}$. O treinamento da PI-ESN utiliza dados no intervalo $[0, T]$ e <span v-mark.underline.red = "2">**pontos de colocação** no intervalo $(T, T_f]$ </span> para a perda física .

<div class="grid grid-cols-1 place-items-center mt-6">
  <img src="/images/Data_collocation_1.jpg" alt="Representação dos pontos de colocação da PI-ESN" class="w-2/3 rounded-lg shadow-xl">
</div>

---
transition: fade-out
---
```python
class PIESN:
    def __init__(self, ESN, dt, cons_phy, colocation_points, subsampling):
        self.ESN = ESN
        self.PI_ESN = deepcopy(ESN)
        self.dt = dt
        self.cons_phy = cons_phy
        self.colocation_points = colocation_points
        self.subsampling = subsampling
        self.cost_data_list = []
        self.cost_physics_list = []
        self.output_prediction_PI = None
        self.MSE_list = []
        self.param_data_list = []
        self.param_phy_list = []
```

---
transition: fade-out
---
```python
    def physical_regularization_vanderpol(self):
        def Wout_to_woutparam(Wout):
            # TRANSFORM THE 2D Wout array into 1D array for LBFGS FUNCTION
            Wout_1 = np.reshape(Wout[0, :], self.PI_ESN.reservoir_size)
            Wout_2 = np.reshape(Wout[1, :], self.PI_ESN.reservoir_size)
            Wout_param = np.hstack((Wout_1, Wout_2))
            return Wout_param

        def woutparam_to_Wout(Wout_param):
            # TRANSFORM THE 1D Wout array into 2D array
            Wout = np.reshape(
                Wout_param[: int(self.PI_ESN.reservoir_size * self.PI_ESN.output_size)],
                (self.PI_ESN.output_size, self.PI_ESN.reservoir_size),
            )
            return Wout
```

---
transition: fade-out
---
```python
        def loss_data(Wout_param):
            y1 = tf.linalg.matvec(
                self.PI_ESN.X_train.T, Wout_param[: self.PI_ESN.reservoir_size]
            )
            y2 = tf.linalg.matvec(
                self.PI_ESN.X_train.T,
                Wout_param[
                    self.PI_ESN.reservoir_size : int(self.PI_ESN.reservoir_size * 2)
                ],
            )

            cost_data_1 = (1 / (self.PI_ESN.train_size)) * tf.experimental.numpy.sum(
                (y1 - self.PI_ESN.Yt[0, :]) ** 2
            )
            cost_data_2 = (1 / (self.PI_ESN.train_size)) * tf.experimental.numpy.sum(
                (y2 - self.PI_ESN.Yt[1, :]) ** 2
            )
            cost_data = (cost_data_1 + cost_data_2) / 2

            return cost_data
```

---
transition: fade-out
---
```python
        def loss_physics(Wout_param):
            u_data_phy = self.PI_ESN.input_data_test[:, : self.PI_ESN.test_size - 1]
            dt_phy = self.dt * self.subsampling

            y1 = tf.linalg.matvec(self.PI_ESN.X_test.T, Wout_param[: self.PI_ESN.reservoir_size])
            y2 = tf.linalg.matvec(
                self.PI_ESN.X_test.T,Wout_param[self.PI_ESN.reservoir_size : int(self.PI_ESN.reservoir_size * 2)],)

            h1 = y1[: self.colocation_points - 1]
            h2 = y2[: self.colocation_points - 1]
            hn1 = y1[1 : self.colocation_points]
            hn2 = y2[1 : self.colocation_points]
            mu = 1

            cost_physics_1 = (h2 * dt_phy - hn1 + h1) ** 2
            cost_physics_2 = ((-mu * (h1**2 - 1) * h2 - h1 + u_data_phy[0, : (self.colocation_points - 1)]) 
            * dt_phy+ h2 - hn2) ** 2

            cost_physics = tf.experimental.numpy.sum(
                cost_physics_1 + cost_physics_2
            ) / (self.PI_ESN.output_size * self.colocation_points)

            return cost_physics
```

---
transition: fade-out
---
```python
        def cost_gradient(Wout_param):
            Wout_param = tf.Variable(Wout_param, trainable=True)

            with tf.GradientTape() as tape:
                cost_data = loss_data(Wout_param)
                self.cost_data_list.append(cost_data)
                cost_physics = loss_physics(Wout_param)
                self.cost_physics_list.append(cost_physics)

                cost = 1 * cost_data + self.cons_phy * cost_physics
                dcost = tape.gradient(cost, Wout_param)

                return cost, dcost
```

---
transition: fade-out
---
```python
        def np_value(tensor):
            if isinstance(tensor, tuple):
                return type(tensor)(*(np_value(t) for t in tensor))
            else:
                return tensor.numpy()

        def run(optimizer):
            optimizer()  # Warmup.
            result = optimizer()
            return np_value(result)

        def cost_gradient_lbfgs():
            init_value = Wout_to_woutparam(self.PI_ESN.Wout)
            return tfp.optimizer.lbfgs_minimize(
                cost_gradient,
                initial_position=init_value,
                tolerance=1e-9,
                max_iterations=100,
                max_line_search_iterations=100,)
```
---
transition: fade-out
---
```python
        for number_X_test_update in range(100):
            results = run(cost_gradient_lbfgs)
            theta = woutparam_to_Wout(results.position)
            self.PI_ESN.Wout = theta
            self.PI_ESN.predict(theta)

            MSE_1 = np.sum(
                ((theta @ self.PI_ESN.X_test)[0, :] - self.ESN.output_data_test[0, :]) ** 2) / (self.PI_ESN.test_size)
            MSE_2 = np.sum(
                ((theta @ self.PI_ESN.X_test)[1, :] - self.ESN.output_data_test[1, :]) ** 2) / (self.PI_ESN.test_size)
            MSE = (MSE_1 + MSE_2) / 2

            MSE_ESN_1 = np.sum(
                ((self.ESN.Wout @ self.ESN.X_test)[0, :] - self.ESN.output_data_test[0, :]) ** 2) / (self.PI_ESN.test_size)
            MSE_ESN_2 = np.sum(
                ((self.ESN.Wout @ self.ESN.X_test)[1, :] - self.ESN.output_data_test[1, :]) ** 2) / (self.PI_ESN.test_size)
            MSE_ESN = (MSE_ESN_1 + MSE_ESN_2) / 2

            self.MSE_list.append(MSE)

        self.output_prediction_PI = theta @ self.PI_ESN.X_test

        return MSE
```

---
transition: fade-out
---

# Resultados PI-ESN vs ESN

- Abrir os HTMLs

