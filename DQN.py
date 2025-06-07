import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from lunar import LunarLanderEnv

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):  
        super(DQN, self).__init__()
        #Definimos las capas de la red neuroal
        #Linear: Capa totalmente conectada (es lo mismo que Dense en Keras)
        self.fc1 = nn.Linear(state_size, hidden_size) #state_size son las entradas, hidden_size las salidas
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)  
        self.fc4 = nn.Linear(hidden_size//2, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) #Función de activación ReLU, la que vimos en clase vaya
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #La última no tiene activación porque la salida de la red son los q-values para cada acción
        #y puede ser negativa o positiva
        return self.fc4(x) 

class ReplayBuffer():
    def __init__(self, buffer_size=10000):
        #Definimos el tamaño del buffer.
        #Usamos deque que es una estructura de datos que se comporta como una pila o cola
        #Es decir, podemos añadir elementos al final o al principio. 
        self.buffer = deque(maxlen=buffer_size) 

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        #Añadimos la experiencia al buffer
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) #Esto del zip es una fumada. Lo explico más abajo

        #El batch es una lista de tupla, y cada tupla es una experiencia:
        # [(state1, action1, reward1, next_state1, done1), (state2, action2, reward2, next_state2, done2), ...]
        #Con zip(*batch) lo que hacemos es "desempaquetar" las tuplas en listas separadas: una lista para states, otra para actions, y así para todas.
        
        # Convertir a numpy arrays primero, luego a tensores
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent():
    def __init__(self, lunar: LunarLanderEnv, gamma=0.99, 
                epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
                learning_rate=0.0003, batch_size=64, 
                memory_size=15000, episodes=1600, 
                target_network_update_freq=5,
                replays_per_episode=1000):
        
        # Inicializamos los hiperparámetros
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        
        self.target_updt_freq = target_network_update_freq
        self.replays_per_episode = replays_per_episode

        # Esto es para saber cuando estamos teniendo éxito o estamos mamando
        self.success_threshold = 200
        self.success_episodes = []
        
        # Creamos el replayBuffer
        self.memory = ReplayBuffer(memory_size)
        
        # Inicializamos el entorno LunarLander
        self.lunar = lunar
        
        observation_space = lunar.env.observation_space
        action_space = lunar.env.action_space
        
        # Esto es para saber si usamos GPU o CPU. A mi con CPU me tira rápido, 2-3 minutos para entrenar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Creamos las redes Q y las objetivas
        hidden_size = 128  # Tamaño recomendado para LunarLander
        self.q_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.target_network = DQN(
            state_size=observation_space.shape[0],
            action_size=action_space.n,
            hidden_size=hidden_size
        ).to(self.device)
        
        # Copiamos los pesos de la red Q a la red objetivo para que empiecen iguales
        self.update_target_network()
        
        # Inicializamos el optimizador. Nos sirve para actualizar los pesos, calcular y limpiar gradientes. 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        print(f"QNetwork:\n {self.q_network}")
        print(f"Device: {self.device}")

    def calculate_success_rate(self, window=100):
        """Calcula el porcentaje de éxito en los últimos episodios"""
        if len(self.success_episodes) < window:
            recent_episodes = self.success_episodes
        else:
            recent_episodes = self.success_episodes[-window:]
        
        if len(recent_episodes) == 0:
            return 0.0
        
        success_count = sum(1 for score in recent_episodes if score >= self.success_threshold)
        return (success_count / len(recent_episodes)) * 100
          
    #Hay dos métodos act() en esta clase, uno para compatibilidad con test_lunar_lander() (que es la función del Jupyter) 
    # y otro para entrenamiento
    def act(self):
        """
        Método sin parámetros para compatibilidad con test_lunar_lander()
        Obtiene el estado actual del entorno y devuelve la acción
        """
        # Obtener el estado actual del entorno
        current_state = self.lunar.state
        
        #Exploración: acción random con probabilidad epsilon
        if np.random.rand() <= self.epsilon:
            action = self.lunar.env.action_space.sample()
        else: #Explotación: acción con mayor Q-value
            # Convertir a numpy array primero si no lo es
            if not isinstance(current_state, np.ndarray):
                current_state = np.array(current_state)
            
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
        return next_state, reward, done, action
    
    #Este método es para entrenamiento. Le metemos el reward shaping para 'penalizar' el uso de motores.
    def act_with_state(self, state):
        """Método usado durante el entrenamiento con reward shaping mejorado"""
        if np.random.rand() <= self.epsilon:
            action = self.lunar.env.action_space.sample()
        else:
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done = self.lunar.take_action(action, verbose=False)
        
            
        # Penalización normal por usar motores
        if action != 0:
            reward -= 0.1  # Penalización ligera por usar motor

        return next_state, reward, done, action

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0 #Si no hay muestras para entrenar, no se entrena

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) #Cogemos experiencias del buffer

        #Movemos los datos al dispositivo que estamos usando
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Valores Q actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Valores Q de la red objetivo
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Función de pérdida, que hay que cambiarla. Me puse con Copilot a fuego y no la he cambiado honestly
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        #Calculamos los gradientes y actualizamos pesos.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    #Actualiza los pesos de la red objetivo con los de la red Q
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    #Guarda el modelo en un archivo
    def save_model(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Modelo guardado en {path}")

    #Carga el modelo desde un archivo
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Modelo cargado desde {path}")
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {path}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    
    def _save_training_data(self, scores, losses):
        """Guarda los datos de entrenamiento en archivos .txt para graficar"""
        import datetime
        
        # Crear timestamp para archivos únicos
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar scores individuales
        with open(f"training_scores_{timestamp}.txt", "w") as f:
            f.write("# Episodio, Score\n")
            for episode, score in enumerate(scores):
                f.write(f"{episode}, {score:.2f}\n")
        
        # Guardar scores promedio móvil (ventana de 50 episodios)
        with open(f"training_avg_scores_{timestamp}.txt", "w") as f:
            f.write("# Episodio, Score_Promedio_50ep\n")
            for episode in range(len(scores)):
                if episode >= 49:  # Empezar cuando tengamos 50 episodios
                    avg_score = np.mean(scores[episode-49:episode+1])
                    f.write(f"{episode}, {avg_score:.2f}\n")
        
        # Guardar losses
        with open(f"training_losses_{timestamp}.txt", "w") as f:
            f.write("# Episodio, Loss\n")
            for episode, loss in enumerate(losses):
                f.write(f"{episode}, {loss:.4f}\n")
        
        # Guardar resumen de métricas
        with open(f"training_summary_{timestamp}.txt", "w") as f:
            final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            successful_episodes = sum(1 for s in scores if s >= 200)
            efficient_episodes = sum(1 for s in scores if s > 0)
            
            f.write("=== RESUMEN DE ENTRENAMIENTO ===\n")
            f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episodios totales: {len(scores)}\n")
            f.write(f"Score promedio final: {final_avg:.2f}\n")
            f.write(f"Mejor score: {max(scores):.2f}\n")
            f.write(f"Peor score: {min(scores):.2f}\n")
            f.write(f"Aterrizajes exitosos (>=200): {successful_episodes} ({successful_episodes/len(scores)*100:.1f}%)\n")
            f.write(f"Aterrizajes eficientes (>0): {efficient_episodes} ({efficient_episodes/len(scores)*100:.1f}%)\n")
            f.write(f"Epsilon final: {self.epsilon:.4f}\n")
        
        print(f"\n📊 Datos de entrenamiento guardados:")
        print(f"   - training_scores_{timestamp}.txt")
        print(f"   - training_avg_scores_{timestamp}.txt") 
        print(f"   - training_losses_{timestamp}.txt")
        print(f"   - training_summary_{timestamp}.txt")


    def train(self):
        scores = []
        losses = []
        
        print("Entrenamiento optimizado para EFICIENCIA DE COMBUSTIBLE")
        print("Objetivo: Score > 200 con uso eficiente de motores")
        print("=" * 60)

        for episode in range(self.episodes):
            state = self.lunar.reset()
            total_reward = 0
            episode_loss = 0
            steps = 0
            
            while True:
                next_state, reward, done, action = self.act_with_state(state)
                
                # Store experience
                self.memory.push(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                
                if len(self.memory) >= self.batch_size and steps % 2 == 0: 
                    loss = self.update_model()
                    episode_loss += loss
                
                if done:
                    break
            
            scores.append(total_reward)
            self.success_episodes.append(total_reward)
            losses.append(episode_loss / max(steps//2, 1))
            
            
            if episode % self.target_updt_freq == 0:
                self.update_target_network()
            
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            
            success_rate = self.calculate_success_rate()
            
            
            efficient_landings = sum(1 for s in self.success_episodes[-100:] if s > 0) if len(self.success_episodes) >= 100 else sum(1 for s in self.success_episodes if s > 0)
            efficiency_rate = (efficient_landings / min(len(self.success_episodes), 100)) * 100
            
            
            if episode % 50 == 0:
                avg_score = np.mean(scores[-50:]) if len(scores) >= 100 else np.mean(scores)
                avg_loss = np.mean(losses[-50:]) if len(losses) >= 100 else np.mean(losses)
                
                print(f"Ep {episode:4d} | "
                      f"Score promedio ult. 50 episodios: {avg_score:7.2f} | "
                      f"Último: {total_reward:6.1f} | "
                      f"Éxito: {success_rate:4.1f}% | "
                      f"Eficiente: {efficiency_rate:4.1f}% | "
                      f"Loss promedio ult. 50 episodios: {avg_loss:7.2f} | "
                      f"ε: {self.epsilon:.4f}")
                
        
        self._save_training_data(scores, losses)

        # Guardar modelo final
        final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        final_efficient = sum(1 for s in scores[-100:] if s > 0) if len(scores) >= 100 else sum(1 for s in scores if s > 0)
        
        self.save_model(f"modelo_DQN.h5")
        
        # Estadísticas finales mejoradas
        print("\n" + "=" * 60)
        print("RESULTADOS FINALES:")
        print("=" * 60)
        print(f"Episodios completados: {len(scores)}")
        print(f"Score promedio final: {final_avg:.2f}")
        print(f"Mejor score: {max(scores):.2f}")
        print(f"Aterrizajes exitosos (>200): {sum(1 for s in scores if s >= 200)}")
        print(f"Aterrizajes eficientes (>0): {final_efficient}")
        print(f"Porcentaje de eficiencia: {(final_efficient/min(len(scores), 100))*100:.1f}%")
        print(f"Objetivo eficiencia: {'✅ ALCANZADO' if final_avg >= 150 else '❌ NO ALCANZADO'}")
        print("=" * 60)

        return scores, losses