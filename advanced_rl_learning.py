import numpy as np
import random
import json
import os
from collections import deque
import heapq
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

class PrioritizedReplayBuffer:
    """Буфер воспроизведения с приоритетами для важных опытов"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """Добавление опыта с максимальным приоритетом"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Выборка батча с учетом приоритетов"""
        if self.size == 0:
            return None
            
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
            torch.FloatTensor(weights),
            indices
        )
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Обновление приоритетов для выбранных опытов"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Добавляем маленькое значение чтобы избежать нуля

class NoisyLinear(nn.Module):
    """Шумный линейный слой для лучшего исследования"""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Сброс параметров"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Сброс шума"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Масштабирование шума"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с шумом"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class AdvancedDQN(nn.Module):
    """Продвинутая DQN сеть с шумными слоями и улучшенной архитектурой"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, noisy: bool = True):
        super(AdvancedDQN, self).__init__()
        
        LinearLayer = NoisyLinear if noisy else nn.Linear
        
        self.fc1 = LinearLayer(state_size, hidden_size)
        self.fc2 = LinearLayer(hidden_size, hidden_size)
        self.fc3 = LinearLayer(hidden_size, hidden_size // 2)
        
        # Dueling DQN
        self.value_stream = LinearLayer(hidden_size // 2, 1)
        self.advantage_stream = LinearLayer(hidden_size // 2, action_size)
        
        self.noisy = noisy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с dueling architecture"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Сброс шума во всех слоях"""
        if self.noisy:
            for module in [self.fc1, self.fc2, self.fc3, self.value_stream, self.advantage_stream]:
                if hasattr(module, 'reset_noise'):
                    module.reset_noise()

class AdvancedReinforcementLearner:
    """
    ПРОДВИНУТАЯ СИСТЕМА ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ
    С улучшенной стабильностью и скоростью обучения
    """
    
    def __init__(self, state_size: int = 8, action_size: int = 4, learning_rate: float = 0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Улучшенные нейронные сети
        self.policy_net = AdvancedDQN(state_size, action_size, noisy=True)
        self.target_net = AdvancedDQN(state_size, action_size, noisy=False)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Приоритизированный буфер воспроизведения
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        
        # Параметры обучения
        self.batch_size = 64
        self.gamma = 0.99  # Увеличенный коэффициент дисконтирования
        self.tau = 0.005  # Для мягкого обновления target сети
        self.learn_step_counter = 0
        self.update_target_every = 100
        
        # Трекеры для визуализации
        self.reward_history = []
        self.loss_history = []
        self.q_value_history = []
        
        # Загрузка предыдущего обучения
        self.load_model()
        
        print("🧠 Продвинутая RL система инициализирована")
        print(f"   - Dueling DQN с Noisy Layers")
        print(f"   - Prioritized Experience Replay")
        print(f"   - Double DQN + Soft Updates")
    
    def get_state_vector(self, game_state) -> np.ndarray:
        """Преобразование состояния игры в вектор для нейросети"""
        return np.array([
            game_state.hero_health,           # 0-1
            game_state.hero_mana,             # 0-1  
            game_state.hero_level / 25.0,     # Нормализованный уровень
            min(game_state.hero_gold / 10000.0, 1.0),  # Нормализованное золото
            game_state.nearby_enemies / 5.0,  # Нормализованные враги
            game_state.nearby_allies / 4.0,   # Нормализованные союзники
            game_state.game_time / 3600.0,    # Нормализованное время
            game_state.networth_advantage / 10000.0  # Нормализованное преимущество
        ], dtype=np.float32)
    
    def choose_action(self, state_vector: np.ndarray, available_actions: List[str]) -> int:
        """Выбор действия с использованием noisy DQN (без epsilon-greedy)"""
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        # Записываем Q значения для анализа
        self.q_value_history.append(q_values.max().item())
        if len(self.q_value_history) > 1000:
            self.q_value_history.pop(0)
        
        return np.argmax(q_values.numpy())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Сохранение опыта в приоритизированный буфер"""
        self.memory.add(state, action, reward, next_state, done)
        
        # Сохраняем награду для визуализации
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
    
    def compute_advanced_reward(self, game_state, action: str, success: bool, 
                              environment: Dict, previous_state = None) -> float:
        """Улучшенная функция вычисления награды"""
        reward = 0.0
        
        # Базовые награды за действия
        action_rewards = {
            "attack": 8.0 if success and environment.get('enemy_heroes', 0) > 0 else -3.0,
            "retreat": 12.0 if success and game_state.hero_health < 0.3 else -2.0,
            "farm": 6.0 if success and environment.get('farm_opportunity', 0) > 0.3 else -1.0,
            "objective": 15.0 if success else -2.0,
            "wait": -0.5
        }
        reward += action_rewards.get(action, 0.0)
        
        # Награда за выживание (очень важно)
        if game_state.hero_health > 0.9:
            reward += 5.0
        elif game_state.hero_health < 0.1:
            reward -= 20.0  # Большой штраф за смерть
        elif game_state.hero_health < 0.3:
            reward -= 5.0   # Штраф за низкое HP
        
        # Награда за прогресс
        if previous_state:
            # Улучшение характеристик
            health_improvement = game_state.hero_health - previous_state.hero_health
            gold_improvement = game_state.hero_gold - previous_state.hero_gold
            level_improvement = game_state.hero_level - previous_state.hero_level
            
            reward += health_improvement * 20.0
            reward += gold_improvement / 50.0
            reward += level_improvement * 10.0
        
        # Командные награды
        if environment.get('threat_level', 0) < 0.2:
            reward += 3.0  # Награда за безопасную позицию
        if environment.get('farm_opportunity', 0) > 0.7:
            reward += 4.0  # Награда за хорошие условия фарма
        
        # Награда за эффективность
        reward += min(game_state.hero_gold / 1000.0, 10.0)  # Награда за золото, но не более 10
        reward += game_state.hero_level * 2.0  # Награда за уровни
        
        # Штраф за бездействие в хороших условиях
        if action == "wait" and environment.get('farm_opportunity', 0) > 0.5:
            reward -= 3.0
        
        return reward
    
    def learn(self):
        """Улучшенное обучение с Double DQN и приоритизированным воспроизведением"""
        if self.memory.size < self.batch_size:
            return
        
        # Выборка из приоритизированного буфера
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return
            
        states, actions, rewards, next_states, dones, weights, indices = sample
        
        # Double DQN
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
        expected_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Текущие Q значения
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Вычисление потерь с весами
        td_errors = (expected_q_values - current_q_values).abs().detach().numpy()
        loss = (weights * F.mse_loss(current_q_values, expected_q_values, reduction='none')).mean()
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Обновление приоритетов
        self.memory.update_priorities(indices, td_errors)
        
        # Мягкое обновление target сети
        self.soft_update_target_network()
        
        # Сброс шума
        self.policy_net.reset_noise()
        
        # Сохранение статистики
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        
        self.learn_step_counter += 1
    
    def soft_update_target_network(self):
        """Мягкое обновление target сети"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str = "models/advanced_rl_model.pth"):
        """Сохранение обученной модели"""
        # Создаем папку если не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_history': self.reward_history,
            'loss_history': self.loss_history,
            'learn_step_counter': self.learn_step_counter
        }, filepath)
        print(f"💾 Продвинутая RL модель сохранена: {filepath}")
    
    def load_model(self, filepath: str = "models/advanced_rl_model.pth"):
        """Загрузка обученной модели"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.reward_history = checkpoint.get('reward_history', [])
            self.loss_history = checkpoint.get('loss_history', [])
            self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
            print(f"💾 Продвинутая RL модель загружена: {filepath}")
    
    def plot_learning_progress(self, save_path: str = "visualizations/learning_progress.png"):
        """Визуализация прогресса обучения"""
        if len(self.reward_history) < 10:
            return
        
        # Создаем папку если не существует
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # График наград
        ax1.plot(self.reward_history)
        ax1.set_title('История наград')
        ax1.set_xlabel('Шаг')
        ax1.set_ylabel('Награда')
        ax1.grid(True)
        
        # График потерь
        if self.loss_history:
            ax2.plot(self.loss_history)
            ax2.set_title('История потерь')
            ax2.set_xlabel('Шаг обучения')
            ax2.set_ylabel('Потери')
            ax2.grid(True)
        
        # График Q значений
        if self.q_value_history:
            ax3.plot(self.q_value_history)
            ax3.set_title('Максимальные Q значения')
            ax3.set_xlabel('Шаг')
            ax3.set_ylabel('Q значение')
            ax3.grid(True)
        
        # Статистика
        stats_text = f"""
        Статистика обучения:
        Шагов обучения: {self.learn_step_counter}
        Размер памяти: {self.memory.size}
        Средняя награда: {np.mean(self.reward_history[-100:]):.2f}
        Средние потери: {np.mean(self.loss_history[-100:]):.2f}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 График обучения сохранен: {save_path}")
    
    def get_learning_stats(self) -> Dict:
        """Получение расширенной статистики обучения"""
        recent_rewards = self.reward_history[-100:] if self.reward_history else [0]
        recent_losses = self.loss_history[-100:] if self.loss_history else [0]
        
        return {
            'learn_steps': self.learn_step_counter,
            'memory_size': self.memory.size,
            'avg_reward': np.mean(recent_rewards),
            'avg_loss': np.mean(recent_losses),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards) if recent_rewards else 0,
            'min_reward': np.min(recent_rewards) if recent_rewards else 0,
            'avg_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0
        }