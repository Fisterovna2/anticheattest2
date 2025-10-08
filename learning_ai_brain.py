import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

class GameState:
    """Состояние игры"""
    def __init__(self):
        self.hero_health = 100.0
        self.hero_mana = 100.0
        self.hero_level = 1
        self.hero_gold = 0
        self.position = (0, 0)
        self.nearby_allies = 0
        self.nearby_enemies = 0
        self.game_time = 0
        self.networth_advantage = 0

class HeroRole(Enum):
    CARRY = "carry"
    MID = "mid"
    OFFLANE = "offlane"
    SUPPORT = "support"

class LearningAIBrain:
    """
    AI С ОБУЧЕНИЕМ - учится на опыте и постоянно улучшает решения
    """
    
    def __init__(self, memory_manager, metrics_collector, role: HeroRole = HeroRole.CARRY):
        self.memory = memory_manager
        self.metrics = metrics_collector
        self.role = role
        
        # Система обучения с подкреплением (будет заменена на продвинутую)
        self.rl_learner = None  # Будет установлено извне
        
        # Отслеживание состояния для обучения
        self.last_state = None
        self.last_action = None
        self.last_state_vector = None
        
        # Статистика обучения
        self.learning_enabled = True
        self.total_learning_rewards = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.decision_history = deque(maxlen=1000)
        self.last_decision_time = 0
        
        print(f"🧠 Learning AI Brain инициализирован (роль: {role.value})")
    
    def analyze_game_state(self) -> GameState:
        """Анализ текущего состояния игры"""
        game_state = GameState()
        
        try:
            # Эмуляция чтения данных из игры
            # В реальной реализации здесь будет работа с memory_manager
            
            # Случайные данные для демонстрации
            game_state.hero_health = random.uniform(0.1, 1.0)
            game_state.hero_mana = random.uniform(0.1, 1.0)
            game_state.hero_level = random.randint(1, 25)
            game_state.hero_gold = random.randint(0, 10000)
            game_state.position = (random.uniform(0, 15000), random.uniform(0, 15000))
            game_state.nearby_allies = random.randint(0, 4)
            game_state.nearby_enemies = random.randint(0, 5)
            game_state.game_time = time.time() % 3600
            game_state.networth_advantage = random.uniform(-5000, 5000)
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа состояния игры: {e}")
        
        return game_state
    
    def decide_next_action(self, game_state: GameState) -> Dict:
        """Принятие решения с использованием обученной AI"""
        try:
            current_time = time.time()
            if current_time - self.last_decision_time < 1.0:
                return {"action": "wait", "reason": "cooldown"}
            
            # Анализ окружения (упрощенный)
            environment = {
                'threat_level': random.uniform(0, 1),
                'farm_opportunity': random.uniform(0, 1),
                'enemy_heroes': game_state.nearby_enemies
            }
            
            # Преобразование состояния в вектор для нейросети
            if self.rl_learner:
                state_vector = self.rl_learner.get_state_vector(game_state)
                
                # Выбор действия с помощью RL
                available_actions = ["attack", "retreat", "farm", "objective"]
                action_index = self.rl_learner.choose_action(state_vector, available_actions)
                rl_action = available_actions[action_index]
            else:
                # Заглушка если RL не инициализирован
                rl_action = random.choice(["attack", "retreat", "farm", "objective"])
                state_vector = np.zeros(8)
            
            # Сохраняем состояние для обучения
            if self.last_state_vector is not None and self.learning_enabled and self.rl_learner:
                # Вычисляем награду за предыдущее действие
                reward = self._compute_learning_reward(game_state, environment)
                self.rl_learner.remember(
                    self.last_state_vector, 
                    self.last_action, 
                    reward, 
                    state_vector, 
                    False
                )
                
                # Обучение на опыте
                self.rl_learner.learn()
                self.total_learning_rewards += reward
            
            # Обновляем последнее состояние
            self.last_state = game_state
            self.last_action = action_index if self.rl_learner else 0
            self.last_state_vector = state_vector
            
            # Принимаем тактическое решение с учетом окружения
            final_action = self._apply_tactical_rules(rl_action, game_state, environment)
            
            # Запись в историю
            decision_record = {
                "timestamp": current_time,
                "action": final_action,
                "rl_action": rl_action,
                "environment": environment,
                "confidence": 1.0 - (self.rl_learner.epsilon if self.rl_learner else 0.5),
                "learning_reward": self.total_learning_rewards
            }
            self.decision_history.append(decision_record)
            self.last_decision_time = current_time
            
            return {
                "action": final_action,
                "rl_action": rl_action,
                "reason": self._get_learning_action_reason(final_action, rl_action, environment),
                "confidence": (1.0 - (self.rl_learner.epsilon if self.rl_learner else 0.5)) * 100,
                "environment_analysis": environment,
                "learning_enabled": self.learning_enabled
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка принятия решения с обучением: {e}")
            return {"action": "wait", "reason": "error", "confidence": 0}
    
    def _compute_learning_reward(self, current_state: GameState, environment: Dict) -> float:
        """Вычисление награды для обучения на основе результатов действий"""
        if self.last_state is None or self.last_action is None:
            return 0.0
        
        # Базовые метрики прогресса
        health_improvement = current_state.hero_health - self.last_state.hero_health
        gold_improvement = current_state.hero_gold - self.last_state.hero_gold
        level_improvement = current_state.hero_level - self.last_state.hero_level
        
        reward = 0.0
        
        # Награда за выживание
        if current_state.hero_health > 0.1:  # Выжили
            reward += 5.0
        else:
            reward -= 20.0  # Смерть - большое наказание
        
        # Награда за прогресс
        reward += health_improvement * 10.0
        reward += gold_improvement / 10.0
        reward += level_improvement * 5.0
        
        # Награда за тактические успехи
        if environment['threat_level'] < 0.3:  # Безопасная обстановка
            reward += 3.0
        elif environment['threat_level'] > 0.7:  # Опасная обстановка
            reward -= 5.0
        
        if environment['farm_opportunity'] > 0.5:  # Хорошие возможности для фарма
            reward += 2.0
        
        return reward
    
    def _apply_tactical_rules(self, rl_action: str, game_state: GameState, environment: Dict) -> str:
        """Применение тактических правил к решению RL"""
        # Экстренные ситуации переопределяют решения RL
        if game_state.hero_health < 0.2:
            return "retreat"  # Принудительное отступление при критическом HP
        
        if environment['threat_level'] > 0.8 and game_state.nearby_allies < 2:
            return "retreat"  # Отступление при численном превосходстве врага
        
        if rl_action == "attack" and environment['enemy_heroes'] == 0:
            return "farm"  # Нет врагов для атаки - фармим
        
        if rl_action == "farm" and environment['farm_opportunity'] < 0.2:
            return "objective"  # Нет крипов - идем к цели
        
        return rl_action  # Используем решение RL
    
    def _get_learning_action_reason(self, final_action: str, rl_action: str, environment: Dict) -> str:
        """Обоснование решения с учетом обучения"""
        base_reasons = {
            "attack": "Атакую выбранную цель на основе тактического анализа",
            "retreat": "Отступаю для сохранения героя и перегруппировки", 
            "farm": "Фармлю ресурсы для усиления героя",
            "objective": "Захватываю стратегические цели",
            "wait": "Анализирую ситуацию перед следующим действием"
        }
        
        reason = base_reasons.get(final_action, "Принято решение на основе анализа ситуации")
        
        if final_action != rl_action:
            reason += f" (RL предложил {rl_action}, скорректировано тактическими правилами)"
        
        # Добавляем анализ окружения
        if environment['threat_level'] > 0.6:
            reason += " - Высокий уровень угрозы"
        elif environment['farm_opportunity'] > 0.7:
            reason += " - Благоприятные условия для фарма"
        
        return reason
    
    def get_learning_stats(self) -> Dict:
        """Получение расширенной статистики обучения"""
        total_actions = self.successful_actions + self.failed_actions
        
        base_stats = {
            'total_decisions': len(self.decision_history),
            'successful_actions': self.successful_actions,
            'failed_actions': self.failed_actions,
            'success_rate': self.successful_actions / total_actions if total_actions > 0 else 0,
            'learning_enabled': self.learning_enabled,
            'total_learning_rewards': self.total_learning_rewards
        }
        
        # Добавляем RL статистику если доступна
        if self.rl_learner:
            rl_stats = self.rl_learner.get_learning_stats()
            base_stats.update(rl_stats)
        
        return base_stats
    
    def save_learning_progress(self):
        """Сохранение прогресса обучения"""
        if self.rl_learner:
            self.rl_learner.save_model()
        
        # Сохранение дополнительных данных обучения
        learning_data = {
            'decision_history': list(self.decision_history),
            'successful_actions': self.successful_actions,
            'failed_actions': self.failed_actions,
            'total_learning_rewards': self.total_learning_rewards,
            'save_timestamp': time.time()
        }
        
        import json
        try:
            with open('models/learning_progress.json', 'w') as f:
                json.dump(learning_data, f, indent=2)
            print("💾 Прогресс обучения сохранен")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения прогресса обучения: {e}")
    
    def enable_learning(self, enabled: bool = True):
        """Включение/выключение обучения"""
        self.learning_enabled = enabled
        status = "включено" if enabled else "выключено"
        print(f"🎯 Обучение AI {status}")