#!/usr/bin/env python3
"""
ULTIMATE AI BOT - ФИНАЛЬНАЯ ВЕРСИЯ СО ВСЕМИ УЛУЧШЕНИЯМИ
"""

import time
import json
import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Добавляем пути
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты наших модулей
from advanced_rl_learning import AdvancedReinforcementLearner
from hierarchical_ai import HierarchicalAI, StrategicGoal, TacticalAction
from learning_ai_brain import LearningAIBrain, GameState, HeroRole
from smart_targeting import SmartTargetSelector, GameEntity
from hero_controller import HeroController, GameCommand, ActionType
from real_memory_manager import RealMemoryManager
from working_skin_changer import WorkingSkinChanger
from enhanced_metrics import EnhancedMetricsCollector
from error_handler import handle_errors, safe_execute

class UltimateAIBot:
    """
    ULTIMATE AI БОТ - все улучшения в одной системе
    """
    
    def __init__(self):
        self.is_running = False
        self.components = {}
        self.config = self._load_config()
        
        print("🚀 ULTIMATE DOTA 2 AI BOT - ФИНАЛЬНАЯ ВЕРСИЯ")
        print("=" * 70)
        print("   🧠 Advanced DQN + Noisy Networks")
        print("   📊 Prioritized Experience Replay") 
        print("   🎯 Hierarchical Decision Making")
        print("   📈 Advanced Learning Visualization")
        print("=" * 70)
    
    def _load_config(self) -> dict:
        """Загрузка конфигурации"""
        default_config = {
            "bot": {
                "role": "carry",
                "aggressiveness": 0.7,
                "decision_interval": 2.0,
                "auto_skins": True
            },
            "steam": {
                "auto_restart": True,
                "restart_interval": 3600
            },
            "ai": {
                "learning_enabled": True,
                "adaptation_speed": 0.1
            },
            "security": {
                "max_operations_per_minute": 100,
                "emergency_shutdown": True
            },
            "dashboard": {
                "enabled": True,
                "port": 5000
            }
        }
        
        try:
            if os.path.exists("ultimate_config.json"):
                with open("ultimate_config.json", 'r') as f:
                    return json.load(f)
            else:
                with open("ultimate_config.json", 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            return default_config

    @handle_errors(max_retries=3, delay=2.0)
    def initialize_system(self) -> bool:
        """Инициализация ultimate AI системы"""
        print("🔧 Инициализация Ultimate AI системы...")
        
        try:
            # 1. Memory Manager
            self.components["memory"] = RealMemoryManager()
            print("✅ Memory Manager готов")
            
            # 2. Умный селектор целей
            self.components["target_selector"] = SmartTargetSelector(self.components["memory"])
            print("✅ Умный селектор целей готов")
            
            # 3. Иерархическая AI система
            role = HeroRole(self.config["bot"]["role"])
            self.components["hierarchical_ai"] = HierarchicalAI(
                self.components["memory"], 
                None,  # Metrics будет добавлен позже
                role
            )
            print("✅ Иерархическая AI система готова")
            
            # 4. Продвинутая RL система
            self.components["metrics"] = EnhancedMetricsCollector()
            self.components["ai"] = LearningAIBrain(
                self.components["memory"],
                self.components["metrics"], 
                role
            )
            
            # Замена RL системы на продвинутую версию
            self.components["ai"].rl_learner = AdvancedReinforcementLearner()
            print("✅ Продвинутая RL система готова")
            
            # 5. Hero Controller
            self.components["controller"] = HeroController(self.components["memory"])
            print("✅ Hero Controller готов")
            
            # 6. Skin Changer
            self.components["skins"] = WorkingSkinChanger(self.components["memory"])
            self.components["skins"].load_current_skins()
            print("✅ Skin Changer готов")
            
            # Запуск систем
            self.components["metrics"].start_collection()
            
            print("✅ Ultimate AI система готова к работе!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return False

    @handle_errors(max_retries=5, delay=5.0)
    def connect_to_game(self) -> bool:
        """Подключение к Dota 2"""
        print("🎯 Подключаемся к Dota 2...")
        
        memory = self.components["memory"]
        
        if not memory.connect_to_dota():
            print("❌ Не удалось подключиться к Dota 2")
            return False
        
        # Проверяем игровые модули
        if not memory.get_module_base("client.dll"):
            print("❌ Игровые модули не найдены")
            return False
        
        print("✅ Успешно подключились к Dota 2")
        return True

    def apply_ai_decision(self, decision: dict) -> bool:
        """Применение интеллектуального решения AI"""
        try:
            controller = self.components["controller"]
            target_selector = self.components["target_selector"]
            ai = self.components["ai"]
            
            action = decision["action"]
            game_state = ai.analyze_game_state()
            
            if action == "attack":
                # Умный выбор цели для атаки
                target = target_selector.select_attack_target(game_state.position, game_state.hero_health)
                if target:
                    return controller.attack_entity(target.entity_id)
                else:
                    # Нет целей - двигаемся к агрессивной позиции
                    aggressive_pos = controller.get_aggressive_position(
                        game_state.position[0], game_state.position[1],
                        game_state.position[0] + 500, game_state.position[1] + 500  # Пример
                    )
                    return controller.move_to_position(aggressive_pos[0], aggressive_pos[1])
            
            elif action == "retreat":
                # Умное отступление
                retreat_pos = target_selector.select_retreat_position(game_state.position)
                return controller.move_to_position(retreat_pos[0], retreat_pos[1])
            
            elif action == "farm":
                # Умный выбор локации для фарма
                farm_pos = target_selector.select_farm_location(game_state.position)
                return controller.move_to_position(farm_pos[0], farm_pos[1])
            
            elif action == "objective":
                # Движение к стратегической цели
                objective_x = random.uniform(5000, 10000)
                objective_y = random.uniform(5000, 10000)
                return controller.move_to_position(objective_x, objective_y)
            
            else:  # wait
                return controller.hold_position()
                
        except Exception as e:
            print(f"❌ Ошибка применения AI решения: {e}")
            return False

    def _apply_strategic_correction(self, rl_action: str, strategic_action: str, 
                                  goal: StrategicGoal, game_state: GameState) -> str:
        """Корректировка RL решения на основе стратегических целей"""
        
        # Критические ситуации переопределяют все
        if game_state.hero_health < 0.2:
            return "retreat"
        
        # Стратегические приоритеты
        if goal == StrategicGoal.DEFENSE and rl_action == "farm":
            return "attack"  # Защита важнее фарма
        
        if goal == StrategicGoal.PUSH and rl_action == "retreat":
            return "objective"  # Пуш важнее отступления (кроме критического HP)
        
        # В остальных случаях доверяем RL
        return rl_action

    def game_loop(self):
        """Улучшенный игровой цикл с иерархическим AI"""
        print("🎮 Запуск Ultimate AI цикла...")
        
        ai = self.components["ai"]
        hierarchical_ai = self.components["hierarchical_ai"]
        metrics = self.components["metrics"]
        
        cycle_count = 0
        last_visualization_time = time.time()
        previous_game_state = None
        
        while self.is_running:
            try:
                cycle_count += 1
                
                # 1. Анализ состояния игры
                game_state = ai.analyze_game_state()
                environment = self.components["target_selector"].get_environment_analysis(game_state.position)
                
                # 2. Иерархическое принятие решений
                strategic_goal = hierarchical_ai.update_strategic_goal(game_state, environment)
                tactical_action = hierarchical_ai.select_tactical_action(strategic_goal, game_state, environment)
                low_level_action = hierarchical_ai.get_low_level_action(tactical_action, game_state, environment)
                
                # 3. Принятие RL решения с учетом иерархии
                decision = ai.decide_next_action(game_state)
                
                # 4. Корректировка решения на основе стратегии (если нужно)
                final_action = self._apply_strategic_correction(
                    decision["action"], low_level_action, strategic_goal, game_state
                )
                
                # 5. Применение решения
                success = False
                if final_action != "wait":
                    success = self.apply_ai_decision(decision)
                    
                    if success:
                        ai.successful_actions += 1
                    else:
                        ai.failed_actions += 1
                
                # 6. Улучшенное вычисление награды
                if previous_game_state is not None:
                    reward = ai.rl_learner.compute_advanced_reward(
                        game_state, final_action, success, environment, previous_game_state
                    )
                    # Сохраняем для обучения (если есть последнее состояние)
                    if hasattr(ai, 'last_state_vector') and ai.last_state_vector is not None:
                        ai.rl_learner.remember(
                            ai.last_state_vector,
                            ai.last_action,
                            reward,
                            ai.rl_learner.get_state_vector(game_state),
                            False
                        )
                
                # 7. Детальное логирование
                metrics.record_vac_detection({
                    "cycle": cycle_count,
                    "final_action": final_action,
                    "rl_action": decision["action"],
                    "strategic_goal": strategic_goal.value,
                    "tactical_action": tactical_action.value,
                    "success": success,
                    "environment": environment,
                    "learning_stats": ai.get_learning_stats(),
                    "strategic_analysis": hierarchical_ai.get_strategic_analysis()
                })
                
                # 8. Периодические операции
                if cycle_count % 50 == 0:
                    ai.save_learning_progress()
                    
                    # Визуализация прогресса каждые 5 минут
                    current_time = time.time()
                    if current_time - last_visualization_time > 300:
                        ai.rl_learner.plot_learning_progress()
                        last_visualization_time = current_time
                
                # 9. Отчет о прогрессе
                if cycle_count % 30 == 0:
                    self._print_ultimate_progress(cycle_count, ai, hierarchical_ai)
                
                # 10. Сохраняем состояние для следующей итерации
                previous_game_state = game_state
                
                # 11. Задержка между решениями
                interval = self.config["bot"]["decision_interval"]
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ Ошибка в Ultimate AI цикле: {e}")
                time.sleep(5)

    def _print_ultimate_progress(self, cycle_count: int, ai: LearningAIBrain, hierarchical_ai: HierarchicalAI):
        """Вывод расширенного прогресса"""
        stats = ai.get_learning_stats()
        strategic_info = hierarchical_ai.get_strategic_analysis()
        
        print(f"📈 Ultimate AI - Цикл {cycle_count}:")
        print(f"   🎯 Стратегия: {strategic_info['strategic_goal']}")
        print(f"   ⚡ Тактика: {strategic_info['tactical_action']}")
        print(f"   🧠 Память RL: {stats['memory_size']} опытов")
        print(f"   💰 Средняя награда: {stats['avg_reward']:.2f}")
        print(f"   📊 Success Rate: {stats['success_rate']:.1%}")
        print(f"   🔥 Q-значения: {stats['avg_q_value']:.2f}")

    @safe_execute(default_value=False)
    def run(self) -> bool:
        """Запуск бота"""
        print("🚀 Запуск Ultimate Dota 2 AI Bot...")
        
        try:
            # 1. Инициализация
            if not self.initialize_system():
                return False
            
            # 2. Подключение к игре
            if not self.connect_to_game():
                return False
            
            # 3. Применение скинов
            if self.config["bot"]["auto_skins"]:
                self.components["skins"].apply_skin("Anti-Mage", "The Basher Blades")
                time.sleep(1)
            
            # 4. Запуск основного цикла
            self.is_running = True
            self.game_loop()
            
            return True
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            return False
        finally:
            self.shutdown()

    def shutdown(self):
        """Улучшенное завершение работы"""
        print("🔚 Завершение работы Ultimate AI...")
        self.is_running = False
        
        try:
            # Финальная визуализация прогресса
            if "ai" in self.components:
                self.components["ai"].rl_learner.plot_learning_progress("final_learning_progress.png")
                self.components["ai"].save_learning_progress()
                
                # Финальная статистика
                final_stats = self.components["ai"].get_learning_stats()
                strategic_info = self.components["hierarchical_ai"].get_strategic_analysis()
                
                print("📊 ФИНАЛЬНАЯ СТАТИСТИКА ULTIMATE AI:")
                print(f"   🎯 Финальная стратегия: {strategic_info['strategic_goal']}")
                print(f"   📈 Успешных действий: {final_stats['successful_actions']}")
                print(f"   💰 Итоговая награда: {final_stats['avg_reward']:.2f}")
                print(f"   🧠 Шагов обучения: {final_stats['learn_steps']}")
                print(f"   💾 Размер памяти: {final_stats['memory_size']}")
            
            # Остановка компонентов
            if "metrics" in self.components:
                self.components["metrics"].stop()
            
            if "memory" in self.components:
                self.components["memory"].close()
            
            print("✅ Ultimate AI бот корректно завершил работу")
            
        except Exception as e:
            print(f"⚠️ Ошибка при завершении: {e}")

def main():
    """Главная функция запуска"""
    bot = UltimateAIBot()
    
    try:
        success = bot.run()
        if success:
            print("🎉 Ultimate AI бот успешно завершил работу!")
            return 0
        else:
            print("❌ Ultimate AI бот завершил работу с ошибками")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️ Работа прервана пользователем")
        bot.shutdown()
        return 0
    except Exception as e:
        print(f"💥 Необработанная ошибка: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)