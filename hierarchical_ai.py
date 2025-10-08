from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import random

class StrategicGoal(Enum):
    """Стратегические цели высокого уровня"""
    EARLY_FARM = "early_farm"
    MID_GAME_PRESSURE = "mid_game_pressure"
    LATE_GAME_OBJECTIVES = "late_game_objectives"
    TEAM_FIGHT = "team_fight"
    DEFENSE = "defense"
    PUSH = "push"

class TacticalAction(Enum):
    """Тактические действия среднего уровня"""
    LANE_FARM = "lane_farm"
    JUNGLE_FARM = "jungle_farm"
    GANK = "gank"
    PUSH_LANE = "push_lane"
    DEFEND_TOWER = "defend_tower"
    ROSHAN = "roshan"
    RETREAT = "retreat"

@dataclass
class MacroAction:
    """Макро-действие - последовательность тактических действий"""
    name: str
    tactical_actions: List[TacticalAction]
    duration: float  # Примерная продолжительность
    priority: int

@dataclass  
class GameState:
    """Состояние игры для принятия решений"""
    hero_health: float
    hero_mana: float
    hero_level: int
    hero_gold: int
    position: Tuple[float, float]
    nearby_allies: int
    nearby_enemies: int
    game_time: float
    networth_advantage: float

class HeroRole(Enum):
    """Роли героев"""
    CARRY = "carry"
    MID = "mid"
    OFFLANE = "offlane"
    SUPPORT = "support"
    HARD_SUPPORT = "hard_support"

class HierarchicalAI:
    """
    ИЕРАРХИЧЕСКАЯ AI СИСТЕМА - многоуровневое принятие решений
    """
    
    def __init__(self, memory_manager, metrics_collector, role: HeroRole):
        self.memory = memory_manager
        self.metrics = metrics_collector
        self.role = role
        
        # Текущее состояние
        self.current_strategic_goal = StrategicGoal.EARLY_FARM
        self.current_tactical_action = TacticalAction.LANE_FARM
        self.current_macro_action = None
        
        # Время начала текущих действий
        self.goal_start_time = time.time()
        self.tactical_start_time = time.time()
        
        # Макро-действия для разных ролей
        self.macro_actions = self._initialize_macro_actions()
        
        # Статистика
        self.goal_success_rate = {}
        self.tactical_success_rate = {}
    
    def _initialize_macro_actions(self) -> Dict[HeroRole, List[MacroAction]]:
        """Инициализация макро-действий для разных ролей"""
        return {
            HeroRole.CARRY: [
                MacroAction(
                    name="Safe Farming",
                    tactical_actions=[TacticalAction.LANE_FARM, TacticalAction.JUNGLE_FARM],
                    duration=300.0,
                    priority=1
                ),
                MacroAction(
                    name="Objective Push", 
                    tactical_actions=[TacticalAction.PUSH_LANE, TacticalAction.ROSHAN],
                    duration=180.0,
                    priority=2
                )
            ],
            HeroRole.MID: [
                MacroAction(
                    name="Gank Pressure",
                    tactical_actions=[TacticalAction.LANE_FARM, TacticalAction.GANK],
                    duration=240.0,
                    priority=1
                )
            ],
            HeroRole.SUPPORT: [
                MacroAction(
                    name="Team Support",
                    tactical_actions=[TacticalAction.GANK, TacticalAction.DEFEND_TOWER],
                    duration=200.0,
                    priority=1
                )
            ]
        }
    
    def update_strategic_goal(self, game_state: GameState, environment: Dict) -> StrategicGoal:
        """Обновление стратегической цели на основе состояния игры"""
        game_time = game_state.game_time
        
        # Определение фазы игры
        if game_time < 900:  # 15 минут
            new_goal = StrategicGoal.EARLY_FARM
        elif game_time < 2700:  # 45 минут
            # В мидгейме выбираем цель на основе ситуации
            if environment.get('threat_level', 0) > 0.7:
                new_goal = StrategicGoal.DEFENSE
            elif game_state.networth_advantage > 2000:
                new_goal = StrategicGoal.PUSH
            else:
                new_goal = StrategicGoal.MID_GAME_PRESSURE
        else:  # Лейтгейм
            new_goal = StrategicGoal.LATE_GAME_OBJECTIVES
        
        # Если цель изменилась, обновляем время начала
        if new_goal != self.current_strategic_goal:
            self.current_strategic_goal = new_goal
            self.goal_start_time = time.time()
            print(f"🎯 Новая стратегическая цель: {new_goal.value}")
        
        return new_goal
    
    def select_tactical_action(self, strategic_goal: StrategicGoal, game_state: GameState, environment: Dict) -> TacticalAction:
        """Выбор тактического действия на основе стратегической цели"""
        
        if strategic_goal == StrategicGoal.EARLY_FARM:
            if environment.get('farm_opportunity', 0) > 0.4:
                new_action = TacticalAction.LANE_FARM
            else:
                new_action = TacticalAction.JUNGLE_FARM
                
        elif strategic_goal == StrategicGoal.MID_GAME_PRESSURE:
            if environment.get('enemy_heroes', 0) == 0:
                new_action = TacticalAction.PUSH_LANE
            else:
                new_action = TacticalAction.GANK
                
        elif strategic_goal == StrategicGoal.DEFENSE:
            new_action = TacticalAction.DEFEND_TOWER
            
        elif strategic_goal == StrategicGoal.PUSH:
            new_action = TacticalAction.PUSH_LANE
            
        else:  # LATE_GAME_OBJECTIVES
            # В лейтгейме фокусируемся на Рошане и целях
            new_action = TacticalAction.ROSHAN
        
        # Если тактическое действие изменилось, обновляем время начала
        if new_action != self.current_tactical_action:
            self.current_tactical_action = new_action
            self.tactical_start_time = time.time()
            print(f"   🎯 Новое тактическое действие: {new_action.value}")
        
        return new_action
    
    def get_low_level_action(self, tactical_action: TacticalAction, game_state: GameState, environment: Dict) -> str:
        """Преобразование тактического действия в низкоуровневое действие"""
        
        action_map = {
            TacticalAction.LANE_FARM: "farm",
            TacticalAction.JUNGLE_FARM: "farm", 
            TacticalAction.GANK: "attack",
            TacticalAction.PUSH_LANE: "objective",
            TacticalAction.DEFEND_TOWER: "attack",
            TacticalAction.ROSHAN: "objective",
            TacticalAction.RETREAT: "retreat"
        }
        
        return action_map.get(tactical_action, "wait")
    
    def evaluate_goal_success(self, game_state: GameState, previous_state: GameState) -> bool:
        """Оценка успешности текущей стратегической цели"""
        if self.current_strategic_goal == StrategicGoal.EARLY_FARM:
            # Успех: набрано достаточно золода и уровней
            gold_gain = game_state.hero_gold - (previous_state.hero_gold if previous_state else 0)
            level_gain = game_state.hero_level - (previous_state.hero_level if previous_state else 0)
            return gold_gain > 500 or level_gain > 1
            
        elif self.current_strategic_goal == StrategicGoal.PUSH:
            # Успех: уничтожена башня или достигнут прогресс
            return random.random() > 0.8  # Заглушка
            
        return True  # По умолчанию считаем успешным
    
    def get_strategic_analysis(self) -> Dict:
        """Получение стратегического анализа"""
        return {
            'strategic_goal': self.current_strategic_goal.value,
            'tactical_action': self.current_tactical_action.value,
            'goal_duration': time.time() - self.goal_start_time,
            'tactical_duration': time.time() - self.tactical_start_time,
            'role': self.role.value
        }