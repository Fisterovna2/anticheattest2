import math
import random
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import struct

@dataclass
class GameEntity:
    entity_id: int
    entity_type: str  # "hero", "creep", "tower", "roshan"
    position: Tuple[float, float]
    health: float
    team: str  # "ally", "enemy"
    level: int = 1
    gold_value: int = 0

class SmartTargetSelector:
    """
    УМНЫЙ СЕЛЕКТОР ЦЕЛЕЙ - анализирует окружение и выбирает оптимальные цели
    """
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.last_scan_time = 0
        self.scan_cooldown = 2.0  # Сканировать каждые 2 секунды
        self.detected_entities: List[GameEntity] = []
        
    def scan_environment(self, player_position: Tuple[float, float], scan_radius: float = 2000) -> List[GameEntity]:
        """Сканирование окружения на наличие сущностей"""
        current_time = time.time()
        if current_time - self.last_scan_time < self.scan_cooldown:
            return self.detected_entities
        
        try:
            # Эмуляция сканирования сущностей в радиусе
            # В реальной реализации здесь будет чтение данных из памяти игры
            
            entities = []
            
            # Пример: обнаружение вражеских героев
            for i in range(5):  # Максимум 5 вражеских героев
                # Эмуляция данных сущности
                pos_x = player_position[0] + random.uniform(-scan_radius, scan_radius)
                pos_y = player_position[1] + random.uniform(-scan_radius, scan_radius)
                health = random.uniform(0.1, 1.0)
                
                distance = self._calculate_distance(player_position, (pos_x, pos_y))
                if distance <= scan_radius:
                    entities.append(GameEntity(
                        entity_id=i,
                        entity_type="hero",
                        position=(pos_x, pos_y),
                        health=health,
                        team="enemy",
                        level=random.randint(1, 25)
                    ))
            
            # Обнаружение крипов
            for i in range(10):  # Крипы
                pos_x = player_position[0] + random.uniform(-scan_radius, scan_radius)
                pos_y = player_position[1] + random.uniform(-scan_radius, scan_radius)
                health = random.uniform(0.1, 1.0)
                
                distance = self._calculate_distance(player_position, (pos_x, pos_y))
                if distance <= scan_radius:
                    entities.append(GameEntity(
                        entity_id=100 + i,
                        entity_type="creep", 
                        position=(pos_x, pos_y),
                        health=health,
                        team="enemy",
                        gold_value=random.randint(20, 60)
                    ))
            
            self.detected_entities = entities
            self.last_scan_time = current_time
            
            print(f"🎯 Обнаружено сущностей: {len(entities)}")
            return entities
            
        except Exception as e:
            print(f"⚠️ Ошибка сканирования окружения: {e}")
            return self.detected_entities
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Вычисление расстояния между двумя точками"""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def select_attack_target(self, player_position: Tuple[float, float], player_health: float) -> Optional[GameEntity]:
        """Выбор оптимальной цели для атаки"""
        entities = self.scan_environment(player_position)
        enemy_entities = [e for e in entities if e.team == "enemy"]
        
        if not enemy_entities:
            return None
        
        # Система оценки целей
        scored_targets = []
        for entity in enemy_entities:
            score = self._calculate_target_score(entity, player_position, player_health)
            scored_targets.append((entity, score))
        
        # Выбор цели с наивысшим score
        if scored_targets:
            best_target = max(scored_targets, key=lambda x: x[1])
            return best_target[0]
        
        return None
    
    def _calculate_target_score(self, target: GameEntity, player_position: Tuple[float, float], player_health: float) -> float:
        """Вычисление score для цели"""
        score = 0.0
        
        # Расстояние до цели (ближе = лучше)
        distance = self._calculate_distance(player_position, target.position)
        distance_score = max(0, 1 - distance / 2000)  # Нормализация до 2000 единиц
        score += distance_score * 0.3
        
        # Здоровье цели (меньше здоровья = лучше)
        if target.entity_type == "hero":
            health_score = 1 - target.health
            score += health_score * 0.4
        else:  # Крипы и т.д.
            health_score = 1 - target.health
            score += health_score * 0.2
        
        # Ценность цели
        if target.entity_type == "hero":
            score += 0.5  # Герои - высокоприоритетные цели
        elif target.entity_type == "tower":
            score += 0.7  # Башни - очень важные цели
        elif target.gold_value > 0:
            score += target.gold_value / 100.0  # Крипы с золотом
        
        # Безопасность (избегаем целей когда у нас мало HP)
        if player_health < 0.3 and target.entity_type == "hero":
            score -= 0.5  # Штраф за атаку героев при низком HP
        
        return max(0, score)
    
    def select_farm_location(self, player_position: Tuple[float, float]) -> Tuple[float, float]:
        """Выбор оптимальной локации для фарма"""
        entities = self.scan_environment(player_position)
        farmable_entities = [e for e in entities if e.team == "enemy" and e.entity_type in ["creep"]]
        
        if farmable_entities:
            # Выбираем ближайшего крипа
            closest_creep = min(farmable_entities, 
                              key=lambda e: self._calculate_distance(player_position, e.position))
            return closest_creep.position
        else:
            # Возвращаем безопасную позицию для фарма
            return self._get_safe_farm_position(player_position)
    
    def _get_safe_farm_position(self, player_position: Tuple[float, float]) -> Tuple[float, float]:
        """Получение безопасной позиции для фарма"""
        # Логика выбора безопасной локации на карте
        safe_locations = [
            (2000, 2000),   # Лес радиантов
            (13000, 13000), # Лес диров
            (7000, 7000),   # Центр карты (осторожно)
        ]
        
        # Выбираем ближайшую безопасную локацию
        closest_safe = min(safe_locations, 
                         key=lambda loc: self._calculate_distance(player_position, loc))
        
        # Добавляем случайное смещение для естественности
        offset_x = random.uniform(-500, 500)
        offset_y = random.uniform(-500, 500)
        
        return (closest_safe[0] + offset_x, closest_safe[1] + offset_y)
    
    def select_retreat_position(self, player_position: Tuple[float, float]) -> Tuple[float, float]:
        """Выбор безопасной позиции для отступления"""
        entities = self.scan_environment(player_position)
        enemy_heroes = [e for e in entities if e.team == "enemy" and e.entity_type == "hero"]
        
        if enemy_heroes:
            # Определяем направление от врагов
            avg_enemy_x = sum(e.position[0] for e in enemy_heroes) / len(enemy_heroes)
            avg_enemy_y = sum(e.position[1] for e in enemy_heroes) / len(enemy_heroes)
            
            # Двигаемся в противоположном направлении
            direction_x = player_position[0] - avg_enemy_x
            direction_y = player_position[1] - avg_enemy_y
            
            # Нормализуем направление
            length = math.sqrt(direction_x**2 + direction_y**2)
            if length > 0:
                direction_x /= length
                direction_y /= length
            
            # Отступаем на 1000 единиц
            retreat_x = player_position[0] + direction_x * 1000
            retreat_y = player_position[1] + direction_y * 1000
            
            # Ограничиваем координаты картой
            retreat_x = max(500, min(retreat_x, 14500))
            retreat_y = max(500, min(retreat_y, 14500))
            
            return (retreat_x, retreat_y)
        else:
            # Возвращаемся к базе
            return (1000, 1000)
    
    def get_environment_analysis(self, player_position: Tuple[float, float]) -> Dict:
        """Анализ окружающей обстановки"""
        entities = self.scan_environment(player_position)
        
        analysis = {
            'total_entities': len(entities),
            'enemy_heroes': len([e for e in entities if e.team == 'enemy' and e.entity_type == 'hero']),
            'ally_heroes': len([e for e in entities if e.team == 'ally' and e.entity_type == 'hero']),
            'farmable_creeps': len([e for e in entities if e.team == 'enemy' and e.entity_type == 'creep']),
            'threat_level': self._calculate_threat_level(entities, player_position),
            'farm_opportunity': self._calculate_farm_opportunity(entities, player_position)
        }
        
        return analysis
    
    def _calculate_threat_level(self, entities: List[GameEntity], player_position: Tuple[float, float]) -> float:
        """Вычисление уровня угрозы"""
        enemy_heroes = [e for e in entities if e.team == 'enemy' and e.entity_type == 'hero']
        
        if not enemy_heroes:
            return 0.0
        
        total_threat = 0.0
        for hero in enemy_heroes:
            distance = self._calculate_distance(player_position, hero.position)
            distance_factor = max(0, 1 - distance / 1500)  # Угроза уменьшается с расстоянием
            threat = distance_factor * (1 - hero.health)  # Раненые герои менее опасны
            total_threat += threat
        
        return min(total_threat / len(enemy_heroes), 1.0)
    
    def _calculate_farm_opportunity(self, entities: List[GameEntity], player_position: Tuple[float, float]) -> float:
        """Вычисление возможности для фарма"""
        farmable = [e for e in entities if e.team == 'enemy' and e.entity_type == 'creep']
        
        if not farmable:
            return 0.0
        
        # Учитываем количество и близость крипов
        opportunity = 0.0
        for creep in farmable:
            distance = self._calculate_distance(player_position, creep.position)
            distance_factor = max(0, 1 - distance / 1000)
            opportunity += distance_factor
        
        return min(opportunity / len(farmable), 1.0)