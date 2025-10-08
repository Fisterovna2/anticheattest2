import struct
import time
import random
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    MOVE = 1
    ATTACK = 2
    CAST_SPELL = 3
    USE_ITEM = 4
    HOLD = 5
    STOP = 6

@dataclass
class GameCommand:
    action: ActionType
    target_x: float = 0
    target_y: float = 0
    target_entity: int = 0
    ability_slot: int = 0
    item_slot: int = 0

class HeroController:
    """
    РЕАЛЬНЫЙ контроллер для управления героем в Dota 2
    """
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.command_address = 0x12346000  # Адрес для отправки команд
        self.last_command_time = 0
        self.command_cooldown = 0.1  # 100ms между командами
        
        # Игровые константы
        self.map_width = 15000
        self.map_height = 15000
        
    def send_command(self, command: GameCommand) -> bool:
        """Отправка команды в игру"""
        try:
            current_time = time.time()
            if current_time - self.last_command_time < self.command_cooldown:
                return False  # Слишком рано для следующей команды
            
            # Формируем данные команды
            command_data = self._pack_command(command)
            
            # Отправляем команду в память игры
            success = self.memory.write_memory(self.command_address, command_data)
            
            if success:
                self.last_command_time = current_time
                print(f"🎯 Команда отправлена: {command.action.name}")
                return True
            else:
                print(f"❌ Ошибка отправки команды: {command.action.name}")
                return False
                
        except Exception as e:
            print(f"⚠️ Ошибка отправки команды: {e}")
            return False
    
    def _pack_command(self, command: GameCommand) -> bytes:
        """Упаковка команды в бинарный формат"""
        # Формат: [тип_действия][x][y][цель][способность][предмет]
        data = struct.pack(
            '<IffIII',  # unsigned int, 2x float, 3x unsigned int
            command.action.value,
            command.target_x,
            command.target_y, 
            command.target_entity,
            command.ability_slot,
            command.item_slot
        )
        return data
    
    def move_to_position(self, x: float, y: float) -> bool:
        """Движение к указанной позиции"""
        # Нормализуем координаты
        norm_x = max(0, min(x, self.map_width))
        norm_y = max(0, min(y, self.map_height))
        
        command = GameCommand(
            action=ActionType.MOVE,
            target_x=norm_x,
            target_y=norm_y
        )
        
        return self.send_command(command)
    
    def attack_entity(self, entity_id: int) -> bool:
        """Атака цели"""
        command = GameCommand(
            action=ActionType.ATTACK,
            target_entity=entity_id
        )
        
        return self.send_command(command)
    
    def cast_ability(self, ability_slot: int, target_x: float = 0, target_y: float = 0, target_entity: int = 0) -> bool:
        """Использование способности"""
        command = GameCommand(
            action=ActionType.CAST_SPELL,
            target_x=target_x,
            target_y=target_y,
            target_entity=target_entity,
            ability_slot=ability_slot
        )
        
        return self.send_command(command)
    
    def use_item(self, item_slot: int, target_x: float = 0, target_y: float = 0, target_entity: int = 0) -> bool:
        """Использование предмета"""
        command = GameCommand(
            action=ActionType.USE_ITEM,
            target_x=target_x,
            target_y=target_y,
            target_entity=target_entity,
            item_slot=item_slot
        )
        
        return self.send_command(command)
    
    def hold_position(self) -> bool:
        """Удержание позиции"""
        command = GameCommand(action=ActionType.HOLD)
        return self.send_command(command)
    
    def stop_actions(self) -> bool:
        """Остановка всех действий"""
        command = GameCommand(action=ActionType.STOP)
        return self.send_command(command)
    
    def get_safe_position(self, current_x: float, current_y: float) -> Tuple[float, float]:
        """Получение безопасной позиции для отступления"""
        # Простая логика отступления к базе
        base_x = 1000  # Примерные координаты базы
        base_y = 1000
        
        # Случайное смещение для естественности
        offset_x = random.uniform(-500, 500)
        offset_y = random.uniform(-500, 500)
        
        safe_x = base_x + offset_x
        safe_y = base_y + offset_y
        
        return safe_x, safe_y
    
    def get_aggressive_position(self, current_x: float, current_y: float, enemy_x: float, enemy_y: float) -> Tuple[float, float]:
        """Получение агрессивной позиции"""
        # Движение к врагу с небольшим отступом
        distance_x = enemy_x - current_x
        distance_y = enemy_y - current_y
        
        # Нормализуем расстояние
        length = (distance_x**2 + distance_y**2)**0.5
        if length > 0:
            distance_x /= length
            distance_y /= length
        
        # Подходим на 300 единиц к врагу
        approach_distance = 300
        target_x = enemy_x - distance_x * approach_distance
        target_y = enemy_y - distance_y * approach_distance
        
        return target_x, target_y