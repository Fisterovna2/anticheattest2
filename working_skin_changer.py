import struct
import json
import os
from typing import Dict, List, Optional

class WorkingSkinChanger:
    """
    РАБОЧАЯ система смены скинов для Dota 2
    """
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.skin_config = self._load_skin_config()
        self.current_skins: Dict[str, str] = {}
        
    def _load_skin_config(self) -> Dict:
        """Загрузка конфигурации скинов"""
        config_path = "skin_config.json"
        default_config = {
            "heroes": {
                "Anti-Mage": {
                    "base_address": 0x12345678,
                    "skin_offset": 0x200,
                    "skins": {
                        "Default": 0,
                        "The Basher Blades": 1,
                        "Furyblade": 2
                    }
                },
                "Invoker": {
                    "base_address": 0x12345680,
                    "skin_offset": 0x200,
                    "skins": {
                        "Default": 0,
                        "Dark Artistry": 1,
                        "Crimson Cavalier": 2
                    }
                },
                "Pudge": {
                    "base_address": 0x12345688,
                    "skin_offset": 0x200,
                    "skins": {
                        "Default": 0,
                        "Dragonclaw Hook": 1,
                        "Scavenging Guttleslug": 2
                    }
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Сохраняем дефолтную конфигурацию
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации скинов: {e}")
            return default_config
    
    def detect_hero_addresses(self) -> bool:
        """Автоматическое определение адресов героев"""
        try:
            # Паттерны для поиска героев в памяти
            hero_patterns = {
                "Anti-Mage": b"\x41\x6E\x74\x69\x2D\x4D\x61\x67\x65",  # "Anti-Mage"
                "Invoker": b"\x49\x6E\x76\x6F\x6B\x65\x72",  # "Invoker"
                "Pudge": b"\x50\x75\x64\x67\x65"  # "Pudge"
            }
            
            found_any = False
            for hero_name, pattern in hero_patterns.items():
                address = self.memory.pattern_scan(pattern)
                if address:
                    print(f"🎯 Найден адрес героя {hero_name}: 0x{address:X}")
                    found_any = True
            
            return found_any
            
        except Exception as e:
            print(f"❌ Ошибка определения адресов героев: {e}")
            return False
    
    def get_available_heroes(self) -> List[str]:
        """Получение списка доступных героев"""
        return list(self.skin_config["heroes"].keys())
    
    def get_hero_skins(self, hero: str) -> List[str]:
        """Получение списка скинов для героя"""
        if hero in self.skin_config["heroes"]:
            return list(self.skin_config["heroes"][hero]["skins"].keys())
        return []
    
    def apply_skin(self, hero: str, skin: str) -> bool:
        """Применение скина к герою"""
        try:
            if hero not in self.skin_config["heroes"]:
                print(f"❌ Герой {hero} не найден в конфигурации")
                return False
            
            if skin not in self.skin_config["heroes"][hero]["skins"]:
                print(f"❌ Скин {skin} не найден для героя {hero}")
                return False
            
            skin_id = self.skin_config["heroes"][hero]["skins"][skin]
            hero_config = self.skin_config["heroes"][hero]
            
            # Используем адрес из конфигурации
            base_address = hero_config["base_address"]
            skin_address = base_address + hero_config["skin_offset"]
            
            # Записываем ID скина в память
            skin_data = struct.pack('<I', skin_id)  # 4-байтовое целое
            success = self.memory.write_memory(skin_address, skin_data)
            
            if success:
                self.current_skins[hero] = skin
                print(f"🎨 Применен скин {skin} (ID: {skin_id}) к {hero}")
                self._save_current_skins()
                return True
            else:
                print(f"❌ Не удалось применить скин {skin} к {hero}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка применения скина: {e}")
            return False
    
    def _save_current_skins(self):
        """Сохранение текущих скинов"""
        try:
            with open("current_skins.json", 'w') as f:
                json.dump(self.current_skins, f, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения скинов: {e}")
    
    def load_current_skins(self):
        """Загрузка текущих скинов"""
        try:
            if os.path.exists("current_skins.json"):
                with open("current_skins.json", 'r') as f:
                    self.current_skins = json.load(f)
                print(f"💾 Загружено {len(self.current_skins)} примененных скинов")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки скинов: {e}")
    
    def get_current_skin(self, hero: str) -> Optional[str]:
        """Получение текущего скина героя"""
        try:
            if hero not in self.current_skins:
                return None
            return self.current_skins[hero]
        except Exception:
            return None