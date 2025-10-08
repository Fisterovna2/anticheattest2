#!/usr/bin/env python3
"""
ФАЙЛ ЗАПУСКА ULTIMATE AI BOT
"""

import os
import sys
import time
import signal
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_bot.log'),
        logging.StreamHandler()
    ]
)

def setup_environment():
    """Настройка окружения"""
    print("🔧 Настройка окружения Ultimate AI Bot...")
    
    # Создание необходимых папок
    folders = ['models', 'visualizations', 'metrics', 'logs']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"   ✅ Создана папка: {folder}")
    
    # Проверка зависимостей
    try:
        import torch
        import numpy as np
        import psutil
        print("   ✅ Все зависимости доступны")
    except ImportError as e:
        print(f"   ❌ Отсутствует зависимость: {e}")
        return False
    
    return True

def main():
    """Главная функция запуска"""
    print("🎮 ULTIMATE DOTA 2 AI BOT - ЗАПУСК")
    print("=" * 70)
    
    # Настройка окружения
    if not setup_environment():
        print("❌ Не удалось настроить окружение")
        return 1
    
    try:
        # Импорт основного бота
        from ultimate_ai_bot import UltimateAIBot
        
        # Создание бота
        bot = UltimateAIBot()
        
        # Обработчик сигналов для graceful shutdown
        def signal_handler(sig, frame):
            print(f"\n🛑 Получен сигнал {sig}, завершаем работу...")
            bot.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Запуск бота
        print("🚀 Запускаем Ultimate AI Bot...")
        success = bot.run()
        
        if success:
            print("🎉 Ultimate AI Bot успешно завершил работу!")
            return 0
        else:
            print("❌ Ultimate AI Bot завершил работу с ошибками")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Работа прервана пользователем")
        return 0
    except Exception as e:
        print(f"💥 Необработанная ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)