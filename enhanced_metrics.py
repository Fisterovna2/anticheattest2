import time
import json
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
import psutil

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    disk_io: Dict
    network_io: Dict
    timestamp: float

@dataclass
class GameMetrics:
    fps: float
    ping: float
    packet_loss: float
    server_tick: int
    hero_actions_per_minute: float
    timestamp: float

class EnhancedMetricsCollector:
    """
    УЛУЧШЕННАЯ СИСТЕМА СБОРА МЕТРИК
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = {
            'system': deque(maxlen=max_history),
            'game': deque(maxlen=max_history),
            'vac_detections': deque(maxlen=100),
            'ai_decisions': deque(maxlen=500)
        }
        self.running = True
        self.collection_thread = None
        
    def start_collection(self):
        """Запуск сбора метрик в отдельном потоке"""
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        print("📊 Система сбора метрик запущена")
    
    def _collection_loop(self):
        """Цикл сбора метрик"""
        while self.running:
            try:
                # Сбор системных метрик
                system_metrics = self._collect_system_metrics()
                self.metrics_history['system'].append(system_metrics)
                
                # Сбор игровых метрик
                game_metrics = self._collect_game_metrics()
                if game_metrics:
                    self.metrics_history['game'].append(game_metrics)
                
                time.sleep(5)  # Сбор каждые 5 секунд
                
            except Exception as e:
                print(f"⚠️ Ошибка сбора метрик: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Сбор системных метрик"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_io=psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            network_io=psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            timestamp=time.time()
        )
    
    def _collect_game_metrics(self) -> Optional[GameMetrics]:
        """Сбор игровых метрик"""
        try:
            # В реальной реализации здесь будет чтение из памяти игры
            return GameMetrics(
                fps=120.0 + (time.time() % 10) - 5,  # Имитация колебаний FPS
                ping=45.0 + (time.time() % 5) - 2.5,  # Имитация пинга
                packet_loss=0.0,
                server_tick=128,
                hero_actions_per_minute=180.0,
                timestamp=time.time()
            )
        except Exception:
            return None
    
    def record_vac_detection(self, detection_data: Dict):
        """Запись детектирования VAC"""
        detection_data['timestamp'] = time.time()
        self.metrics_history['vac_detections'].append(detection_data)
        
        # Также сохраняем как AI решение для анализа
        if 'cycle' in detection_data:
            self.metrics_history['ai_decisions'].append(detection_data)
    
    def record_ai_decision(self, decision_data: Dict):
        """Запись AI решения"""
        decision_data['timestamp'] = time.time()
        self.metrics_history['ai_decisions'].append(decision_data)
    
    def get_risk_analysis(self) -> Dict:
        """Анализ рисков на основе метрик"""
        if not self.metrics_history['system']:
            return {'risk_level': 0, 'factors': []}
        
        recent_system = list(self.metrics_history['system'])[-10:]  # Последние 10 записей
        risk_factors = []
        risk_level = 0
        
        # Анализ загрузки CPU
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        if avg_cpu > 80:
            risk_factors.append(f"Высокая загрузка CPU: {avg_cpu:.1f}%")
            risk_level += 25
        
        # Анализ активности VAC процессов
        if self.metrics_history['vac_detections']:
            recent_detections = [d for d in self.metrics_history['vac_detections'] 
                               if time.time() - d['timestamp'] < 3600]
            if recent_detections:
                risk_factors.append(f"Детектирования VAC за час: {len(recent_detections)}")
                risk_level += min(len(recent_detections) * 10, 50)
        
        # Анализ стабильности FPS
        if self.metrics_history['game']:
            recent_fps = [m.fps for m in self.metrics_history['game'][-5:]]
            fps_std = statistics.stdev(recent_fps) if len(recent_fps) > 1 else 0
            if fps_std > 20:
                risk_factors.append(f"Нестабильный FPS: σ={fps_std:.1f}")
                risk_level += 15
        
        return {
            'risk_level': min(risk_level, 100),
            'factors': risk_factors,
            'timestamp': time.time()
        }
    
    def get_ai_performance_stats(self) -> Dict:
        """Статистика производительности AI"""
        if not self.metrics_history['ai_decisions']:
            return {}
        
        recent_decisions = list(self.metrics_history['ai_decisions'])[-100:]
        
        # Анализ успешности действий
        successful_actions = sum(1 for d in recent_decisions if d.get('success', False))
        total_actions = len(recent_decisions)
        
        # Анализ распределения действий
        action_counts = {}
        for decision in recent_decisions:
            action = decision.get('final_action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_decisions': total_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'action_distribution': action_counts,
            'avg_confidence': statistics.mean([d.get('confidence', 0) for d in recent_decisions]) if recent_decisions else 0,
            'recent_learning_rewards': sum(d.get('learning_reward', 0) for d in recent_decisions[-10:])
        }
    
    def export_metrics(self, filepath: str):
        """Экспорт метрик в JSON"""
        export_data = {
            'system_metrics': [asdict(m) for m in self.metrics_history['system']],
            'game_metrics': [asdict(m) for m in self.metrics_history['game']],
            'vac_detections': list(self.metrics_history['vac_detections']),
            'ai_decisions': list(self.metrics_history['ai_decisions']),
            'export_time': time.time(),
            'summary': {
                'risk_analysis': self.get_risk_analysis(),
                'ai_performance': self.get_ai_performance_stats()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"💾 Метрики экспортированы: {filepath}")
    
    def stop(self):
        """Остановка сбора метрик"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("📊 Система сбора метрик остановлена")