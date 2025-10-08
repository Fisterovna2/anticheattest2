import logging
import time
from typing import Callable, Any, Optional
from functools import wraps
import traceback

class UniversalErrorHandler:
    """
    УНИВЕРСАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК
    """
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self.logger = logging.getLogger("ErrorHandler")
    
    def __call__(self, func: Callable) -> Callable:
        """Декоратор для обработки ошибок"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    self.logger.warning(
                        f"⚠️ Ошибка в {func.__name__} (попытка {attempt + 1}/{self.max_retries}): {e}"
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self.logger.error(
                            f"❌ Критическая ошибка в {func.__name__} после {self.max_retries} попыток"
                        )
                        self.logger.debug(f"Трассировка: {traceback.format_exc()}")
                        
            return None
        
        return wrapper
    
    def retry_on_failure(self, func: Callable) -> Callable:
        """Альтернативный синтаксис для декорирования"""
        return self.__call__(func)

# Создание глобального обработчика
global_error_handler = UniversalErrorHandler(max_retries=3, delay=1.0)

def handle_errors(max_retries: int = 3, delay: float = 1.0):
    """Фабрика декораторов для обработки ошибок"""
    return UniversalErrorHandler(max_retries, delay)

def safe_execute(default_value: Any = None):
    """Декоратор для безопасного выполнения с значением по умолчанию"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.getLogger("SafeExecute").warning(
                    f"Безопасное выполнение {func.__name__}: {e}"
                )
                return default_value
        return wrapper
    return decorator