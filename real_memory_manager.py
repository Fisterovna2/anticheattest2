import ctypes
import struct
from ctypes import wintypes
import psutil
from typing import Optional, List
import time

class RealMemoryManager:
    """
    РЕАЛЬНЫЙ менеджер памяти для работы с Dota 2
    """
    
    def __init__(self):
        self.process_handle = None
        self.pid = None
        self.kernel32 = ctypes.windll.kernel32
        self.processed_addresses = set()
        
        # Настройка привилегий отладки
        self._setup_debug_privileges()
    
    def _setup_debug_privileges(self) -> bool:
        """Настройка привилегий SeDebugPrivilege"""
        try:
            ADVAPI32 = ctypes.windll.advapi32
            
            # Получаем текущий токен
            hToken = wintypes.HANDLE()
            ADVAPI32.OpenProcessToken(
                self.kernel32.GetCurrentProcess(),
                0x28,  # TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY
                ctypes.byref(hToken)
            )
            
            # Получаем LUID для SeDebugPrivilege
            luid = wintypes.LUID()
            ADVAPI32.LookupPrivilegeValueW(
                None,
                "SeDebugPrivilege",
                ctypes.byref(luid)
            )
            
            # Включаем привилегию
            new_state = (
                wintypes.DWORD * 3
            )(1, luid.LowPart, 0x2)  # SE_PRIVILEGE_ENABLED
            
            ADVAPI32.AdjustTokenPrivileges(
                hToken,
                False,
                ctypes.byref(new_state),
                0,
                None,
                None
            )
            
            return True
        except Exception as e:
            print(f"⚠️ Ошибка настройки привилегий: {e}")
            return False
    
    def find_dota_process(self) -> Optional[int]:
        """Поиск процесса Dota 2"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    if proc.info['name'] and 'dota2.exe' in proc.info['name'].lower():
                        print(f"🎯 Найден процесс Dota 2: PID {proc.info['pid']}")
                        return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None
        except Exception as e:
            print(f"❌ Ошибка поиска процесса: {e}")
            return None
    
    def connect_to_dota(self) -> bool:
        """Подключение к процессу Dota 2"""
        try:
            self.pid = self.find_dota_process()
            if not self.pid:
                print("❌ Процесс Dota 2 не найден")
                return False
            
            # PROCESS_ALL_ACCESS
            self.process_handle = self.kernel32.OpenProcess(
                0x1F0FFF,  # PROCESS_ALL_ACCESS
                False,
                self.pid
            )
            
            if not self.process_handle:
                print("❌ Не удалось открыть процесс Dota 2")
                return False
            
            print(f"✅ Успешно подключились к Dota 2 (PID: {self.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            return False
    
    def read_memory(self, address: int, size: int) -> bytes:
        """Чтение памяти процесса"""
        try:
            buffer = ctypes.create_string_buffer(size)
            bytes_read = ctypes.c_size_t()
            
            success = self.kernel32.ReadProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                buffer,
                size,
                ctypes.byref(bytes_read)
            )
            
            if success and bytes_read.value == size:
                self.processed_addresses.add(address)
                return buffer.raw
            else:
                return b'\x00' * size
                
        except Exception as e:
            print(f"⚠️ Ошибка чтения памяти 0x{address:X}: {e}")
            return b'\x00' * size
    
    def write_memory(self, address: int, data: bytes) -> bool:
        """Запись в память процесса"""
        try:
            bytes_written = ctypes.c_size_t()
            
            success = self.kernel32.WriteProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                data,
                len(data),
                ctypes.byref(bytes_written)
            )
            
            if success and bytes_written.value == len(data):
                self.processed_addresses.add(address)
                print(f"📝 Записано {len(data)} байт по адресу 0x{address:X}")
                return True
            else:
                print(f"❌ Ошибка записи по адресу 0x{address:X}")
                return False
                
        except Exception as e:
            print(f"⚠️ Ошибка записи памяти: {e}")
            return False
    
    def get_module_base(self, module_name: str) -> Optional[int]:
        """Получение базового адреса модуля"""
        try:
            from ctypes.wintypes import DWORD, HMODULE, MAX_PATH
            
            TH32CS_SNAPMODULE = 0x00000008
            TH32CS_SNAPMODULE32 = 0x00000010
            
            # Создаем снимок процессов
            hSnapshot = self.kernel32.CreateToolhelp32Snapshot(
                TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, 
                self.pid
            )
            
            if hSnapshot == -1:
                return None
            
            class MODULEENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", DWORD),
                    ("th32ModuleID", DWORD),
                    ("th32ProcessID", DWORD),
                    ("GlblcntUsage", DWORD),
                    ("ProccntUsage", DWORD),
                    ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
                    ("modBaseSize", DWORD),
                    ("hModule", HMODULE),
                    ("szModule", ctypes.c_char * (MAX_PATH + 1)),
                    ("szExePath", ctypes.c_char * (MAX_PATH + 1))
                ]
            
            entry = MODULEENTRY32()
            entry.dwSize = ctypes.sizeof(MODULEENTRY32)
            
            # Ищем модуль
            if self.kernel32.Module32First(hSnapshot, ctypes.byref(entry)):
                while True:
                    current_module = entry.szModule.decode('utf-8', errors='ignore')
                    
                    if module_name.lower() in current_module.lower():
                        base_address = ctypes.addressof(entry.modBaseAddr.contents)
                        self.kernel32.CloseHandle(hSnapshot)
                        print(f"📦 Найден модуль {current_module} по адресу 0x{base_address:X}")
                        return base_address
                    
                    if not self.kernel32.Module32Next(hSnapshot, ctypes.byref(entry)):
                        break
            
            self.kernel32.CloseHandle(hSnapshot)
            print(f"❌ Модуль {module_name} не найден")
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска модуля: {e}")
            return None
    
    def pattern_scan(self, pattern: bytes, module_name: str = "client.dll") -> Optional[int]:
        """Поиск паттерна в памяти модуля"""
        try:
            base_address = self.get_module_base(module_name)
            if not base_address:
                return None
            
            # Получаем размер модуля (примерно)
            module_size = 0x2000000  # 32MB для client.dll
            
            print(f"🔍 Сканируем {module_name} на паттерн...")
            
            # Сканируем чанками для эффективности
            chunk_size = 4096
            for offset in range(0, module_size, chunk_size):
                chunk = self.read_memory(base_address + offset, chunk_size)
                if not chunk:
                    continue
                
                # Ищем паттерн в чанке
                pos = chunk.find(pattern)
                if pos != -1:
                    found_address = base_address + offset + pos
                    print(f"🎯 Найден паттерн по адресу 0x{found_address:X}")
                    return found_address
            
            print("❌ Паттерн не найден")
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка сканирования паттерна: {e}")
            return None
    
    def close(self):
        """Закрытие handle процесса"""
        if self.process_handle:
            self.kernel32.CloseHandle(self.process_handle)
            self.process_handle = None
            print("🔒 Handle процесса закрыт")