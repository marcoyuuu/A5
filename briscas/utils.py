"""
briscas/utils.py

Utilidades generales:
 - Logger simple por niveles
"""

import time


class Logger:
    """Imprime mensajes condicionalmente según nivel de detalle."""
    def __init__(self, level: int = 1):
        self.level = level

    def log(self, msg: str, lvl: int = 1):
        if self.level >= lvl:
            print(msg)


def tiempo_ejecucion(func):
    """
    Decorador para medir el tiempo de ejecución de una función.
    Devuelve (resultado, duración).
    """
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        return res, dt
    return wrapper
