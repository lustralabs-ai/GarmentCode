import pkgutil, importlib
import pygarment
for m in pkgutil.walk_packages(pygarment.__path__, 'pygarment.'):
    try:
        importlib.import_module(m.name); print('OK  ', m.name)
    except BaseException as e:
        print('FAIL', m.name, type(e).__name__, e)
