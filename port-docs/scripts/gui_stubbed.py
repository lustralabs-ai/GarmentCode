"""Step 5: start gui.py with warp.sim modules stubbed (they don't exist yet) to verify the rest of the GUI binds its port.
Run from GarmentCode/. NOT a real sim; remove once warp.sim is ported."""
import sys, runpy
from unittest.mock import MagicMock
for m in ['warp.sim','warp.sim.render','warp.sim.utils','warp.sim.collide','warp.sim.integrator_xpbd','warp.collision','warp.collision.panel_assignment']:
    sys.modules[m] = MagicMock()
import warp; warp.sim = sys.modules['warp.sim']; warp.collision = sys.modules['warp.collision']
sys.path.insert(0, '.')
runpy.run_path('gui.py', run_name='__main__')
