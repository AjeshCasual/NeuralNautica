import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from terrain.visualise import PerlinExplorer

explorer = PerlinExplorer(x=0, y=0, z=0, rng=42, radius=15)
explorer.start()
