# Vendor packages path setup — make vendored gigapath & diffusion_ffpe importable
import sys
from pathlib import Path
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))
