import sys
from types import ModuleType

# Check if 'mbodied.data' already exists in sys.modules
if "mbodied.data" not in sys.modules:
    # Create a new module named 'mbodied.data'
    data = ModuleType("mbodied.data")

    # Set __path__ to simulate that 'mbodied.data' is a package
    data.__path__ = []  # This makes Python treat 'mbodied.data' as a package

    # Link the relevant modules from embdata
    import embdata.sample
    import embdata.episode
    import embdata.sense

    # Assign the submodules from embdata to mbodied.data
    data.sample = embdata.sample
    data.episode = embdata.episode
    data.sense = embdata.sense
    data.features = embdata.features
    # Register 'mbodied.data' in sys.modules
    sys.modules["mbodied.data"] = data
    sys.modules["mbodied.data.sample"] = embdata.sample
    sys.modules["mbodied.data.episode"] = embdata.episode
    sys.modules["mbodied.data.sense"] = embdata.sense
    sys.modules["mbodied.data.features"] = embdata.features
# Ensure 'data' is part of the exports
__all__ = ['data']

print("Initializing mbodied")