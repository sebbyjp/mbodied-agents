# This is a stub file for the gradio_example_openvla module
# It should not conflict with the actual module in examples/servers/gradio_example_openvla.py

from typing import Any

class OpenVLAInterface:
    def predict_action(self, image_base64: str, instruction: str, unnorm_key: Any = None, image_path: Any = None) -> Any: ...

def create_interface() -> Any: ...

# Add a note to clarify this is a stub file
__all__ = ['OpenVLAInterface', 'create_interface']
