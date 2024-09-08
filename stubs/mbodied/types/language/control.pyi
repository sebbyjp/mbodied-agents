from _typeshed import Incomplete
from enum import Enum

def create_enum_from_list(name, string_list) -> Enum:
    """Dynamically create an Enum type from a list of strings.

    Args:
        name (str): The name of the Enum class.
        string_list (list[str]): The list of strings to be converted into Enum members.

    Returns:
        Enum: A dynamically created Enum type with members based on the string list.
    """
def language_control_to_list(enum: Enum) -> list[str]:
    """Convert an Enum type to a list of its values. So it's easier to pass i.e. as prompt."""
def get_command_from_string(command_str) -> Enum:
    """Get the Enum member corresponding to a given command string."""

CommandControl: Incomplete
MobileControl: Incomplete
HeadControl: Incomplete
HandControl: Incomplete
MobileSingleArmControl: Incomplete
LangControl: Enum
MobileSingleArmLangControl: Enum
