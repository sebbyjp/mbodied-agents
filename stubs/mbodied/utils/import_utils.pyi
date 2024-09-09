def reload(module: str) -> None:
    """Reload an existing module or import it if not already imported.

        If the specified module is already present in the global namespace,
        it will be reloaded. Otherwise, the module will be imported.

        Args:
            module (str): The name of the module to reload or import.

        Returns:
            None
    """
def smart_import(name: str, mode=...) -> module:
    '''Import a module with optional lazy loading.

        This function imports a module by name. If the module is already
        present in the global namespace, it will return the existing module.
        If the `mode` is set to "lazy", the module will be loaded lazily,
        meaning that the module will only be fully loaded when an attribute
        or function within the module is accessed.

        Args:
            name (str): The name of the module to import.
            mode (Literal["lazy"] | None, optional): If "lazy", the module will
                be imported using a lazy loader. Defaults to None.

        Returns:
            ModuleType: The imported module.

        Raises:
            NameError: If the module cannot be found or imported.
    '''
