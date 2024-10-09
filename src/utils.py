import yaml

def read_yaml(file_path):
    """
    Read a YAML file and convert it to a Python dictionary.

    Parameters:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The YAML file content as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data