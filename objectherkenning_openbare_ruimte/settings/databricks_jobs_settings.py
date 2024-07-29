import yaml

def load_settings(file_path):
    """
    :param file_path: str, path to the config file
    :return: dict, parsed content as a dictionary
    """
    try:
        with open(file_path, 'r') as file:
            settings = yaml.safe_load(file)
        return settings
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    
def print_class_attributes(instance):
    """
    Prints all attributes of a class instance.

    :param instance: An instance of a class
    """
    for key, value in instance.__dict__.items():
        print(f"{key}: {value}")    