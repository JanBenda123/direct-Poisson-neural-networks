import os

def check_folder(name):
    """
    The function `check_folder` checks if the specified folder exists, and if not, creates it along with
    two subfolders named "data" and "saved_models".
    
    :param name: The `name` parameter is the name of the folder that you want to check and create if it doesn't exist
    """
    print("Checking folder ", name)
    name = os.getcwd()+"/"+name
    data_name = name+"/data"
    models_name = name+"/saved_models"
    graphics_name = name+"/graphics"
    if not os.path.exists(data_name):
        print("Making folder: "+data_name)
        os.makedirs(data_name)
    if not os.path.exists(models_name):
        print("Making folder: "+models_name)
        os.makedirs(models_name)
    if not os.path.exists(graphics_name):
        print("Making folder: "+graphics_name)
        os.makedirs(graphics_name)