
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateWSIDataLoader(opt):
    from data.custom_dataset_data_loader import WSIDatasetDataLoader
    data_loader = WSIDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader