# constrói o dataset de acordo com o 'dataset_file' informado
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data

    return None