import h5py

def load_h5(path, substruct=None):
    """
    Loading .h5 images

    ``substruct`` is the dataset path inside the file. When omitted, the
    file must hold exactly one dataset, which is used.
    """
    h5file = h5py.File(name=path, mode="r")
    if substruct is None:
        datasets = []
        h5file.visititems(
            lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        if len(datasets) != 1:
            raise ValueError(
                f"{path}: cannot pick an image dataset automatically, "
                f"found {len(datasets)}: {datasets[:10]}"
            )
        substruct = datasets[0]
    img = h5file[substruct][:]
    return(img)
