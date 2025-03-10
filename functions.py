from os import path

import numpy as np
import polars as pl
import yaml


class Config:
    learning_rate: float
    batch_size: int  # Number of samples per batch
    test_batch_size: int  # Number of samples per batch
    epochs: int  # Total training epochs
    # optimizer: "ranger"       # Optimization algorithm
    dropout: float  # Dropout regularization rate
    weight_decay: float
    k: int
    ninp: int
    nlayers: int
    nclass: int
    ntoken: int  # AUGC + padding/N token
    nhead: int
    # use_bpp: False
    use_flip_aug: bool
    # bpp_file_folder: "../../input/bpp_files/"
    gradient_accumulation_steps: int
    use_triangular_attention: bool
    pairwise_dimension: int

    # Data scaling
    use_data_percentage: int
    use_dirty_data: bool  # turn off for data scaling and data dropout experiments

    # Other configurations
    fold: int
    nfolds: int
    # input_dir: "../../input/"
    input_dir: str
    gpu_id: str

    def __init__(self, **entries):
        """
        Initializes a configuration object with provided key-value settings.
        
        Updates the instance's attributes with the given keyword arguments and retains
        the original dictionary of entries in the 'entries' attribute.
        """
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        """
        Prints the configuration entries.
        
        Outputs the configuration settings stored in the instance's entries attribute to the console.
        """
        print(self.entries)


def drop_pk5090_duplicates(df):
    """
    Filters duplicate PK50 and PK90 dataset entries by retaining specific variants.
    
    This function processes a DataFrame with a 'dataset_name' column by removing general
    entries that start with "PK50" or "PK90" and retaining only non-PK entries alongside rows
    that specifically start with "PK50_AltChemMap_NovaSeq" and "PK90_Twist_epPCR". It also
    asserts that the counts of these variant entries match the expected numbers (2729*2 for PK50 
    and 2173*2 for PK90) before concatenating the subsets into a single DataFrame.
    
    Args:
        df: A DataFrame containing a 'dataset_name' column used for filtering.
    
    Returns:
        A concatenated DataFrame that combines non-PK entries with the filtered PK50 and PK90 variant entries.
    """
    pk50_filter = df["dataset_name"].str.starts_with("PK50")
    pk90_filter = df["dataset_name"].str.starts_with("PK90")
    no_pk_df = df.filter((~pk50_filter) & (~pk90_filter))
    pk50_df = df.filter(df["dataset_name"].str.starts_with("PK50_AltChemMap_NovaSeq"))
    pk90_df = df.filter(df["dataset_name"].str.starts_with("PK90_Twist_epPCR"))

    assert len(pk50_df) == 2729 * 2
    assert len(pk90_df) == 2173 * 2

    new_df = pl.concat([no_pk_df, pk50_df, pk90_df])

    return new_df


def dataset_dropout(dataset_name, train_indices, dataset2drop):

    # dataset_name=pl.Series(dataset_name)
    """
    Filters out training indices whose corresponding dataset names start with a specified prefix.
    
    This function converts the provided dataset names into a series and identifies the indices
    for which the names do not start with the given prefix. It then intersects these indices with
    the given training indices, printing counts before and after filtering, and returns an array
    of the filtered training indices.
    
    Args:
        dataset_name: An array-like sequence of strings representing dataset names.
        train_indices: An array-like collection of integer indices for training examples.
        dataset2drop: A string prefix. Training examples with dataset names starting with this prefix
            will be dropped.
    
    Returns:
        A NumPy array of training indices after filtering.
    """
    dataset_filter = pl.Series(dataset_name).str.starts_with(dataset2drop)
    dataset_filter = dataset_filter.to_numpy()

    dropout_indcies = set(np.where(dataset_filter == False)[0])
    # print(dropout_indcies)
    # exit()

    print(f"number of training examples before droppint out {dataset2drop}")
    print(train_indices.shape)
    before = len(train_indices)

    train_indices = set(train_indices).intersection(
        set(np.where(dataset_filter == False)[0])
    )
    train_indices = np.array(list(train_indices))

    print(f"number of training examples after droppint out {dataset2drop}")
    print(len(train_indices))
    after = len(train_indices)
    print(before - after, " sequences are dropped")

    # print(set([dataset_name[i] for i in train_indices]))
    # print(len(set([dataset_name[i] for i in train_indices])))
    # exit()

    return train_indices


def get_pl_train(pl_train, seq_length=457):

    """
    Extracts training data from a DataFrame by filtering unique sequences and formatting reactivity labels.
    
    The function removes duplicate records based on sequence identifiers and experiment types,
    then extracts unique sequences and their identifiers. Reactivity measurements are collected,
    reshaped into a two-channel array per sequence, and converted to float16. A corresponding
    error array is initialized to zeros, and signal-to-noise ratios are set uniformly to 10.
    
    Args:
        pl_train: A DataFrame containing training records with columns for sequence identifiers,
                  experiment types, sequences, reactivity measurements (prefixed with "reactivity_"),
                  and signal-to-noise ratios.
        seq_length: The expected sequence length used to generate reactivity label column names (default is 457).
    
    Returns:
        A dictionary with the following keys:
            "sequences": List of unique sequences.
            "sequence_ids": List of unique sequence identifiers.
            "labels": Numpy array of reactivity labels reshaped to (num_sequences, seq_length, 2) as float16.
            "errors": Numpy array of zeros with the same shape as "labels" as float16.
            "SN": Numpy array of signal-to-noise ratios with shape (-1, 2) as float16.
    """
    print(f"before filtering pl_train has shape {pl_train.shape}")
    pl_train = pl_train.unique(subset=["sequence_id", "experiment_type"]).sort(
        ["sequence_id", "experiment_type"]
    )
    print(f"after filtering pl_train has shape {pl_train.shape}")
    # seq_length=206

    label_names = [
        "reactivity_{:04d}".format(number + 1) for number in range(seq_length)
    ]
    error_label_names = [
        "reactivity_error_{:04d}".format(number + 1) for number in range(seq_length)
    ]

    sequences = pl_train.unique(subset=["sequence_id"], maintain_order=True)[
        "sequence"
    ].to_list()
    sequence_ids = pl_train.unique(subset=["sequence_id"], maintain_order=True)[
        "sequence_id"
    ].to_list()
    labels = (
        pl_train[label_names]
        .to_numpy()
        .astype("float16")
        .reshape(-1, 2, seq_length)
        .transpose(0, 2, 1)
    )
    errors = np.zeros_like(labels).astype("float16")
    SN = pl_train["signal_to_noise"].to_numpy().astype("float16").reshape(-1, 2)

    SN[:] = 10  # set SN to 10 so they don't get masked

    data_dict = {
        "sequences": sequences,
        "sequence_ids": sequence_ids,
        "labels": labels,
        "errors": errors,
        "SN": SN,
    }

    return data_dict


def load_config_from_yaml(file_path):
    """
    Load configuration settings from a YAML file.
    
    Opens the specified YAML file and safely loads its content as a dictionary,
    then creates and returns a Config object initialized with these settings.
    
    Args:
        file_path: The path to the YAML configuration file.
    
    Returns:
        A Config object populated with the configuration parameters from the YAML file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


def write_config_to_yaml(config, file_path):
    """
    Writes a configuration object to a YAML file.
    
    Opens the specified file in write mode and serializes the provided configuration
    using YAML's safe dump method. Any existing content at the file path is overwritten.
    
    Args:
        config: Configuration data to be serialized (typically a dict or Config instance).
        file_path: The target file path for the YAML output.
    """
    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)


def get_distance_mask(L):

    """
    Creates a distance mask matrix.
    
    Generates a square matrix of shape (L, L) where each diagonal entry is set to 1 and each off-diagonal
    entry is computed as the inverse of the square of the distance between the indices.
    
    Args:
        L: An integer specifying the size of the matrix.
    
    Returns:
        A numpy ndarray of shape (L, L) containing the distance-based weights.
    """
    m = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if abs(i - j) > 0:
                m[i, j] = 1 / abs(i - j) ** 2
            elif i == j:
                m[i, j] = 1
    return m


class CSVLogger:
    def __init__(self, columns, file):
        """
        Initializes a CSVLogger with the specified columns and file.
        
        This constructor sets up the CSVLogger by saving the provided header columns and file.
        It verifies whether the file already contains the CSV header by using the check_header method.
        If the header is absent, it writes the header using the _write_header method.
        
        Args:
            columns: A list of header names to be used in the CSV file.
            file: A file path or file-like object representing the CSV file.
        """
        self.columns = columns
        self.file = file
        if not self.check_header():
            self._write_header()

    def check_header(self):
        """
        Check if the CSV file exists.
        
        Returns:
            bool: True if the CSV file exists (indicating that a header has been written); otherwise, False.
        """
        if path.exists(self.file):
            header = True
        else:
            header = False
        return header

    def _write_header(self):
        """
        Write the CSV header row to the log file.
        
        Opens the file in append mode and writes a comma-separated line based on the instance's
        columns, ensuring the trailing comma is removed and a newline is appended. Returns the
        logger instance.
        """
        with open(self.file, "a") as f:
            string = ""
            for attrib in self.columns:
                string += "{},".format(attrib)
            string = string[: len(string) - 1]
            string += "\n"
            f.write(string)
        return self

    def log(self, row):
        """
        Appends a row of values to the CSV file.
        
        Validates that the row length matches the number of defined columns and
        formats the row as a comma-separated string before appending it to the file.
        Returns the logger instance for method chaining.
        
        Raises:
            Exception: If the length of the row does not match the number of columns.
        """
        if len(row) != len(self.columns):
            raise Exception(
                "Mismatch between row vector and number of columns in logger"
            )
        with open(self.file, "a") as f:
            string = ""
            for attrib in row:
                string += "{},".format(attrib)
            string = string[: len(string) - 1]
            string += "\n"
            f.write(string)
        return self


if __name__ == "__main__":
    print(load_config_from_yaml("configs/sequence_only.yaml"))
