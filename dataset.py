import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# tokens='ACGU().BEHIMSX'


def load_bpp(filename, seq_length=177):
    """
    Initializes a square base pair probability matrix.

    This function creates and returns a two-dimensional NumPy array of zeros with shape
    (seq_length, seq_length). The 'filename' parameter is intended for a file containing
    base pair probability data, but the data processing is currently disabled, so the
    matrix serves as a placeholder.

    Args:
        filename: Path to the file with base pair probabilities (currently unused).
        seq_length: The size of one side of the square matrix (default is 177).

    Returns:
        A NumPy array of shape (seq_length, seq_length) filled with zeros.
    """
    matrix = [[0.0 for x in range(seq_length)] for y in range(seq_length)]
    #   #matrix=0
    # data processing
    #  for line in open(filename):
    #     line = line.strip()
    #    if line == "":
    #       break
    #  i,j,prob = line.split()
    # matrix[int(j)-1][int(i)-1] = float(prob)
    # matrix[int(i)-1][int(j)-1] = float(prob)

    matrix = np.array(matrix)

    # ap=np.array(matrix).sum(0)
    return matrix


class RNADataset(Dataset):
    def __init__(self, indices, data_dict, k=5, train=True, flip=False):
        """
        Initialize the RNA dataset with indices and associated sequence data.

        Args:
            indices: A collection of identifiers for RNA sequences.
            data_dict: A dictionary mapping each identifier to its RNA sequence data and metadata.
            k: An integer (default 5) that influences sequence encoding or masking.
            train: A boolean flag indicating whether the dataset is used for training, which may enable data augmentations.
            flip: A boolean flag that, if True, allows random sequence flipping during training.

        Also sets up a nucleotide token mapping that converts 'A', 'C', 'G', and 'U' to integers and assigns a special token 'P'.
        """
        self.indices = indices
        self.data_dict = data_dict
        self.k = k
        self.tokens = {nt: i for i, nt in enumerate("ACGU")}
        self.tokens["P"] = 4
        self.train = train
        self.flip = flip

    def generate_src_mask(self, L1, L2, k):
        """
        Generates a source mask for RNA sequence processing.

        This function creates a (k, L2) NumPy array of type int8 initialized with ones.
        For each row i, elements from the index (L1 + i + 1 - k) to the end are set to zero,
        adjusting the mask based on the provided sequence length offset and number of rows.

        Args:
            L1 (int): Baseline index used to compute the start of the zeroed region.
            L2 (int): Total number of columns in the mask.
            k (int): Number of rows in the mask and a factor for computing the offset.

        Returns:
            np.ndarray: A (k, L2) mask with elements set to 1 or 0.
        """
        mask = np.ones((k, L2), dtype="int8")
        for i in range(k):
            mask[i, L1 + i + 1 - k :] = 0
        return mask

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
            int: The total count of indices in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves and processes an RNA sequence sample with its associated metadata.

        This method selects a sample using an internal index mapping, converts the RNA
        sequence from nucleotide characters to numerical tokens based on a predefined
        mapping, and extracts the corresponding labels and error values, trimming them
        to the sequence length. NaN values in the labels and errors are replaced with zeros,
        and a loss mask is generated to indicate valid label positions. Label values are
        clipped between 0 and 1 before all data are converted to torch tensors. In training
        mode, if sequence flipping is enabled, the sequence, labels, and loss mask are randomly
        reversed to augment the data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - "sequence" (torch.LongTensor): Tokenized RNA sequence.
                - "labels" (torch.FloatTensor): Clipped label values.
                - "mask" (torch.Tensor): Mask indicating valid sequence positions.
                - "loss_mask" (torch.BoolTensor): Boolean mask identifying valid labels.
                - "errors" (torch.FloatTensor): Error values with NaNs replaced by zeros.
                - "signal_to_noise" (torch.FloatTensor): Signal-to-noise ratio.
        """
        idx = self.indices[idx]

        sequence = [self.tokens[nt] for nt in self.data_dict["sequences"][idx]]
        sequence = np.array(sequence)

        seq_length = len(sequence)

        # labels are in the order 2A3, DMS
        labels = self.data_dict["labels"][idx][:seq_length]
        errors = self.data_dict["errors"][idx][:seq_length]

        loss_mask = labels == labels  # mask nan labels
        # assert len(loss_mask)==
        # loss_mask[seq_length:]=0 #mask padding tokens

        label_mask = labels != labels

        labels[label_mask] = 0
        errors[errors != errors] = 0

        labels = labels.clip(0, 1)

        sequence = torch.tensor(sequence).long()
        labels = torch.tensor(labels).float()
        loss_mask = torch.tensor(loss_mask).bool()
        # mask=torch.tensor(self.src_masks[idx])
        mask = torch.ones(seq_length)
        # mask=torch.tensor(mask)
        errors = torch.tensor(errors).float()

        signal_to_noise = torch.tensor(self.data_dict["signal_to_noise"][idx]).float()

        if (self.train and np.random.uniform() > 0.5) and self.flip:
            sequence = sequence.flip(-1)
            # attention_mask=attention_mask.flip(-1).flip(-2)
            # mask=mask.flip(-1)
            labels = labels.flip(-2)
            loss_mask = loss_mask.flip(-2)

        data = {
            "sequence": sequence,
            "labels": labels,
            "mask": mask,
            "loss_mask": loss_mask,
            "errors": errors,
            "signal_to_noise": signal_to_noise,
        }

        return data


class TestRNAdataset(RNADataset):
    def __getitem__(self, idx):

        # id=self.ids[idx]

        # rows=self.df.loc[self.df['id']==id].reset_index(drop=True)
        # print()
        # idx=int(idx)
        # print(self.tokens)
        """
        Retrieves a tokenized RNA sequence and its mask.

        This method converts the nucleotide characters in the sequence at the given index into token indices using a predefined mapping. It then constructs a mask tensor of ones that matches the sequence length.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            dict: A dictionary containing:
                "sequence" - A PyTorch tensor (dtype long) with the tokenized RNA sequence.
                "mask" - A tensor of ones with the same length as the sequence.
        """
        sequence = [self.tokens[nt] for nt in self.data_dict["sequences"][idx]]
        sequence = np.array(sequence)

        seq_length = len(sequence)
        sequence = torch.tensor(sequence).long()
        mask = torch.ones(seq_length)
        # errors=torch.tensor(errors).float()

        id = self.data_dict["sequence_ids"][idx]
        # bpp=load_bpp(f"../../bpp_files_v2.0.3/{id}.txt",len(sequence))
        # bpp=torch.tensor(bpp).float()
        data = {
            "sequence": sequence,
            "mask": mask,
        }

        return data


class Custom_Collate_Obj:

    def __call__(self, data):
        """
        Collate and pad RNA sequence samples for batching.

        This function processes a list of sample dictionaries, each containing keys such as "sequence",
        "labels", "mask", "loss_mask", "errors", and "signal_to_noise". It computes the maximum sequence length in the batch
        and pads each field accordingly to ensure uniform tensor shapes. If the samples include base pair
        probability data under the key "bpp", that data is also padded and incorporated into the result.
        The output is a dictionary containing the stacked tensors and a tensor of the original sequence lengths.
        """
        length = []
        for i in range(len(data)):
            length.append(len(data[i]["sequence"]))
        max_len = max(length)

        sequence = []
        labels = []
        masks = []
        loss_masks = []
        errors = []
        signal_to_noise = []
        use_bpp = "bpp" in data[0]
        # print(use_bpp)
        # print(data['bpp'])
        if use_bpp:
            bpps = []
        for i in range(len(data)):
            to_pad = max_len - length[i]

            # if to_pad>0:
            sequence.append(F.pad(data[i]["sequence"], (0, to_pad), value=4))
            # masks.append(data[i]['mask'])
            masks.append(F.pad(data[i]["mask"], (0, to_pad), value=0))
            loss_masks.append(F.pad(data[i]["loss_mask"], (0, 0, 0, to_pad), value=0))
            # print(data[i]['labels'].shape)
            labels.append(F.pad(data[i]["labels"], (0, 0, 0, to_pad), value=0))
            errors.append(F.pad(data[i]["errors"], (0, 0, 0, to_pad), value=0))
            signal_to_noise.append(data[i]["signal_to_noise"])
            if use_bpp:
                bpps.append(F.pad(data[i]["bpp"], (0, to_pad, 0, to_pad), value=0))

        sequence = torch.stack(sequence)
        labels = torch.stack(labels)  # .permute(0,2,1)
        masks = torch.stack(masks)
        loss_masks = torch.stack(loss_masks)  # .permute(0,2,1)
        errors = torch.stack(errors)  # .permute(0,2,1)
        signal_to_noise = torch.stack(signal_to_noise)
        if use_bpp:
            bpps = torch.stack(bpps)
        # print(sequence.shape)
        # print(labels.shape)
        # exit()

        length = torch.tensor(length)

        data = {
            "sequence": sequence,
            "labels": labels,
            "masks": masks,
            "loss_masks": loss_masks,
            "errors": errors,
            "signal_to_noise": signal_to_noise,
            "length": length,
        }

        if use_bpp:
            data["bpps"] = bpps

        return data


class Custom_Collate_Obj_test(Custom_Collate_Obj):

    def __call__(self, data):
        """
        Collates and pads RNA sample data for batch processing.

        This method processes a list of dictionaries where each dictionary contains a "sequence" tensor and a
        "mask" tensor, with an optional "bpp" tensor for base pair probabilities. It computes the maximum sequence
        length in the batch and pads each "sequence" with the constant value 4 and each "mask" with 0 to match this
        length. If present, the "bpp" matrices are padded on both dimensions with zeros. Finally, the padded tensors are
        stacked and returned in a dictionary containing the batched "sequence", "masks", "length", and, if applicable, "bpps".
        """
        length = []
        for i in range(len(data)):
            length.append(len(data[i]["sequence"]))

        use_bpp = "bpp" in data[0]
        if use_bpp:
            bpps = []
        max_len = max(length)
        sequence = []
        masks = []
        for i in range(len(data)):
            to_pad = max_len - length[i]
            sequence.append(F.pad(data[i]["sequence"], (0, to_pad), value=4))
            masks.append(F.pad(data[i]["mask"], (0, to_pad), value=0))
            # masks.append(F.pad(data[i]['mask'],(0,to_pad,0,0),value=0))
            if use_bpp:
                bpps.append(F.pad(data[i]["bpp"], (0, to_pad, 0, to_pad), value=0))
        sequence = torch.stack(sequence)
        masks = torch.stack(masks)
        length = torch.tensor(length)

        data = {
            "sequence": sequence,
            "masks": masks,
            "length": length,
        }

        if use_bpp:
            bpps = torch.stack(bpps)
            data["bpps"] = bpps

        return data
