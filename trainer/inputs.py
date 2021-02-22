import datetime
import torch
from google.cloud import storage
from torch.utils.data import DataLoader
import datasets
from transformers import RobertaTokenizer


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        #return len(self.labels)
        return len(self.encodings.input_ids)


def load_data(args):
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    nli_data = datasets.load_dataset('multi_nli')

    # For testing purposes get a slammer slice of the training data
    all_examples = len(nli_data['train']['label'])
    num_examples = int(round(all_examples * args.fraction_of_train_data))

    print("Training with {}/{} examples.".format(num_examples, all_examples))
    
    train_dataset = nli_data['train'][:num_examples]

    dev_dataset = nli_data['validation_matched']
    test_dataset = nli_data['validation_matched']

    train_labels = train_dataset['label']

    val_labels = dev_dataset['label']
    test_labels = test_dataset['label']

    train_encodings = tokenizer(train_dataset['premise'], train_dataset['hypothesis'], truncation=True, padding=True)
    val_encodings = tokenizer(dev_dataset['premise'], dev_dataset['hypothesis'], truncation=True, padding=True)
    test_encodings = tokenizer(test_dataset['premise'], test_dataset['hypothesis'], truncation=True, padding=True)

    train_dataset = NLIDataset(train_encodings, train_labels)
    val_dataset = NLIDataset(val_encodings, val_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader


def save_model(args):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    scheme = 'gs://'
    bucket_name = args.job_dir[len(scheme):].split('/')[0]

    prefix = '{}{}/'.format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix):].rstrip('/')

    datetime_ = datetime.datetime.now().strftime('model_%Y%m%d_%H%M%S')

    if bucket_path:
        model_path = '{}/{}/{}'.format(bucket_path, datetime_, args.model_name)
    else:
        model_path = '{}/{}'.format(datetime_, args.model_name)

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(args.model_name)
