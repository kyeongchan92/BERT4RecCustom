from options import args
from model.bert import BERT
from dataset.dataset import ML1MDataset
from dataloader.dataloader import BertDataloader
from trainer.trainer import BERTTrainer
from utils import *

if __name__ == '__main__':
    export_root = setup_train(args)

    dataset = ML1MDataset(args)

    dataloader = BertDataloader(args, dataset)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

    model = BERT(args, dataloader.i2attr_map)

    trainer = BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()