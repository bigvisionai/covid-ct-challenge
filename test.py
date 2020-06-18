from dataset import CTData
from model import CovidCT
from config import config

import torch
from torch.utils.tensorboard import SummaryWriter
from utils import _test_model
import time
import torch.utils.data as data
import os

def test(config : dict):
    """
    Function where actual training takes place
    Args:
        config (dict) : Configuration to train with
    """
    
    print('Starting to Train Model...')

    val_data = CTData(type_data='test')
    val_loader = data.DataLoader(
        val_data, batch_size=4, num_workers=8, shuffle=False
    )

    print('Initializing Model...')
    model = CovidCT()

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss()
    val_criterion = torch.nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()

    state_dict = torch.load(open('./weights/model_test_val_auc_0.5693_train_auc_0.5895_epoch_15.pth','rb'))

    model.load_state_dict(state_dict['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()

    conf_matrix_val, val_loss, val_auc, val_acc = _test_model(
            model, val_loader, config['batch_size'], val_criterion)

    print("Confusion matrix here...")
    print(conf_matrix_val)
    
    print("val loss {:0.4f} | val auc {:0.4f} | val acc {:0.4f}".format(
            val_loss, val_auc, val_acc))


if __name__ == '__main__':

    print('Testing Configuration')
    print(config)

    test(config=config)

    print('Testing Ended...')
 