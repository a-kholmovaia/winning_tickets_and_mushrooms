from MushroomClassifier import MushroomClassifier
from MushroomDataset import MushroomDataSet
from Trainer import Trainer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

def plot_meta(meta_train, meta_pruned_final):
    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure()
    fig.tight_layout()


    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(meta_train.epoch, meta_train.val_loss, c='red', label="original")
    ax1.plot(meta_pruned_final.epoch, meta_pruned_final.val_loss, c='blue', label="pruned")
    plt.title("Val Loss")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(meta_train.epoch, meta_train.test_acc, c='red', label="original")
    ax1.plot(meta_pruned_final.epoch, meta_pruned_final.test_acc, c='blue', label="pruned")
    plt.title("Test Accuracy")
    plt.legend(loc='lower right')
    plt.savefig('res_base_vs_winning')


if __name__ == "__main__":
    train_dl = MushroomDataSet('train').dataloader
    val_dl = MushroomDataSet('val').dataloader
    model = MushroomClassifier()
    model.cuda()
    trainer = Trainer(model, train_dl=train_dl, val_dl=val_dl)
    trainer.find_winning_tickets(
        epochs_pretrain=15,
        percent_prune=0.01,
        pruning_rounds=60,
        epochs_posttrain=30
    )
    model_base= MushroomClassifier().cuda()
    model_base.load_state_dict(model.init_dict)
    trainer_base = Trainer(model_base, train_dl=train_dl, val_dl=val_dl)
    trainer_base.train(30)
    print(f'Sparcity of a base model: {model_base.get_sparcity()}')
    print(f'Sparcity of a pruned model: {model.get_sparcity()}')
    plot_meta(meta_train=trainer_base.meta_train,
              meta_pruned_final=trainer.meta_train)
    torch.save(model.state_dict(), "models/model_pruned")
    torch.save(model_base.state_dict(), "models/model_baseline")
    trainer.meta_train.to_csv('log_pruned.csv')
    trainer_base.meta_train.to_csv('log_base.csv')