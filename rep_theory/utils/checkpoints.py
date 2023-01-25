import torch

def save_checkpoint(model, epoch, task_dir, final=False):
    """
    Save model checkpoint.
    """
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    torch.save(model.state_dict(), path)

def load_checkpoint(model, task_dir, epoch=None, final=False):
    """
    Load model checkpoint.
    """
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    model.load_state_dict(torch.load(path), strict=False)
    return model