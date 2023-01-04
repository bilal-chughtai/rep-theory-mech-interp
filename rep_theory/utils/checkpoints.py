import torch

def save_checkpoint(model, epoch, task_dir, final=False):
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    torch.save(model, path)

def load_checkpoint(task_dir, epoch=None, final=False):
    path = f'{task_dir}/checkpoints/epoch_{epoch}.pt'
    if final:
        path = f'{task_dir}/model.pt'
    model = torch.load(path)
    return model