import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from args import get_args
from models.unet import UNet
from dataset.datasetMAPF import MAPFDataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 训练函数
def train(args, model, train_loader, test_loader=None):
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=1e-5, momentum=0.999, foreach=True)
    criterion = nn.CrossEntropyLoss(reduction="none")
    global_step = 0
    model.to(device)
    
    train_losses = []  # 记录训练损失
    test_losses = []  # 记录测试损失
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        
        # 训练循环
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}"):
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)
            
            pred = model(feature)
            loss = criterion(pred, action_y)
            loss = (loss * mask.float()).mean()  # 修改为平均损失
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            global_step += 1
            epoch_loss += loss.item()
        
        # 记录训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}")
        
        # 在每个 epoch 后进行测试评估
        if test_loader:
            test_loss = evaluate(model, test_loader, criterion)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}, Test Loss: {test_loss:.4f}")
        
        # 每 n 个 epoch 可视化测试集上的行走路径
        if epoch % args.eval_every_n_epochs == 0:
            visualize_walk(model, test_loader, epoch)
    
    # 绘制训练损失和测试损失曲线
    plot_losses(train_losses, test_losses)
    
# 测试集评估函数
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            feature = batch["feature"].to(device)
            action_y = batch["action"].to(device)
            mask = batch["mask"].to(device)
            
            pred = model(feature)
            loss = criterion(pred, action_y)
            loss = (loss * mask.float()).mean()
            test_loss += loss.item()
    return test_loss / len(test_loader)


# 可视化智能体行走路径
def visualize_walk(model, test_loader, epoch):
    model.eval()
    batch = next(iter(test_loader))  # 获取一批测试数据
    feature = batch["feature"].to(device)
    action_y = batch["action"].to(device)
    
    with torch.no_grad():
        pred = model(feature).cpu().numpy()
    
    # 假设这里的 pred 包含智能体的动作信息
    # 我们可以根据 pred 来绘制智能体的路径
    # 在此处实现绘图
    plt.figure(figsize=(10, 10))
    plt.title(f"Epoch {epoch} - Agent Path")
    
    # 遍历智能体的预测路径并进行绘制
    for i in range(pred.shape[0]):  # 遍历每个样本
        path_x = []  # 用于存储路径的 x 坐标
        path_y = []  # 用于存储路径的 y 坐标
        # 这里假设 pred 包含智能体的动作，并且可以从中推导出路径
        # 您需要根据 pred 的具体格式来修改下面的代码
        # 示例代码仅作为占位符
        for t in range(pred.shape[1]):  # 遍历每个时间步
            x, y = pred[i, t, 0], pred[i, t, 1]  # 假设 (x, y) 是智能体的位置
            path_x.append(x)
            path_y.append(y)
        
        plt.plot(path_x, path_y, label=f"Agent {i+1}")
    
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"agent_path_epoch_{epoch}.png")  # 保存图像
    plt.show()

# 绘制训练和测试损失曲线
def plot_losses(train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    if test_losses:
        plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Test Loss")
    plt.show()


if __name__ == "__main__":
    args = get_args() 
    
    # the number of binary bits can be used to represent the number of agents
    args.feature_dim = int(np.ceil(np.log2(args.max_agent_num)))
    
    # every grid of a input map is represented by a feature vector with feature_dim*2+1 features
    # the first feature represents the existence of obstacle
    # the next feature_dim features represents the goal position of a specific agent
    # the next feature_dim features represents the start position of a specific agent
    net = UNet(args.feature_dim*2+1, args.action_dim)  
    
    train_data = MAPFDataset(args.dataset_path, args.feature_dim)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=args.batch_size, 
                              num_workers=1)
    
    # 假设有一个测试数据集
    test_data = MAPFDataset(args.test_dataset_path, args.feature_dim)
    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=1)
    
    train(args, net, train_loader, test_loader)