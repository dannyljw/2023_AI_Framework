import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import wandb
import os
import argparse
import pickle
import time

#argparse 를 이용해 명령줄 인자를 받아온다. 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default =1, choices=[1, 2, 3, 4, 5], help='random seed')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'fashionmnist'], help='dataset')
parser.add_argument('--model', type=str, default='mobilenet_v2', choices=['mobilenet_v2', 'efficientnet_b0'], help='model')
parser.add_argument('--weight', type=str, default='xavier_uniform', choices=['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform',
                                                                             'm1_kaiming_normal','m1_kaiming_uniform', 'm1_xavier_normal', 'm1_xavier_uniform',
                                                                             'm2_kaiming_normal','m2_kaiming_uniform', 'm2_xavier_normal', 'm2_xavier_uniform',
                                                                             ], help='weight_init')
args = parser.parse_args()

# argparse의 인자들 출력
print("\033[91mUsing following command line arguments:\033[0m")  
for arg in vars(args):
    print(f"\033[91m{arg}: {getattr(args, arg)}\033[0m")
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.weight == 'kaiming_normal' or args.weight=='m1_kaiming_normal' or args.weight=='m2_kaiming_normal':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
elif args.weight=='kaiming_uniform' or args.weight=='m1_kaiming_uniform' or args.weight=='m2_kaiming_uniform':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
elif args.weight=='xavier_normal' or args.weight=='m1_xavier_normal' or args.weight=='m2_xavier_normal':
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
elif args.weight=='xavier_uniform' or args.weight=='m1_xavier_uniform' or args.weight=='m2_xavier_uniform':
    os.environ["CUDA_VISIBLE_DEVICES"]="3"


# seed 설정
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)  
    
# wandb 초기화
wandb.init(project="AI-framework_", name=f'{args.dataset}_{args.model}_{args.weight}_seed{args.seed}')

# 각 epoch의 결과를 저장
results = {
    'val_loss': [],
    'val_acc': [],
    'train_acc': [],
    'train_loss': [],
    'learning_rate': [],
}

# 데이터 전처리
if args.dataset =='cifar10':
    cifar10_n_100_train_transform = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        ),    
    ])

    cifar10_n_100_val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            ),    
    ])
    trainset = torchvision.datasets.CIFAR10(root='/workspace/6_AI_Framework', train=True, download=True, transform=cifar10_n_100_train_transform)
    testset = torchvision.datasets.CIFAR10(root='/workspace/6_AI_Framework', train=False, download=True, transform=cifar10_n_100_val_transform)

elif args.dataset =='cifar100':
    cifar10_n_100_train_transform = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        ),    
    ])

    cifar10_n_100_val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            ),    
    ])
    trainset = torchvision.datasets.CIFAR100(root='/workspace/6_AI_Framework', train=True, download=True, transform=cifar10_n_100_train_transform)
    testset = torchvision.datasets.CIFAR100(root='/workspace/6_AI_Framework', train=False, download=True, transform=cifar10_n_100_val_transform)
    
elif args.dataset =='fashionmnist':
    fashionmnist_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28),
        transforms.Grayscale(num_output_channels=3),  # 이 부분이 추가되었습니다
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
            ), 
            ])
    fashionmnist_val_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5],
        )
    ])
    trainset = torchvision.datasets.FashionMNIST(root='/workspace/6_AI_Framework', train=True, download=True, transform=fashionmnist_train_transform)
    testset = torchvision.datasets.FashionMNIST(root='/workspace/6_AI_Framework', train=False, download=True, transform=fashionmnist_val_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)


# --------------------------------MODEL
# mobilenet_v2 or efficientnet_b0
if args.model=='mobilenet_v2':
    if args.dataset=='cifar10' or args.dataset=='fashionmnist':
        model=models.mobilenet_v2(pretrained=False, num_classes=10)
    elif args.dataset=='cifar100':
        model=models.mobilenet_v2(pretrained=False, num_classes=100)
elif args.model=='efficientnet_b0':
    if args.dataset=='cifar10' or args.dataset=='fashionmnist':
        model=models.efficientnet_b0(pretrained=False, num_classes=10)
    elif args.dataset=='cifar100':
        model=models.efficientnet_b0(pretrained=False, num_classes=100)

# -----------------------------WEIGHT_INIT
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        if args.weight == 'kaiming_normal':
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='kaiming_uniform':
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='xavier_normal':
            init.xavier_normal_(m.weight),
        elif args.weight=='xavier_uniform':
            init.xavier_uniform_(m.weight),
        elif args.weight=='m1_kaiming_normal':
            init.m1_kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='m1_kaiming_uniform':
            init.m1_kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='m1_xavier_normal':
            init.m1_xavier_normal_(m.weight),
        elif args.weight=='m1_xavier_uniform':
            init.m1_xavier_uniform_(m.weight),
        elif args.weight=='m2_kaiming_normal':
            init.m2_kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='m2_kaiming_uniform':
            init.m2_kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu'),
        elif args.weight=='m2_xavier_normal':
            init.m2_xavier_normal_(m.weight),
        elif args.weight=='m2_xavier_uniform':
            init.m2_xavier_uniform_(m.weight),
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight,1)
        init.constant_(m.bias,0)
        
model.apply(weights_init)
model = model.cuda()

# 손실함수와 최적화함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# 학습률을 가져오는 함수
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 모델 학습
for epoch in range(400):
    start_time = time.time()
    
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # gradient 매개변수를 0으로 설정
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    print('[%d] Train Loss: %.3f | Train_Acc: %.3f | LR : %.3f' % (epoch + 1, running_loss / (i+1), 100.*correct/total, get_lr(optimizer)))
    wandb.log({"Train Loss": running_loss / (i+1), "Train Accuracy": 100.*correct/total, "Learning Rate": get_lr(optimizer)})
    
    #결과 저장
    results['train_acc'].append(running_loss / (i+1))   
    results['train_loss'].append(100.*correct/total)
    results['learning_rate'].append(get_lr(optimizer))

    # validation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    end_time = time.time()  # epoch 종료 시간 측정
    epoch_time = end_time - start_time  # 걸린 시간 = 종료 시간 - 시작 시간
    print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d) | time : %.3fsec' 
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch_time))
    print('time : %.3fsec' % (epoch_time))
    wandb.log({"Validation Accuracy": test_loss / (batch_idx+1), "Validation Loss": 100.*correct/total})
    #결과 저장
    results['val_acc'].append(test_loss / (batch_idx+1))
    results['val_loss'].append(100.*correct/total)  
    
    # 학습률 조정
    scheduler.step(test_loss)
    
    model.train()
    
#pickle 파일 저장
pickle_dir = "/workspace/6_AI_Framework/1207_method1/1207_pickle"
pickle_filename = f"1207_{pickle_dir}/{args.dataset}_{args.model}_{args.weight}_seed{args.seed}.pickle"
with open(pickle_filename, 'wb') as f:
    pickle.dump(results, f)


print('Finished Training')
