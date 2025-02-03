## Load the MNIST dataset


train_steps = 1200
eval_every = 200
batch_size = 32

## pytorch data loader 사용
## pytorch를 쓴다는 것은 아니고, pytorch data loader가 간편하기 때문

import numpy as np
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
from jax.tree_util import tree_map

# 1. 데이터 변환 함수 (torch DataLoader를 위한 collate_fn)
def numpy_collate(batch):
    """JAX에서 사용 가능하도록 NumPy 배열로 변환 (JAX tensor 변환은 train_step 내부에서)"""
    return tree_map(np.asarray, torch.utils.data._utils.collate.default_collate(batch))

# 2. JAX Dataset 클래스
class JAXMNISTDataset(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.MNIST(
            root='./data', train=train, download=True, transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image.numpy(), dtype=np.float32)  # (1, 28, 28)
        image = np.transpose(image, (1, 2, 0))  # (28, 28, 1)로 변환(채널을 마지막으로)
        return {
            'image': image,  # JAX 변환 X (DataLoader에서 처리)
            'label': np.array(label, dtype=np.int32)
        }
        
# 3. JAX DataLoader (PyTorch의 DataLoader 기반)
class JAXDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=True):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=numpy_collate  # NumPy 변환만 수행
        )

# 4. DataLoader 인스턴스 생성
train_loader = JAXDataLoader(JAXMNISTDataset(train=True), batch_size=batch_size, shuffle=True)
test_loader = JAXDataLoader(JAXMNISTDataset(train=False), batch_size=batch_size, shuffle=False)

# 5. 데이터 확인 (NumPy 형식 유지)
train_batch = next(iter(train_loader))
print(type(train_batch['image']), type(train_batch['label']))  # numpy.ndarray 확인
print(train_batch['image'].shape, train_batch['label'].shape)  # (32, 1, 28, 28), (32,)

# 6. train_step() 호출 전에 JAX 변환
def prepare_batch(batch):
    """JAX 모델에 넣기 전에 np.array -> jnp.array 변환"""
    return {k: jnp.array(v) for k, v in batch.items()}

## Define the model with Flax NNX

from flax import nnx # Flax NNX API
from functools import partial

class CNN(nnx.Module):
    '''
    Simple CNN model
    '''
    def __init__(self, *, rngs:nnx.Rngs): # nnx.Rngs: 모듈 내부에서 필요한 RNG 키를 자동으로 생성하고 관리
        self.conv1 = nnx.Conv(1, 32, kernel_size = (3,3), rngs = rngs) # 입력 채널 수 = 1, 출력 채널 수 = 32, 커널 크기 = (3,3)
        self.conv2 = nnx.Conv(32, 64, kernel_size = (3,3), rngs = rngs) # 입력 채널 수 = 32, 출력 채널 수 = 64, 커널 크기 = (3,3)
        self.avg_pool = partial(nnx.avg_pool, window_shape = (2,2), strides = (2,2))
        self.linear1 = nnx.Linear(3136, 256, rngs = rngs)
        self.linear2 = nnx.Linear(256, 10, rngs = rngs) # 출력 노드 수 = 10 (클래스 수)
        
    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x))) # ReLu + AvgPool
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape((x.shape[0], -1)) # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
# Instantiate the model
model = CNN(rngs = nnx.Rngs(0))

# Visualize
# nnx.display(model)

# import jax.numpy as jnp

# # 입력 데이터 전달
# y = model(jnp.ones((1, 28, 28, 1))) # (batch=1 한 개의 이미지 입력, height, width, channels=1 흑백)
# y

## Create the optimizer and define some metrics
import optax

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
    accuracy = nnx.metrics.Accuracy(),
    loss = nnx.metrics.Average('loss'),
)

# nnx.display(optimizer)

## Define training step functions
def loss_fn(model: CNN, batch):
    logits = model(batch['image']) # 모델에 이미지 배치를 입력하여 로짓(logits) 계산
    loss = optax.softmax_cross_entropy_with_integer_labels( # jnp.int32 를 요구
        logits = logits, labels = batch['label']
    ).mean() # 배치 내 모든 샘플의 손실 값을 평균
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
   ''' Train for a single step '''
   grad_fn = nnx.value_and_grad(loss_fn, has_aux=True) # 주어진 함수(loss_fn)의 출력값과 해당 출력값에 대한 그래디언트를 동시에 계산; has_aux=True -> 보조 출력값(loss, logits) 반환하도록 설정
   (loss, logits), grads = grad_fn(model, batch)
   metrics.update(loss = loss, logits = logits, labels = batch['label']) # In-place updates.
   optimizer.update(grads)

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss = loss, logits = logits, labels = batch['label']) # In-place updates.
    
## Train and evaluate the model
from IPython.display import clear_output
import matplotlib.pyplot as plt

metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
}

for step, batch in enumerate(iter(train_loader)): # training generator로 사용 (pytorch data loader 버전) - training_generator
    # Run the optimization for one step and make a stateful update for
    # - the train state's model parameters
    # - the optimizer state
    # - the training loss and accuracy batch metrics
    batch = prepare_batch(batch)  # JAX 모델에 넣기 전에 jnp 변환
    train_step(model, optimizer, metrics, batch)
    
    if step > 0 and (step % eval_every == 0 or step == train_steps - 1): # 1 에폭마다
        # log the training metrics
        for metric, value in metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        metrics.reset()
        
        # compute the metrics on the test set after each training epoch
        for test_batch in iter(test_loader):
            eval_step(model, metrics, test_batch)
        
        # log the test metrics
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        metrics.reset() # reset the metrics for the next training epoch

        clear_output(wait=True)
        # plot the loss and accuracy in subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title('Loss')
        ax2.set_title('Accuracy')
        for dataset in ('train', 'test'):
            ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        ax1.legend()
        ax2.legend()
        plt.show()
        
## Perform inference on the test set

model.eval() # evaluation mode.

@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)

test_batch = next(iter(test_loader))
pred = pred_step(model, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
  ax.set_title(f'label={pred[i]}')
  ax.axis('off')
