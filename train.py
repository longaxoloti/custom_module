import os
import sys
import torch
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from . import data_setup, train_test, model_builder, utils

from torchvision import transforms

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "/data/pizza_steak_sushi/train"
test_dir = "/data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = data_transform,
    batch_size = BATCH_SIZE
)

model = model_builder.TinyVGG(
    input_shape = 3,
    hidden_units = HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_test.train(model = model,
             train_dataloader = train_dataloader,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = NUM_EPOCHS,
             device = device)
utils.save_model(model = model,
                 target_dir = "models",
                 model_name = "tinyvgg_model.pth")
print(f"Model trained and saved to 'models/tinyvgg_model.pth' with {NUM_EPOCHS} epochs.")