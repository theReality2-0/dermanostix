import os, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.manifold.t_sne as TSNE

np.random.seed(10)
torch.cuda.manual_seed(10)

main_dir = "../input/skin-cancer-mnist-ham10000"
image_paths1 = glob(os.path.join(main_dir, 'ham10000_images_part_1', '*.jpg'))
image_paths2 = glob(os.path.join(main_dir, 'ham10000_images_part_2', '*.jpg'))
image_paths = image_paths1 + image_paths2
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

norm_mean, norm_std = [0.763038, 0.54564667, 0.57004464], [0.14092727, 0.15261286, 0.1699712]

df_original = pd.read_csv(os.path.join(main_dir, 'HAM10000_metadata.csv'))
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type'], ordered=True).codes
print(df_original['cell_type_idx'].value_counts())

data_aug_rate = [20, 13, 6, 60, 0, 6, 45]
for i in range(7):
    if data_aug_rate[i]:
        df_original = df_original.append(
            [df_original.loc[df_original['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1), ignore_index=True)
print(df_original['cell_type_idx'].value_counts())


y = df_original['cell_type_idx']
df_train, df_val = train_test_split(df_original, test_size=0.25, random_state=101, stratify=y)
print("Train Dataset Length: " + str(len(df_train)))
print("Validation Dataset Length: " + str(len(df_val)))


df_train = df_train.reset_index()
df_val = df_val.reset_index()

num_classes = 7
input_size = 224
batch_size = 32

model_ft_1 = models.densenet121(pretrained=True)
num_ftrs = model_ft_1.classifier.in_features
model_ft_1.classifier = nn.Linear(num_ftrs, num_classes)
model_1 = model_ft_1

model_ft_3 = models.resnet50(pretrained=True)
num_ftrs = model_ft_3.fc.in_features
model_ft_3.fc = nn.Linear(num_ftrs, num_classes)
model_3 = model_ft_3

model_ft = models.vgg11_bn(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
model_2 = model_ft


class Ensemble(nn.module):
    def __init__(self, model_1, model_2, model_3):
        super(Ensemble, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3

    def forward(self, x):
        pred_1 = self.model_1(x)
        pred_2 = self.model_2(x)
        pred_3 = self.model_3(x)
        pred_sum = pred_1.add(pred_2)
        pred_sum = pred_sum.add(pred_3)
        avg = pred_sum / 3
        return avg, pred_1, pred_2, pred_3


model = Ensemble(model_1, model_2, model_3)

train_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                      transforms.Normalize(norm_mean, norm_std),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(),

                                      ])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])


class SpecializedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


training_set = SpecializedDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

validation_set = SpecializedDataset(df_val, transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=.001)
criterion = nn.CrossEntropyLoss()
epochs = 12
total_num_batches = len(train_loader)
step_size = 100


def top_n_accuracy(preds, labels, n, batch_size):
    label_list = np.zeros(shape=(n, batch_size))
    total_acc = 0
    for x in range(n):
        for image in range(batch_size):
            label_list[x][image] = (preds.topk(n)[1][image][x])
        pred_labels = torch.Tensor(label_list)
        total_acc = total_acc + (pred_labels[x].cpu() == labels.cpu()).float().sum()
    top_n_acc = total_acc / len(preds)
    return top_n_acc


ensemble_train_acc = []
densenet_train_acc = []
vgg_train_acc = []
resnet_train_acc = []

num_steps = []

for epoch in range(epochs):

    for batch, (image_batch, labels) in enumerate(train_loader):

        image_batch = Variable(image_batch)
        labels = Variable(labels)
        optimizer.zero_grad()

        all_preds = model(image_batch)
        ensemble_preds = all_preds[0]
        densenet_preds = all_preds[1]
        vgg_preds = all_preds[2]
        resnet_preds = all_preds[3]

        # Rate Testing

        # print(batch+1)

        loss = criterion(ensemble_preds, labels)
        loss.backward()

        optimizer.step()

        ensemble_accuracy = top_n_accuracy(ensemble_preds, labels, 1, len(image_batch))
        densenet_accuracy = top_n_accuracy(densenet_preds, labels, 1, len(image_batch))
        vgg_accuracy = top_n_accuracy(vgg_preds, labels, 1, len(image_batch))
        resnet_accuracy = top_n_accuracy(resnet_preds, labels, 1, len(image_batch))

        if (batch + 1) % step_size == 0:
            print('''Epoch: [{}] Batch: [{}/{}] Train Loss: [{:.4f}] 
        Ensemble Training Accuracy: [{:.4%}] 
        DenseNet Training Accuracy: [{:.4%}] 
        VGG Training Accuracy: [{:.4%}]
        ResNet Training Accuracy: [{:.4%}]
        '''.format(epoch + 1, batch + 1, total_num_batches, loss, ensemble_accuracy, densenet_accuracy, vgg_accuracy,
                   resnet_accuracy))

            num_steps.append(1)

            ensemble_train_acc.append(100 * ensemble_accuracy)
            densenet_train_acc.append(100 * densenet_accuracy)
            vgg_train_acc.append(100 * vgg_accuracy)
            resnet_train_acc.append(100 * resnet_accuracy)

            batch = 0

        if (len(image_batch) != batch_size):
            print('''Epoch: [{}] Batch: [{}/{}] Train Loss: [{:.4f}] 
        Ensemble Training Accuracy: [{:.4%}] 
        DenseNet Training Accuracy: [{:.4%}] 
        VGG Training Accuracy: [{:.4%}]
        ResNet Training Accuracy: [{:.4%}]
        '''.format(epoch + 1, batch + 1, total_num_batches, loss, ensemble_accuracy, densenet_accuracy, vgg_accuracy,
                   resnet_accuracy))

            batch = 0

print()
print()
print()

train_iterations = np.arange(1, len(num_steps) + 1)
plt.plot(num_steps, ensemble_train_acc, label='Ensemble')
plt.plot(num_steps, densenet_train_acc, label='DenseNet')
plt.plot(num_steps, vgg_train_acc, label='VGG')
plt.plot(num_steps, resnet_train_acc, label='ResNet')
plt.ylabel("Training Accuracy Percentage")
plt.xlabel("Step")
plt.legend(framealpha=1, frameon=True)
plt.show()

print()
print()
print()

y_label = []
y_predict = []
features_list = []
print("Validation")

ensemble_val_acc = []
densenet_val_acc = []
vgg_val_acc = []
resnet_val_acc = []

num_iters = []

with torch.no_grad():
    for batch, (image_batch, labels) in enumerate(val_loader):
        image_batch = Variable(image_batch)

        labels = Variable(labels)

        all_preds = model(image_batch)
        ensemble_preds = all_preds[0]
        densenet_preds = all_preds[1]
        vgg_preds = all_preds[2]
        resnet_preds = all_preds[3]

        ensemble_pred_labels = ensemble_preds.max(1, keepdim=True)[0]

        features = ensemble_preds.cpu().detach().numpy()
        features_list.extend(features)

        ensemble_accuracy = top_n_accuracy(ensemble_preds, labels, 1, len(image_batch))
        densenet_accuracy = top_n_accuracy(densenet_preds, labels, 1, len(image_batch))
        vgg_accuracy = top_n_accuracy(vgg_preds, labels, 1, len(image_batch))
        resnet_accuracy = top_n_accuracy(resnet_preds, labels, 1, len(image_batch))

        y_label.extend(labels.cpu().detach().numpy())
        y_predict.extend(np.squeeze(ensemble_pred_labels.cpu().detach().numpy()))

        print('''Ensemble Validation Accuracy: [{:.4%}]
        DenseNet Validation Accuracy: [{:.4%}]
        VGG Validation Accuracy: [{:.4%}]
        ResNet Validation Accuracy: [{:.4%}]'''.format(ensemble_accuracy, densenet_accuracy, vgg_accuracy,
                                                       resnet_accuracy))

        num_iters.append(batch + 1)

        ensemble_val_acc.append(100 * ensemble_accuracy)
        densenet_val_acc.append(100 * densenet_accuracy)
        vgg_val_acc.append(100 * vgg_accuracy)
        resnet_val_acc.append(100 * resnet_accuracy)

print()
print()
print()

fig, ax = plt.subplots()
plt.plot(num_iters, ensemble_val_acc, label='Ensemble')
plt.plot(num_iters, densenet_val_acc, label='DenseNet')
plt.plot(num_iters, vgg_val_acc, label='VGG')
plt.plot(num_iters, resnet_val_acc, label='ResNet')
ax.set_ylabel("Validation Accuracy Percentage")
ax.set_xlabel("Batch")
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()

features_array = np.array(features_list)
embeddings = TSNE(n_components=3, n_jobs=4).fit_transform(features_array)

vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
vis_z = embeddings[:, 2]

plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

scatter = ax.scatter(vis_x, vis_y, vis_z, c=y_label, cmap=plt.cm.get_cmap("jet", 7), marker='.')

ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")

ax.grid(False)

handles, labels = scatter.legend_elements()
leg1 = ax.legend(handles, plot_labels)
ax.add_artist(leg1)

plt.tight_layout()
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    # plt.xlabel('Predicted label')


confusion_mtx = confusion_matrix(y_label, y_predict)
plot_confusion_matrix(confusion_mtx, plot_labels)

report = classification_report(y_label, y_predict, target_names=plot_labels)

print()
print()
print()

print(report)

torch.save(model.state_dict(), "functioning.pth")