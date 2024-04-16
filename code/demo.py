import torch
from PIL import Image
import open_clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def image_loader(root_dir, batch_size=32, train=True, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    print(dataset.classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader, dataset.classes

def eval(model, tokenizer, test_loader, classes):
    model.eval()

    for i in range(len(classes)):
        classes[i] = classes[i].replace('TV_Series', '')
        classes[i] = classes[i].replace('_', ' ')
    labelformat="a photo of the TV series {}"
    labels=[labelformat.format(c) for c in classes]
    text=tokenizer(labels).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    pred=[]
    gd=[]
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        print(f"processing  {batch_idx} batch..")
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred.append(text_probs)
        gd.append(targets)
    pred=torch.cat(pred, dim=0)
    gd=torch.cat(gd, dim=0)
    acc = (pred.argmax(dim=1) == gd).float().mean()
    print('Zero-shot accuracy of the network on the test images: %.2f %%' % ( 100 * acc))

def main():
    assert 0, "You should specify the folder of your dataset"
    test_dir = "[DATA-PATH]/test/"

    test_loader,classes = image_loader(test_dir, batch_size=128, train=False)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    model= model.cuda()
    
    eval(model, tokenizer, test_loader, classes)

main()





